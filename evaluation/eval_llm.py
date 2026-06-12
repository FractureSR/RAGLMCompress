"""LLM compression evaluation — text.

Usage examples
--------------
# enwiki (HF dataset saved to disk, 'text' field)
python evaluation/eval_llm.py \\
    --dataset  datasets/enwiki \\
    --model    pretrained/SmolLM2-135M \\
    --n-docs 500 \\
    --device cuda:0 \\
    --output results/llm_enwiki.csv

# cosmopedia
python evaluation/eval_llm.py \\
    --dataset  datasets/cosmopedia-100k \\
    --model    pretrained/SmolLM2-135M \\
    --n-docs 100 \\
    --device cuda:0,cuda:1 \\
    --output results/llm_cosmopedia.csv

# Compression only (skip decode), save compressed artefacts
python evaluation/eval_llm.py --dataset ... --no-decompress --save-compressed results/comp

# New dataset: register a loader in utils/text_utils.py with @register_text_loader("name")
"""

from __future__ import annotations
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.eval_utils import (
    EvalResult, EvalStats, MemoryTracker,
    auto_batch_size,
    parse_devices, run_multi_gpu,
    save_csv, save_compressed, save_decompressed,
)
from utils.text_utils import (
    load_text_documents,
    chunk_documents_for_compression, pad_token_ids,
)
from compression.llm_compressor import LLMCompressor


# ---------------------------------------------------------------------------
# Model loader (module-level → picklable for mp.spawn)
# ---------------------------------------------------------------------------

def _load_model(model_path: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16,
    ).to(device).eval()
    return m, tok


# ---------------------------------------------------------------------------
# Worker (module-level → picklable)
# ---------------------------------------------------------------------------

def _text_worker(
    device: torch.device,
    indices: List[int],
    model_path: str,
    texts: List[str],
    no_decomp: bool,
    max_tokens: Optional[int],
    save_comp_dir: Optional[str],
    save_decomp_dir: Optional[str],
) -> List[EvalResult]:
    import tqdm

    model, tokenizer = _load_model(model_path, device)
    comp = LLMCompressor(model, tokenizer, device=device)
    max_tok = max_tokens or tokenizer.model_max_length
    pad_id = tokenizer.pad_token_id

    # ── 1. Preprocessing: split documents that exceed the context window ──────
    # This is a pure data step — no compression logic involved.
    worker_texts = [texts[i] for i in indices]
    all_chunks = chunk_documents_for_compression(
        worker_texts, tokenizer, max_tok)
    chunk_lens = [len(c.token_ids) for c in all_chunks]

    # ── 2. Auto-select batch size ─────────────────────────────────────────────
    def _probe(batch_size: int, seq_len: int) -> None:
        dummy = torch.zeros(batch_size, seq_len,
                            dtype=torch.long, device=device)
        with torch.inference_mode():
            model(dummy, use_cache=False)
        del dummy

    sample_lens = chunk_lens[:200]
    batch_size = auto_batch_size(
        _probe, device, sample_lens, max_batch=256, verbose=True)
    n_split = sum(1 for c in all_chunks if c.total_chunks > 1)
    print(f"  [{device}] batch_size={batch_size}  docs={len(indices)}  "
          f"chunks={len(all_chunks)}  split={n_split}")

    # ── 3. Bucket sort chunks by token length ─────────────────────────────────
    sorted_order = sorted(range(len(all_chunks)), key=lambda k: chunk_lens[k])

    # ── 4. Compress (and optionally decompress) all chunks ────────────────────
    # chunk position → (cd, roundtrip_ok, compress_s, decompress_s, mem)
    chunk_results: Dict[int, tuple] = {}

    def _process_chunk_batch(batch_positions: List[int]) -> None:
        batch_chunks = [all_chunks[p] for p in batch_positions]
        input_ids, attn_mask = pad_token_ids(
            [c.token_ids for c in batch_chunks], pad_id, device=device,
        )
        with MemoryTracker(device) as mem:
            t0 = time.time()
            cds = comp.compress_batch(input_ids, attn_mask)
            compress_s = time.time() - t0

        per_c_s = compress_s / len(batch_chunks)

        recs = None
        per_d_s = -1.0
        if not no_decomp:
            t0 = time.time()
            recs = comp.decompress_batch(cds, show_progress=True)
            per_d_s = (time.time() - t0) / len(batch_chunks)

        for local_i, (chunk, cd) in enumerate(zip(batch_chunks, cds)):
            rt_ok = -1
            if recs is not None:
                rec = recs[local_i]
                rt_ok = int(rec[0].cpu().tolist() == chunk.token_ids)
                if save_decomp_dir:
                    tag = f"doc{indices[chunk.doc_idx]:06d}_c{chunk.chunk_idx}"
                    rec_text = tokenizer.decode(rec[0].cpu().tolist(),
                                                skip_special_tokens=True)
                    save_decompressed(rec_text.encode(),
                                      save_decomp_dir, tag, "txt")
            if save_comp_dir:
                tag = f"doc{indices[chunk.doc_idx]:06d}_c{chunk.chunk_idx}"
                save_compressed(cd.compressed_bytes, cd.metadata,
                                cd.original_length, save_comp_dir, tag)
            chunk_results[batch_positions[local_i]] = (
                cd, rt_ok, per_c_s, per_d_s, mem)

    pbar = tqdm.tqdm(total=len(all_chunks), desc=f"Text [{device}]",
                     position=getattr(device, "index", 0) or 0)
    cursor = 0
    effective_bs = batch_size
    while cursor < len(sorted_order):
        batch_pos = sorted_order[cursor:cursor + effective_bs]
        try:
            _process_chunk_batch(batch_pos)
            cursor += len(batch_pos)
            pbar.update(len(batch_pos))
            effective_bs = batch_size
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if effective_bs == 1:
                raise
            effective_bs = max(1, effective_bs // 2)
            print(f"\n  OOM — retrying with batch_size={effective_bs}")
    pbar.close()

    # ── 5. Aggregate chunk results per original document ──────────────────────
    doc_chunk_pos: Dict[int, List[int]] = defaultdict(list)
    for pos, chunk in enumerate(all_chunks):
        doc_chunk_pos[chunk.doc_idx].append(pos)

    results = []
    for local_idx, global_idx in enumerate(indices):
        text = worker_texts[local_idx]
        orig_b = len(text.encode())
        positions = doc_chunk_pos[local_idx]      # already in chunk order
        doc_data = [chunk_results[p] for p in positions]

        comp_b = sum(d[0].compressed_length for d in doc_data)
        rt_ok = (1 if all(d[1] == 1 for d in doc_data) else
                 -1 if all(d[1] == -1 for d in doc_data) else 0)
        c_s = sum(d[2] for d in doc_data)
        d_s_vals = [d[3] for d in doc_data if d[3] >= 0]
        d_s = sum(d_s_vals) if d_s_vals else -1.0
        mem = doc_data[0][4]

        results.append(EvalResult(
            sample_id=f"doc{global_idx:06d}",
            original_bytes=orig_b,
            compressed_bytes=comp_b,
            bpb=comp_b * 8 / max(orig_b, 1),
            ratio=orig_b / max(comp_b, 1),
            compress_s=c_s,
            decompress_s=d_s,
            peak_gpu_mb=mem.peak_gpu_mb,
            peak_ram_mb=mem.peak_ram_mb,
            roundtrip_ok=rt_ok,
        ))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLM compression evaluation — text")
    p.add_argument("--dataset",    required=True,
                   help="Dataset path (must match a registered text loader name)")
    p.add_argument("--model",      required=True,
                   help="Model name or local path (AutoModelForCausalLM)")
    p.add_argument("--n-docs",     type=int,  default=None,
                   help="Max documents (default: all)")
    p.add_argument("--max-tokens", type=int,  default=None,
                   help="Truncate each doc to this many tokens (default: model max)")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Comma-separated devices, e.g. cuda:0,cuda:1")
    p.add_argument("--save-compressed",   default=None, metavar="DIR")
    p.add_argument("--save-decompressed", default=None, metavar="DIR")
    p.add_argument("--output",     default=None, metavar="CSV")
    p.add_argument("--no-decompress", action="store_true")
    p.add_argument("--tmp-dir", default="tmp",
                   help="Directory for inter-process temp files (default: tmp)")
    return p


def main():
    args = _build_parser().parse_args()
    args.model = os.path.normpath(args.model)
    devices = parse_devices(args.device)

    texts = load_text_documents(args.dataset, num_documents=args.n_docs)
    print(f"Loaded {len(texts)} documents | devices: {devices}")

    indices = list(range(len(texts)))
    results = run_multi_gpu(
        _text_worker, indices, devices,
        fn_kwargs=dict(
            model_path=args.model, texts=texts,
            no_decomp=args.no_decompress,
            max_tokens=args.max_tokens,
            save_comp_dir=args.save_compressed,
            save_decomp_dir=args.save_decompressed,
        ),
        tmp_prefix=os.path.join(args.tmp_dir, '_eval_worker')
    )

    stats = EvalStats()
    for r in results:
        stats.update(r)
    stats.print_summary(label="LLM text")

    if args.output:
        save_csv(results, args.output)


if __name__ == "__main__":
    main()
