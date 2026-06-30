"""Oracle RAC compression evaluation — text."""
from __future__ import annotations

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from typing import Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.llm_compressor import LLMCompressor
from compression.rac_compressor import RACCompressor
from compression.rac_index import CalibratedIndexCoder, FixedIndexCoder, load_index_coder
from utils.eval_utils import (
    EvalResult, EvalStats,
    auto_batch_size,
    parse_devices, run_multi_gpu,
    save_csv,
)
from utils.text_utils import (
    chunk_documents_for_compression,
    load_text_documents,
    make_text_retriever,
)


# ---------------------------------------------------------------------------
# Model / database loading
# ---------------------------------------------------------------------------

def _load_model(model_path: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16,
    ).to(device).eval()
    return model, tok


def _load_database(database_dir: str, signals: str, embed_model: str,
                   device: torch.device):
    with open(os.path.join(database_dir, "base_chunks.json")) as f:
        base = json.load(f)
    base_tokens = [b["token_ids"] for b in base]
    if not base_tokens:
        raise ValueError(f"No base chunks found in {database_dir}/base_chunks.json")

    retriever = make_text_retriever(embed_model, str(device), signals=signals)
    retriever.load(os.path.join(database_dir, "retriever"))
    return base_tokens, retriever


def _prefix_metrics(doc_data: List[tuple]) -> dict:
    hist: Dict[int, int] = defaultdict(int)
    gain_sums: Dict[int, float] = defaultdict(float)
    net_gain_sums: Dict[int, float] = defaultdict(float)
    id_bit_sums: Dict[int, float] = defaultdict(float)
    gain_counts: Dict[int, int] = defaultdict(int)

    for cd, *_ in doc_data:
        n_prefix = len(cd.metadata.get("ctx_ids", []))
        hist[n_prefix] += 1
        gains = cd.metadata.get("ctx_gain_bits", [])
        net_gains = cd.metadata.get("ctx_net_gain_bits", [])
        id_bits = cd.metadata.get("ctx_id_bits", [])
        for pos, gain in enumerate(gains, start=1):
            gain_sums[pos] += float(gain)
            net_gain_sums[pos] += float(net_gains[pos - 1])
            id_bit_sums[pos] += float(id_bits[pos - 1])
            gain_counts[pos] += 1

    return dict(
        prefix_hist=dict(hist),
        prefix_gain_sums=dict(gain_sums),
        prefix_net_gain_sums=dict(net_gain_sums),
        prefix_id_bit_sums=dict(id_bit_sums),
        prefix_gain_counts=dict(gain_counts),
    )


def _merge_numeric_dict(dst: Dict[int, float], src: Dict[int, float]) -> None:
    for k, v in src.items():
        dst[int(k)] += v


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _rac_worker(
    device: torch.device,
    indices: List[int],
    model_path: str,
    texts: List[str],
    database_dir: str,
    signals: str,
    embed_model: str,
    m: int,
    max_tokens: int,
    cfg: dict,
    index_path: Optional[str],
    no_decomp: bool,
) -> List[dict]:
    model, tokenizer = _load_model(model_path, device)
    llm = LLMCompressor(model, tokenizer, device=device)
    base_tokens, retriever = _load_database(database_dir, signals, embed_model, device)

    index_coder = load_index_coder(index_path) if index_path else FixedIndexCoder(len(base_tokens))
    rac = RACCompressor(
        llm, base_tokens,
        index_coder=index_coder,
        max_ctx=cfg["max_ctx"],
        margin_bits=cfg["margin_bits"],
        batch_size=cfg["batch_size"] or 1,
        cascade=cfg["cascade"],
        cascade_max_cond=cfg["cascade_max_cond"],
        cascade_nll_thresh=cfg["cascade_nll_thresh"],
        cascade_min_frac=cfg["cascade_min_frac"],
        cascade_top_k=cfg["cascade_top_k"],
        retriever=retriever if cfg["cascade_retriever"] else None,
        device=device,
    )

    # ── 1. Preprocessing: split documents into compression windows ────────────
    worker_texts = [texts[i] for i in indices]
    all_chunks = [
        c for c in chunk_documents_for_compression(
            worker_texts, tokenizer, max_tokens, decode=True,
        )
        if c.token_ids
    ]
    datas = [c.token_ids for c in all_chunks]

    # ── 2. Retrieve top-k candidates for every window ─────────────────────────
    queries = [c.text for c in all_chunks]
    cand_lists = (
        [[cid for cid, _ in hits]
         for hits in retriever.retrieve_many(queries, top_k=m)]
        if queries else []
    )

    # ── 3. Auto-select RAC scoring batch size ─────────────────────────────────
    chunk_lens = [len(d) for d in datas]

    def _probe(batch_size: int, seq_len: int) -> None:
        dummy = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        with torch.inference_mode():
            model(dummy, use_cache=False)
        del dummy

    score_lens = [n + cfg["max_ctx"] for n in chunk_lens[:200]]
    score_batch_size = cfg["batch_size"] or auto_batch_size(
        _probe, device, score_lens, max_batch=256,
        n_samples=max(1, len(datas) * m), verbose=True,
    )
    rac.batch_size = score_batch_size
    n_split = sum(1 for c in all_chunks if c.total_chunks > 1)
    print(f"  [{device}] score_batch_size={score_batch_size}  docs={len(indices)}  "
          f"chunks={len(all_chunks)}  split={n_split}")

    # ── 4. Compress (and optionally decompress) all chunks ────────────────────
    effective_score_bs = score_batch_size
    while True:
        try:
            rac.batch_size = effective_score_bs
            t0 = time.time()
            cds = rac.compress_batch(datas, cand_lists)
            compress_s = time.time() - t0
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if effective_score_bs == 1:
                raise
            effective_score_bs = max(1, effective_score_bs // 2)
            print(f"\n  OOM — retrying with score_batch_size={effective_score_bs}")

    per_c_s = compress_s / max(len(all_chunks), 1)

    recs = None
    per_d_s = -1.0
    if not no_decomp:
        t0 = time.time()
        recs = rac.decompress_batch(cds)
        per_d_s = (time.time() - t0) / max(len(all_chunks), 1)

    # chunk position → (cd, roundtrip_ok, compress_s, decompress_s)
    chunk_results: Dict[int, tuple] = {}
    for pos, (chunk, cd) in enumerate(zip(all_chunks, cds)):
        rt_ok = -1
        if recs is not None:
            rt_ok = int(recs[pos][0].cpu().tolist() == chunk.token_ids)
        chunk_results[pos] = (cd, rt_ok, per_c_s, per_d_s)

    # ── 5. Aggregate chunk results per original document ──────────────────────
    doc_chunk_pos: Dict[int, List[int]] = defaultdict(list)
    for pos in chunk_results:
        chunk = all_chunks[pos]
        doc_chunk_pos[chunk.doc_idx].append(pos)

    results = []
    for local_idx, global_idx in enumerate(indices):
        positions = doc_chunk_pos.get(local_idx, [])
        doc_data = [chunk_results[p] for p in positions]

        if not doc_data:
            results.append({
                "rac": EvalResult(
                    sample_id=f"rac:doc{global_idx:06d}",
                    original_bytes=0,
                    compressed_bytes=0,
                    bpb=0.0,
                    ratio=0.0,
                    compress_s=0.0,
                    decompress_s=-1.0,
                    peak_gpu_mb=-1,
                    peak_ram_mb=-1,
                    roundtrip_ok=-1,
                ),
                "used": 0,
                "ncond": 0,
                "n_chunks": 0,
                **_prefix_metrics([]),
            })
            continue

        orig_b = len(worker_texts[local_idx].encode())
        comp_b = sum(
            cd.compressed_length + cd.metadata["index_bits"] / 8.0
            for cd, *_ in doc_data
        )
        rt_ok = (1 if all(d[1] == 1 for d in doc_data) else
                 -1 if all(d[1] == -1 for d in doc_data) else 0)
        c_s = sum(d[2] for d in doc_data)
        d_s_vals = [d[3] for d in doc_data if d[3] >= 0]
        d_s = sum(d_s_vals) if d_s_vals else -1.0
        ncond = sum(len(d[0].metadata["ctx_ids"]) for d in doc_data)
        prefix_metrics = _prefix_metrics(doc_data)

        results.append({
            "rac": EvalResult(
                sample_id=f"rac:doc{global_idx:06d}",
                original_bytes=orig_b,
                compressed_bytes=comp_b,
                bpb=comp_b * 8 / max(orig_b, 1),
                ratio=orig_b / max(comp_b, 1),
                compress_s=c_s,
                decompress_s=d_s,
                peak_gpu_mb=-1,
                peak_ram_mb=-1,
                roundtrip_ok=rt_ok,
            ),
            "used": sum(1 for d in doc_data if d[0].metadata["ctx_ids"]),
            "ncond": ncond,
            "n_chunks": len(doc_data),
            **prefix_metrics,
        })
    return results


def _calibrate(device, model_path, texts, calib_idx, database_dir, signals, embed_model,
               m, max_tokens, cfg, alpha, save_path):
    model, tokenizer = _load_model(model_path, device)
    llm = LLMCompressor(model, tokenizer, device=device)
    base_tokens, retriever = _load_database(database_dir, signals, embed_model, device)
    rac = RACCompressor(
        llm, base_tokens,
        max_ctx=cfg["max_ctx"],
        margin_bits=cfg["margin_bits"],
        batch_size=cfg["batch_size"] or 1,
        cascade=cfg["cascade"],
        cascade_max_cond=cfg["cascade_max_cond"],
        cascade_nll_thresh=cfg["cascade_nll_thresh"],
        cascade_min_frac=cfg["cascade_min_frac"],
        cascade_top_k=cfg["cascade_top_k"],
        retriever=retriever if cfg["cascade_retriever"] else None,
        device=device,
    )

    worker_texts = [texts[i] for i in calib_idx]
    chunks = [
        c for c in chunk_documents_for_compression(
            worker_texts, tokenizer, max_tokens, decode=True,
        )
        if c.token_ids
    ]
    datas = [c.token_ids for c in chunks]
    queries = [c.text for c in chunks]
    cand_lists = (
        [[cid for cid, _ in hits]
         for hits in retriever.retrieve_many(queries, top_k=m)]
        if queries else []
    )

    def _probe(batch_size: int, seq_len: int) -> None:
        dummy = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        with torch.inference_mode():
            model(dummy, use_cache=False)
        del dummy

    score_lens = [len(d) + cfg["max_ctx"] for d in datas[:200]]
    rac.batch_size = cfg["batch_size"] or auto_batch_size(
        _probe, device, score_lens, max_batch=256,
        n_samples=max(1, len(datas) * m), verbose=False,
    )
    cds = rac.compress_batch(datas, cand_lists)
    seqs = [list(cd.metadata["ctx_ids"]) for cd in cds]
    CalibratedIndexCoder.calibrate(seqs, len(base_tokens), alpha=alpha).save(save_path)
    print(f"  calibrated index on {len(seqs)} chunks "
          f"(n_base={len(base_tokens)}) -> {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Oracle RAC compression evaluation — text")
    p.add_argument("--database", required=True,
                   help="prepare_rac_data --out dir (base + index + meta + eval_docs)")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default=None,
                   help="eval corpus (default: --database/eval_docs.jsonl)")
    p.add_argument("--n-docs", type=int, default=None,
                   help="Max eval documents (default: all)")
    p.add_argument("--m", type=int, default=16,
                   help="top-k candidates tried per window")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="data piece size (default: LM ctx - max-ctx)")
    p.add_argument("--max-ctx", type=int, default=1024,
                   help="total prefix-token budget")
    p.add_argument("--margin-bits", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=None,
                   help="candidates scored per LM forward (default: auto-probe)")
    p.add_argument("--cascade", action="store_true")
    p.add_argument("--cascade-max-cond", type=int, default=2)
    p.add_argument("--cascade-nll-thresh", type=float, default=4.0)
    p.add_argument("--cascade-min-frac", type=float, default=0.05)
    p.add_argument("--cascade-top-k", type=int, default=16)
    p.add_argument("--cascade-retriever", action="store_true",
                   help="re-retrieve the next condition from high-entropy residual")
    p.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--calib-docs", type=int, default=50)
    p.add_argument("--calib-alpha", type=float, default=0.5)
    p.add_argument("--save-index", default=None, metavar="JSON")
    p.add_argument("--no-decompress", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Comma-separated devices, e.g. cuda:0,cuda:1")
    p.add_argument("--tmp-dir", default="tmp")
    p.add_argument("--output", default=None, metavar="CSV")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    args.model = os.path.normpath(args.model)
    devices = parse_devices(args.device)

    with open(os.path.join(args.database, "meta.json")) as f:
        meta = json.load(f)

    ctx_len = AutoConfig.from_pretrained(args.model).max_position_embeddings
    max_tokens = args.max_tokens or (ctx_len - args.max_ctx)

    eval_path = args.dataset or os.path.join(args.database, "eval_docs.jsonl")
    n_calib = args.calib_docs if args.calibrate else 0
    n_load = args.n_docs + n_calib if args.n_docs else None
    texts = load_text_documents(eval_path, num_documents=n_load)

    all_idx = list(range(len(texts)))
    if n_calib >= len(all_idx) and n_calib > 0:
        raise ValueError(
            f"--calib-docs={n_calib} leaves no eval docs; "
            f"loaded only {len(all_idx)} docs")
    if n_calib:
        # the latter part of the eval set is used for calibration, the former for testing
        calib_idx = all_idx[-n_calib:]
        test_idx = all_idx[:-n_calib]
    else:
        calib_idx = []
        test_idx = all_idx
    print(f"Loaded {len(texts)} docs from {eval_path} | eval {len(test_idx)} | "
          f"data piece {max_tokens} tok | m {args.m} | devices: {devices}")

    cfg = dict(
        max_ctx=args.max_ctx,
        margin_bits=args.margin_bits,
        batch_size=args.batch_size,
        cascade=args.cascade,
        cascade_max_cond=args.cascade_max_cond,
        cascade_nll_thresh=args.cascade_nll_thresh,
        cascade_min_frac=args.cascade_min_frac,
        cascade_top_k=args.cascade_top_k,
        cascade_retriever=args.cascade_retriever,
    )

    index_path = None
    if args.calibrate:
        index_path = args.save_index or os.path.join(tempfile.mkdtemp(), "index.json")
        print(f"Calibrating index coder on {len(calib_idx)} held-out docs ...")
        _calibrate(
            devices[0], args.model, texts, calib_idx, args.database, meta["signals"],
            args.embed_model, args.m, max_tokens, cfg, args.calib_alpha, index_path,
        )

    results = run_multi_gpu(
        _rac_worker, test_idx, devices,
        fn_kwargs=dict(
            model_path=args.model,
            texts=texts,
            database_dir=args.database,
            signals=meta["signals"],
            embed_model=args.embed_model,
            m=args.m,
            max_tokens=max_tokens,
            cfg=cfg,
            index_path=index_path,
            no_decomp=args.no_decompress,
        ),
        tmp_prefix=os.path.join(args.tmp_dir, "_eval_rac"),
    )

    stats = EvalStats()
    rows: List[EvalResult] = []
    used = ncond = n_chunks = 0
    prefix_hist: Dict[int, int] = defaultdict(int)
    prefix_gain_sums: Dict[int, float] = defaultdict(float)
    prefix_net_gain_sums: Dict[int, float] = defaultdict(float)
    prefix_id_bit_sums: Dict[int, float] = defaultdict(float)
    prefix_gain_counts: Dict[int, int] = defaultdict(int)
    for r in results:
        stats.update(r["rac"])
        rows.append(r["rac"])
        used += r["used"]
        ncond += r["ncond"]
        n_chunks += r["n_chunks"]
        _merge_numeric_dict(prefix_hist, r["prefix_hist"])
        _merge_numeric_dict(prefix_gain_sums, r["prefix_gain_sums"])
        _merge_numeric_dict(prefix_net_gain_sums, r["prefix_net_gain_sums"])
        _merge_numeric_dict(prefix_id_bit_sums, r["prefix_id_bit_sums"])
        _merge_numeric_dict(prefix_gain_counts, r["prefix_gain_counts"])

    stats.print_summary(label="rac (oracle)")
    print(f"\n  condition used on {100 * used / max(n_chunks, 1):.1f}% "
          f"of {n_chunks} chunks | {ncond / max(n_chunks, 1):.3f} cond/chunk")
    if n_chunks:
        print("\n  prefix count distribution:")
        for n_prefix in sorted(prefix_hist):
            count = prefix_hist[n_prefix]
            print(f"    {n_prefix} prefix: {count} chunks "
                  f"({100 * count / n_chunks:.1f}%)")
    if prefix_gain_counts:
        print("\n  average gain per accepted prefix:")
        for pos in sorted(prefix_gain_counts):
            count = prefix_gain_counts[pos]
            gain = prefix_gain_sums[pos] / max(count, 1)
            net_gain = prefix_net_gain_sums[pos] / max(count, 1)
            id_bits = prefix_id_bit_sums[pos] / max(count, 1)
            print(f"    prefix {pos}: data_gain={gain:.2f} bits  "
                  f"net_gain={net_gain:.2f} bits  id_cost={id_bits:.2f} bits  "
                  f"n={count}")

    if args.output:
        save_csv(rows, args.output)
        print(f"CSV -> {args.output}")


if __name__ == "__main__":
    main()
