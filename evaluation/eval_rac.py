"""Evaluate Retrieval-Augmented Compression (RAC) on a prepared test split.

Compares, on the held-out test chunks, several conditioning strategies through
the same frozen LM + arithmetic coder:

    none        no context (plain LLMCompressor baseline)
    token_rag   raw retrieved chunk tokens as a prefix (naive RAG, B=1)
    rac         trained ChunkEncoder latents as prefix (the method)
    rac_rerank  rac + trained reranker selecting the top-m of k candidates

RAC methods also pay a small side-information cost: the m selected base ids are
transmitted so the decoder can rebuild the prefix.  That overhead is added to the
compressed size for honest bits-per-byte / ratio.

Example
-------
    python evaluation/eval_rac.py \\
        --data datasets/rac/cosmopedia --model pretrained/SmolLM2-135M \\
        --encoder pretrained/rac_encoder/cosmopedia \\
        --reranker pretrained/rac_reranker/cosmopedia \\
        --methods none,rac,rac_rerank --m 4 --device cuda:0 \\
        --output results/rac_cosmopedia.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.llm_compressor import LLMCompressor
from compression.rac_encoder import ChunkEncoder
from compression.rac_compressor import RACCompressor
from compression.rac_reranker import Reranker, RerankerScorer
from compression.types import PromptContext
from utils.eval_utils import EvalResult, EvalStats, save_csv
from utils.text_utils import pad_token_ids
from utils.rag_utils import load_splits_from_cache, load_retrieval_cache


# ---------------------------------------------------------------------------
# Per-method compression
# ---------------------------------------------------------------------------

def _result(sample_id, orig_b, comp_b, c_s, d_s, rt) -> EvalResult:
    return EvalResult(
        sample_id=sample_id, original_bytes=orig_b, compressed_bytes=comp_b,
        bpb=comp_b * 8 / max(orig_b, 1), ratio=orig_b / max(comp_b, 1),
        compress_s=c_s, decompress_s=d_s, peak_gpu_mb=-1, peak_ram_mb=-1,
        roundtrip_ok=rt,
    )


def run_none(llm, samples, batch, no_decomp) -> List[EvalResult]:
    pad_id = llm.tokenizer.pad_token_id or 0
    out: List[EvalResult] = []
    for s in range(0, len(samples), batch):
        bs = samples[s:s + batch]
        ids, attn = pad_token_ids([x["data"] for x in bs], pad_id, device=llm.device)
        t0 = time.time(); cds = llm.compress_batch(ids, attn); c_s = (time.time() - t0) / len(bs)
        d_s, recs = -1.0, None
        if not no_decomp:
            t0 = time.time(); recs = llm.decompress_batch(cds); d_s = (time.time() - t0) / len(bs)
        for i, x in enumerate(bs):
            rt = int(recs[i][0].cpu().tolist() == x["data"]) if recs else -1
            out.append(_result(f"none:{x['id']}", x["orig_b"], cds[i].compressed_length, c_s, d_s, rt))
    return out


def run_rac(rac, samples, batch, no_decomp, n_base, tag) -> List[EvalResult]:
    overhead = rac.ctx_overhead_bytes(n_base)
    out: List[EvalResult] = []
    for s in range(0, len(samples), batch):
        bs = samples[s:s + batch]
        t0 = time.time()
        cds = rac.compress([x["data"] for x in bs], [x["cands"] for x in bs])
        c_s = (time.time() - t0) / len(bs)
        d_s, recs = -1.0, None
        if not no_decomp:
            t0 = time.time(); recs = rac.decompress(cds); d_s = (time.time() - t0) / len(bs)
        for i, x in enumerate(bs):
            rt = int(recs[i][0].cpu().tolist() == x["data"]) if recs else -1
            comp_b = cds[i].compressed_length + overhead
            out.append(_result(f"{tag}:{x['id']}", x["orig_b"], comp_b, c_s, d_s, rt))
    return out


def run_token_rag(llm, samples, m, max_ctx, no_decomp, n_base) -> List[EvalResult]:
    """Naive RAG: concatenated raw tokens of the top-m base chunks as prefix (B=1)."""
    import math
    pad_id = llm.tokenizer.pad_token_id or 0
    overhead = m * max(1, math.ceil(math.log2(max(n_base, 2)) / 8))
    out: List[EvalResult] = []
    for x in samples:
        ctx_ids = [t for cid in x["cands"][:m] for t in llm_base_tok[cid]][:max_ctx]
        prompt = PromptContext(mode="tokens",
                               token_ids=torch.tensor([ctx_ids], dtype=torch.long))
        ids, attn = pad_token_ids([x["data"]], pad_id, device=llm.device)
        t0 = time.time(); cds = llm.compress_batch(ids, attn, prompt); c_s = time.time() - t0
        d_s, rt = -1.0, -1
        if not no_decomp:
            t0 = time.time(); recs = llm.decompress_batch(cds, prompt); d_s = time.time() - t0
            rt = int(recs[0][0].cpu().tolist() == x["data"])
        out.append(_result(f"token_rag:{x['id']}", x["orig_b"],
                           cds[0].compressed_length + overhead, c_s, d_s, rt))
    return out


# token-rag needs base tokens at module scope for the helper above
llm_base_tok: List[List[int]] = []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate RAC vs baselines")
    p.add_argument("--data", required=True,
                   help="Retrieval-cache dir (prepare_rac_data --out)")
    p.add_argument("--model", required=True)
    p.add_argument("--encoder", default=None, help="ChunkEncoder dir (needed for rac/rac_rerank)")
    p.add_argument("--reranker", default=None, help="Reranker dir (needed for rac_rerank)")
    p.add_argument("--methods", default="none,rac",
                   help="Comma list: none,token_rag,rac,rac_rerank")
    p.add_argument("--m", type=int, default=4, help="Context chunks used at compression")
    p.add_argument("--k", type=int, default=None, help="Candidate pool from cache (default: all)")
    p.add_argument("--n-test", type=int, default=None, help="Limit test docs")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--token-rag-max-ctx", type=int, default=1024)
    p.add_argument("--no-decompress", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None, metavar="CSV")
    return p


def main() -> None:
    global llm_base_tok
    args = _build_parser().parse_args()
    device = torch.device(args.device)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(device).eval()
    llm = LLMCompressor(model, tokenizer, device=device)

    # --- load split + retrieval cache ---
    base, _, test = load_splits_from_cache(args.data)
    base_tok = tokenizer([b["text"] for b in base],
                         add_special_tokens=False)["input_ids"]
    llm_base_tok = base_tok
    n_base = len(base_tok)

    retr = load_retrieval_cache(args.data, "test")

    samples: List[dict] = []
    need_ctx = any(meth in ("rac", "rac_rerank", "token_rag") for meth in methods)
    for p in test:
        cands = retr.get(str(p["id"]), [])
        if args.k:
            cands = cands[: args.k]
        if need_ctx and len(cands) < 1:
            continue
        ids = tokenizer(p["text"], add_special_tokens=False)["input_ids"]
        if len(ids) < 2:
            continue
        samples.append({"id": p["id"], "data": ids, "cands": cands,
                        "orig_b": len(p["text"].encode())})
        if args.n_test and len(samples) >= args.n_test:
            break
    samples.sort(key=lambda x: len(x["data"]))
    print(f"Eval samples: {len(samples)} | methods: {methods}")

    # --- build RAC components if needed ---
    rac = rac_rr = None
    if "rac" in methods or "rac_rerank" in methods:
        if not args.encoder:
            raise SystemExit("--encoder required for rac/rac_rerank")
        encoder = ChunkEncoder.from_pretrained(args.encoder, map_location=device).to(device)
        if "rac" in methods:
            rac = RACCompressor(llm, encoder, base_tok, m=args.m, device=device)
        if "rac_rerank" in methods:
            if not args.reranker:
                raise SystemExit("--reranker required for rac_rerank")
            rr = Reranker.from_pretrained(args.reranker, map_location=device)
            scorer = RerankerScorer(rr, model, base_tok, pad_id, device)
            rac_rr = RACCompressor(llm, encoder, base_tok, m=args.m, reranker=scorer, device=device)

    # --- run each method ---
    all_results: List[EvalResult] = []
    for meth in methods:
        print(f"\n=== {meth} ===")
        if meth == "none":
            res = run_none(llm, samples, args.batch_size, args.no_decompress)
        elif meth == "rac":
            res = run_rac(rac, samples, args.batch_size, args.no_decompress, n_base, "rac")
        elif meth == "rac_rerank":
            res = run_rac(rac_rr, samples, args.batch_size, args.no_decompress, n_base, "rac_rerank")
        elif meth == "token_rag":
            res = run_token_rag(llm, samples, args.m, args.token_rag_max_ctx,
                                args.no_decompress, n_base)
        else:
            print(f"  unknown method {meth!r}, skipping"); continue
        stats = EvalStats()
        for r in res:
            stats.update(r)
        stats.print_summary(label=meth)
        all_results.extend(res)

    if args.output:
        save_csv(all_results, args.output)


if __name__ == "__main__":
    main()
