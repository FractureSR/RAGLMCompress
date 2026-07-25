"""Stage-1: oracle RAC over the byte/hex LLM, vs no-condition base and codecs.

Splits the loaded samples into a base corpus and a held-out eval set, chunks the
base into byte conditions, indexes them with byte-k-gram BM25, then for each
held-out window retrieves + oracle-selects a condition (see hex_rac_compressor).
Reports RAC bpb (payload code + transmitted id bits) against the SAME model with
no retrieval and against gzip/PNG/JPEG XL/FLAC, plus the diagnostics that decide
whether retrieval is actually paying: how often a condition is accepted and the
average data-bit gain per accepted window.

    python hex_llm/eval_hex_rac.py --modality image --dataset datasets/eurosat/Forest \
        --model Qwen/Qwen3-0.6B --route byte --n-samples 60 --base-frac 0.5 \
        --condition-bytes 768 --max-payload-bytes 3072 --m 8 --device cuda:0

RAC helps only where the base corpus holds byte-near-duplicates of the payload
(redundant / non-deduplicated data). On deduplicated natural data expect ~0 —
that is the point of the diagnostics, not a bug. TEXT uses the project's native
pipeline; this is for image/audio bytes.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import List, Tuple

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from compression.rac_index import FixedIndexCoder
from hex_llm.eval_hex_llm import (
    _augment_path_for_codecs,
    _load_model,
    _load_samples,
    _reference_codecs,
    _windows,
)
from hex_llm.hex_llm_compressor import build_compressor
from hex_llm.hex_rac_compressor import HexRACCompressor, chunk_bytes


def _build_retriever(base_chunks: List[bytes], kgram: int):
    from utils.bgpt_codec_utils import make_bgpt_retriever
    retriever = make_bgpt_retriever(signals="bm25", kgram=kgram)
    retriever.build(list(base_chunks))
    return retriever


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage-1 oracle RAC over the byte/hex LLM")
    p.add_argument("--modality", required=True, choices=("text", "image", "audio"))
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--route", choices=("byte", "hex"), default="byte")
    p.add_argument("--n-samples", type=int, default=60, help="total loaded, then split")
    p.add_argument("--base-frac", type=float, default=0.5, help="fraction used as the base corpus")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--condition-bytes", type=int, default=768,
                   help="base is chunked into conditions this many bytes long")
    p.add_argument("--max-ctx", type=int, default=None,
                   help="prefix byte budget (default: --condition-bytes)")
    p.add_argument("--max-payload-bytes", type=int, default=3072)
    p.add_argument("--m", type=int, default=8, help="top-m candidates the oracle tries")
    p.add_argument("--kgram", type=int, default=8, help="byte k-gram size for BM25")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verify-bytes", type=int, default=64,
                   help="round-trip check this many payload bytes with a condition (0=skip)")
    p.add_argument("--codec-path", default=os.path.join(_REPO_ROOT, "baseline_coders"))
    p.add_argument("--tmp-dir", default=None)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    device = torch.device(args.device)
    max_ctx = args.max_ctx if args.max_ctx is not None else args.condition_bytes
    _augment_path_for_codecs(args.codec_path)

    model, tok = _load_model(os.path.normpath(args.model) if os.path.isdir(args.model)
                             else args.model, device)
    base = build_compressor(model, tok, route=args.route, device=device)
    spb = base.symbols_per_byte
    ctx = getattr(model.config, "max_position_embeddings", None)
    need = spb * (max_ctx + args.max_payload_bytes) + 1
    if ctx is not None and need > ctx:
        raise SystemExit(
            f"condition {max_ctx} + payload {args.max_payload_bytes} B -> {need} tokens "
            f"> model context {ctx}; lower --condition-bytes/--max-payload-bytes")

    # ── split loaded samples into base corpus and held-out eval ───────────────
    samples = _load_samples(args.modality, args.dataset, args.n_samples)
    if len(samples) < 2:
        raise SystemExit("need at least 2 samples to split base/eval")
    order = list(range(len(samples)))
    random.Random(args.seed).shuffle(order)
    n_base = max(1, round(args.base_frac * len(samples)))
    base_idx, eval_idx = set(order[:n_base]), order[n_base:]
    if not eval_idx:
        raise SystemExit("base-frac leaves no eval samples")

    base_chunks: List[bytes] = []
    for i in base_idx:
        base_chunks.extend(chunk_bytes(samples[i][0], args.condition_bytes))
    if not base_chunks:
        raise SystemExit("no base chunks — lower --condition-bytes or add samples")
    retriever = _build_retriever(base_chunks, args.kgram)
    rac = HexRACCompressor(
        base, base_chunks, retriever,
        index_coder=FixedIndexCoder(len(base_chunks)), max_ctx=max_ctx, m=args.m)

    eval_samples = [samples[i] for i in eval_idx]
    total_orig = sum(len(d) for d, _ in eval_samples)
    print(f"hex-RAC | route={args.route} | {args.modality} | base {len(base_idx)} samples "
          f"({len(base_chunks)} chunks x {args.condition_bytes} B) | eval {len(eval_idx)} "
          f"samples ({total_orig:,} B) | m={args.m} | model {args.model} | {device}")

    if args.verify_bytes > 0 and base_chunks:
        probe = eval_samples[0][0][: args.verify_bytes]
        prefix = base_chunks[0][:max_ctx]
        print(f"  round-trip check on {len(probe)} B with a condition "
              f"({spb * len(probe)} decode steps) ...", flush=True)
        if not base.roundtrip_ok(probe, prefix=prefix):
            raise SystemExit("round-trip with a condition failed — coder is not lossless")
        print("  round-trip: OK")

    import tqdm
    windows = [w for d, _ in eval_samples for w in _windows(d, args.max_payload_bytes) if w]
    rac_bytes = base_bytes = 0.0
    n_units = used = 0
    gain_sum = net_sum = idbits_sum = 0.0
    t0 = time.monotonic()
    bar = tqdm.tqdm(windows, desc=f"hex-RAC {args.modality}", unit="win")
    for window in bar:
        r = rac.compress(window)
        # No-condition base size: when RAC took no condition its coded bytes ARE
        # the base, so only re-encode when a condition was actually used.
        if r.cond_id is None:
            base_len = len(r.compressed)
        else:
            base_comp, _ = base.compress(window)
            base_len = len(base_comp)
        rac_bytes += r.total_bits / 8
        base_bytes += base_len
        n_units += 1
        if r.cond_id is not None:
            used += 1
            gain_sum += r.gain_bits
            net_sum += r.gain_bits - r.id_bits
            idbits_sum += r.id_bits
        bar.set_postfix(rac=f"{rac_bytes * 8 / max(total_orig, 1):.4f}",
                        base=f"{base_bytes * 8 / max(total_orig, 1):.4f}")
    bar.close()
    elapsed = time.monotonic() - t0

    # ── reference codecs (whole sample) ───────────────────────────────────────
    refs = _reference_codecs(args.modality, args.tmp_dir)
    ref_comp = {label: 0 for label, _ in refs}
    alive = {label: True for label, _ in refs}
    for data, meta in eval_samples:
        for label, fn in refs:
            if alive[label]:
                try:
                    ref_comp[label] += fn(data, meta)
                except Exception as exc:
                    alive[label] = False
                    print(f"\n  {label} disabled: {type(exc).__name__}: {str(exc).splitlines()[0][:90]}")

    def _row(label: str, comp_bytes: float) -> str:
        return (f"  {label:<12} bpb={comp_bytes * 8 / max(total_orig, 1):.4f}  "
                f"ratio={total_orig / max(comp_bytes, 1e-9):.3f}x")

    print(f"\n== {args.modality} RAC ({len(eval_idx)} eval samples, {total_orig:,} B) ==")
    print(_row(f"RAC ({args.route})", rac_bytes) + "   [windowed + index bits]")
    print(_row(f"base ({args.route})", base_bytes) + "   [windowed, no retrieval]")
    for label, _ in refs:
        if alive[label]:
            print(_row(label, ref_comp[label]) + "   [whole-sample]")

    print(f"\n  condition used on {100 * used / max(n_units, 1):.1f}% of {n_units} windows")
    if used:
        print(f"  per accepted window: data_gain={gain_sum / used:.1f} bits  "
              f"net_gain={net_sum / used:.1f} bits  id_cost={idbits_sum / used:.1f} bits")
    saved = base_bytes - rac_bytes
    print(f"  RAC vs base: {8 * saved / max(total_orig, 1):+.4f} bpb "
          f"({100 * saved / max(base_bytes, 1e-9):+.2f}%)")
    print(f"  {elapsed:.1f}s ({elapsed / max(len(eval_idx), 1):.2f}s/sample)")
    print("\n  data_gain/payload-code near 0 => no exploitable cross-sample redundancy here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
