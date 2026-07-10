"""Probe per-document sizes for any registered text dataset."""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _percentile(sorted_values: List[int], q: float) -> int:
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return int(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _length_stats(lengths: List[int]) -> dict:
    ordered = sorted(lengths)
    return dict(
        n=len(lengths),
        mean=sum(lengths) / max(len(lengths), 1),
        p50=_percentile(ordered, 50),
        p90=_percentile(ordered, 90),
        p95=_percentile(ordered, 95),
        p99=_percentile(ordered, 99),
        max=max(lengths),
    )


def _print_stats(label: str, lengths: List[int]) -> None:
    if not lengths:
        print(f"{label}: no documents")
        return
    stats = _length_stats(lengths)
    print(
        f"{label}: n={stats['n']}  mean={stats['mean']:.1f}  "
        f"p50={stats['p50']}  p90={stats['p90']}  p95={stats['p95']}  "
        f"p99={stats['p99']}  max={stats['max']}"
    )


def _histogram(lengths: List[int], bins: int) -> None:
    if not lengths or bins <= 0:
        return
    lo, hi = min(lengths), max(lengths)
    if lo == hi:
        print("\ntoken histogram:")
        print(f"  [{lo}, {hi}]: {len(lengths)}")
        return

    width = (hi - lo) / bins
    counts = [0] * bins
    for length in lengths:
        idx = min(int((length - lo) / width), bins - 1)
        counts[idx] += 1

    print("\ntoken histogram:")
    for i, count in enumerate(counts):
        left = int(lo + i * width)
        right = int(lo + (i + 1) * width)
        bracket = "]" if i == bins - 1 else ")"
        print(f"  [{left}, {right}{bracket}: {count}")


def main() -> None:
    args = _build_parser().parse_args()

    from utils.text_utils import load_text_documents

    docs = load_text_documents(
        args.dataset,
        num_documents=args.n_docs,
        skip_documents=args.skip_docs,
    )
    print(f"Loaded {len(docs)} documents from {args.dataset}")

    byte_lens = [len(doc.encode("utf-8")) for doc in docs]
    char_lens = [len(doc) for doc in docs]
    _print_stats("document bytes", byte_lens)
    _print_stats("document chars", char_lens)

    if args.model:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        token_lens = [
            len(tokenizer(doc, add_special_tokens=False, verbose=False)["input_ids"])
            for doc in docs
        ]
        _print_stats("document tokens", token_lens)
        _histogram(token_lens, args.hist_bins)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Probe per-document sizes via utils.text_utils loaders")
    p.add_argument("--dataset", required=True,
                   help="Dataset path/name matching a registered text loader")
    p.add_argument("--model", default=None,
                   help="Optional tokenizer model/path for token counts")
    p.add_argument("--n-docs", type=int, default=None,
                   help="Max documents to load")
    p.add_argument("--skip-docs", type=int, default=0,
                   help="Documents to skip before probing")
    p.add_argument("--hist-bins", type=int, default=0,
                   help="Print a token-length histogram with this many bins")
    return p


if __name__ == "__main__":
    main()
