"""Prepare a RAC dataset: build + cache the retriever over an i.i.d. base.

Pipeline
--------
1. Load documents, split at document level.
2. Chunk the base into retrieval units (embedding tokenizer) and chunk train/test
   into data pieces sized to the compression LM's context window (LM tokenizer),
   so every piece is compressed with no truncation.
3. Build the retriever (BM25 + dense, fused with RRF) over base chunks.
4. Precompute top-k base hits for every train/test *piece* and cache them.
5. Save base chunks + train/test pieces (as text) so downstream scripts load them
   via --data and re-tokenise with the LM tokenizer.

Example
-------
    python train/prepare_rac_data.py \\
        --dataset datasets/cosmopedia-100k --n-docs 2000 \\
        --chunk-size 256 --base-frac 0.5 --train-frac 0.8 --top-k 32 \\
        --model pretrained/SmolLM2-135M --prompt-budget 128 \\
        --embed-model Qwen/Qwen3-Embedding-0.6B --device cuda:0 \\
        --out datasets/rac/cosmopedia
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_utils import make_text_retriever


def _load_splits(args, lm_tok, piece_size: int) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split docs and chunk each split with the LM tokenizer (length budget).

    Length is bounded in LM tokens (base = retrieval/context unit, train/test =
    compression piece), but every chunk is stored as decoded ``text`` — the
    retriever embeds it with its own tokenizer, and consumers re-tokenize it with
    the LM tokenizer at load time. Same record shape for all three splits.
    """
    if args.modality == "text":
        from utils.text_utils import load_text_documents, chunk_documents_for_compression
        docs = load_text_documents(args.dataset, num_documents=args.n_docs)

        idx = list(range(len(docs)))
        random.Random(args.seed).shuffle(idx)
        n_base = int(round(args.base_frac * len(idx)))
        rest = idx[n_base:]
        n_train = int(round(args.train_frac * len(rest)))
        base_src, train_src, test_src = idx[:n_base], rest[:n_train], rest[n_train:]

        def _chunk(src, size, overlap=0) -> List[dict]:
            chunks = chunk_documents_for_compression(
                [docs[s] for s in src], lm_tok, size, chunk_overlap=overlap, decode=True)
            kept = [c for c in chunks if c.text and c.text.strip()]
            return [{"id": i, "doc_idx": c.doc_idx, "text": c.text}
                    for i, c in enumerate(kept)]

        base = _chunk(base_src, args.chunk_size, args.chunk_overlap)
        return base, _chunk(train_src, piece_size), _chunk(test_src, piece_size)


def _precompute_retrieval(retriever, pieces: List[dict], top_k: int) -> Dict[str, List[int]]:
    # Each piece is also a retrieval query (its text); rank the whole batch at
    # once (dense embed + faiss search run in bulk).
    results = retriever.retrieve_many([p["text"] for p in pieces], top_k=top_k)
    return {str(p["id"]): [cid for cid, _ in hits]
            for p, hits in zip(pieces, results)}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build + cache the RAC retriever")
    p.add_argument("--dataset", required=True, help="Dataset path (registered loader)")
    p.add_argument("--modality", default="text")
    p.add_argument("--n-docs", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=256,
                   help="LM tokens per base retrieval chunk (length via --model tokenizer)")
    p.add_argument("--chunk-overlap", type=int, default=0,
                   help="LM tokens of overlap between consecutive base chunks")
    p.add_argument("--base-frac", type=float, default=0.5)
    p.add_argument("--train-frac", type=float, default=0.8,
                   help="Train fraction of the non-base remainder")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--model", required=True,
                   help="Compression LM (its tokenizer + context window chunk train/test data)")
    p.add_argument("--prompt-budget", type=int, default=128,
                   help="Context tokens reserved for the retrieved prefix; "
                        "data piece size = LM context window - prompt budget")
    p.add_argument("--data-chunk-size", type=int, default=None,
                   help="Override data piece size in LM tokens (default: ctx window - prompt budget)")
    p.add_argument("--top-k", type=int, default=32, help="Hits to cache per query")
    p.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--embed-batch-size", type=int, default=64,
                   help="Batch size for embedding base chunks (index build)")
    p.add_argument("--embed-query-batch-size", type=int, default=None,
                   help="Batch size for embedding train/test pieces (retrieval queries); "
                        "default = --embed-batch-size. Lower it (pieces are long) to avoid OOM")
    p.add_argument("--device", default=None)
    p.add_argument("--rrf-k", type=int, default=60)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    from transformers import AutoConfig, AutoTokenizer
    lm_tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    ctx_len = AutoConfig.from_pretrained(args.model).max_position_embeddings
    piece_size = args.data_chunk_size or (ctx_len - args.prompt_budget)
    print(f"Data piece size: {piece_size} tokens "
          f"(LM ctx {ctx_len} - prompt budget {args.prompt_budget})")

    base, train, test = _load_splits(args, lm_tok, piece_size)
    print(f"Split: base={len(base)} chunks  |  train={len(train)} pieces  |  test={len(test)} pieces")

    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.out, "base_chunks.json"), "w") as f:
        json.dump(base, f)
    with open(os.path.join(args.out, "train_pieces.json"), "w") as f:
        json.dump(train, f)
    with open(os.path.join(args.out, "test_pieces.json"), "w") as f:
        json.dump(test, f)
    print(f"Splits saved -> {args.out}")

    print("Building retriever over base chunks ...")
    retriever = make_text_retriever(args.embed_model, args.device, rrf_k=args.rrf_k,
                                    batch_size=args.embed_batch_size,
                                    query_batch_size=args.embed_query_batch_size)
    retriever.build([rec["text"] for rec in base])
    retriever.save(os.path.join(args.out, "retriever"))

    print("Precomputing retrieval cache ...")
    r_train = _precompute_retrieval(retriever, train, args.top_k)
    r_test  = _precompute_retrieval(retriever, test,  args.top_k)
    with open(os.path.join(args.out, "retrieval_train.json"), "w") as f:
        json.dump(r_train, f)
    with open(os.path.join(args.out, "retrieval_test.json"), "w") as f:
        json.dump(r_test, f)

    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
