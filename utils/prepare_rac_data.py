"""Build the RAC retrieval **database** — that is all this does now.

No training, no test/train pieces, no precomputed retrieval. We fix a slice of a
dataset as the base corpus, chunk it into retrieval units (LM tokens, which double
as the raw-token conditions prepended at compression), and index it.
``eval_rac.py`` then chunks + retrieves the held-out eval docs *live*, exactly the
way ``eval_llm.py`` chunks its input.

Outputs under ``--out``:
  base_chunks.json  [{id, doc_idx, text, token_ids}]  retrieval units / conditions
  retriever/        saved BM25 / dense / hybrid index
  meta.json         {dataset, seed, base_frac, n_docs, base_doc_indices,
                     chunk_size, signals, model}  (eval reads base_doc_indices to
                     take the held-out complement as the eval set)

Example
-------
    python utils/prepare_rac_data.py \\
        --dataset datasets/codeparrot_github_code/C.jsonl --n-docs 4000 \\
        --base-frac 0.5 --chunk-size 512 --retriever bm25 \\
        --model pretrained/SmolLM2-135M --out results/rac_c_db
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_utils import (
    chunk_documents_for_compression, load_text_documents, make_text_retriever,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build the RAC retrieval database")
    p.add_argument("--dataset", required=True, help="Dataset path (registered loader)")
    p.add_argument("--n-docs", type=int, default=None, help="Docs to load from the dataset")
    p.add_argument("--base-frac", type=float, default=0.5,
                   help="Fraction of docs fixed as the base/database (rest are eval)")
    p.add_argument("--chunk-size", type=int, default=512,
                   help="LM tokens per base retrieval unit / condition")
    p.add_argument("--chunk-overlap", type=int, default=0)
    p.add_argument("--model", required=True, help="Compression LM (its tokenizer chunks the base)")
    p.add_argument("--retriever", default="bm25", choices=["bm25", "dense", "hybrid"],
                   help="retrieval signal: bm25 (syntactic, default — claim 2.1.1), dense, hybrid")
    p.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--embed-batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="Output directory")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    from transformers import AutoTokenizer
    lm_tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    docs = load_text_documents(args.dataset, num_documents=args.n_docs)
    idx = list(range(len(docs)))
    random.Random(args.seed).shuffle(idx)
    n_base = int(round(args.base_frac * len(idx)))
    base_set = set(idx[:n_base])
    base_doc_indices = sorted(base_set)
    base_docs = [docs[i] for i in base_doc_indices]
    eval_docs = [docs[i] for i in range(len(docs)) if i not in base_set]   # held-out, original order
    print(f"Docs: {len(docs)} | base: {len(base_docs)} | eval (held-out): {len(eval_docs)}")

    # align_last_window: every base chunk is exactly chunk_size tokens, so all
    # retrieval-unit conditions share one prefix length and the RAC oracle can
    # score them with a single batched forward (one prefix_length).
    chunks = chunk_documents_for_compression(
        base_docs, lm_tok, args.chunk_size, chunk_overlap=args.chunk_overlap,
        decode=True, align_last_window=True)
    base = []
    for c in chunks:
        if not (c.text and c.text.strip() and c.token_ids):
            continue
        base.append({"id": len(base), "doc_idx": base_doc_indices[c.doc_idx],
                     "text": c.text, "token_ids": c.token_ids})
    print(f"Base chunks (retrieval units): {len(base)}")
    if not base:
        raise ValueError("No non-empty base chunks produced; increase --n-docs or --base-frac")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "base_chunks.json"), "w") as f:
        json.dump(base, f)
        
    with open(os.path.join(args.out, "eval_docs.jsonl"), "w") as f:
        for doc in eval_docs:
            f.write(json.dumps({"text": doc}) + "\n")
    print(f"Held-out eval docs: {len(eval_docs)} -> eval_docs.jsonl")

    print(f"Indexing base ({args.retriever}) ...")
    retriever = make_text_retriever(args.embed_model, args.device, rrf_k=args.rrf_k,
                                    batch_size=args.embed_batch_size, signals=args.retriever)
    retriever.build([b["text"] for b in base])
    retriever.save(os.path.join(args.out, "retriever"))

    meta = {"dataset": args.dataset, "seed": args.seed, "base_frac": args.base_frac,
            "n_docs": args.n_docs, "base_doc_indices": base_doc_indices,
            "chunk_size": args.chunk_size, "signals": args.retriever, "model": args.model}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"Database -> {args.out}")


if __name__ == "__main__":
    main()
