"""RAG-LLM compression evaluation.

Entry points:
  test_workflow()   — compress + decompress one document with RAG, verify round-trip.
  benchmark_ratio() — measure compression ratio (including retrieval overhead) over N docs.

Usage examples
--------------
# Round-trip verification (one document)
python evaluation/eval_rag_llm.py --mode test

# Compression ratio benchmark (10 documents)
python evaluation/eval_rag_llm.py --mode benchmark --n-docs 10

# Custom model / dataset
python evaluation/eval_rag_llm.py --mode benchmark \\
    --model pretrained/SmolLM2-135M \\
    --dataset datasets/cosmopedia-100k \\
    --n-docs 50 --top-k 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

import numpy as np
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.rag_compressor import RAGLLMCompressor
from utils.rag_utils import SimpleRagRetriever, load_and_index_documents
from utils.text_utils import load_text_documents


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config:
    LLM_MODEL_NAME       = "pretrained/SmolLM2-135M"
    EMBEDDING_MODEL_NAME = "pretrained/Qwen3-Embedding-0.6B"
    RAG_DATASET          = "datasets/cosmopedia-100k"
    TEST_DATASET         = "datasets/cosmopedia-100k"
    RETRIEVER_PATH       = "retriever_cache/cosmopedia-qwen3-0.6B"
    NUM_TO_INDEX         = 1000
    TOP_K                = 3
    DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_DTYPE          = torch.float16


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def setup_retriever(
    embedding_model: str = Config.EMBEDDING_MODEL_NAME,
    persist_path:    str = Config.RETRIEVER_PATH,
    rag_dataset:     str = Config.RAG_DATASET,
    num_to_index:    int = Config.NUM_TO_INDEX,
) -> SimpleRagRetriever:
    retriever = SimpleRagRetriever(model_name=embedding_model, persist_path=persist_path)
    load_and_index_documents(retriever, dataset_path=rag_dataset, num_documents=num_to_index)
    return retriever


def load_model(device: str = Config.DEVICE):
    model = AutoModelForCausalLM.from_pretrained(
        Config.LLM_MODEL_NAME, dtype=Config.MODEL_DTYPE, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL_NAME, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def load_test_documents(n: int, skip: int = 0) -> List[str]:
    return load_text_documents(
        Config.TEST_DATASET, num_documents=n, skip_documents=skip
    )


# ---------------------------------------------------------------------------
# Test: round-trip workflow
# ---------------------------------------------------------------------------

def test_workflow(
    top_k: int = Config.TOP_K,
    context_mode: str = "tokens",
):
    """Compress and decompress one document with RAG context, verify text matches."""
    retriever = setup_retriever()
    device    = torch.device(Config.DEVICE)
    model, tokenizer = load_model(Config.DEVICE)
    compressor = RAGLLMCompressor(model, tokenizer, retriever, device=device)

    # Skip indexed docs to avoid test-set leakage
    texts = load_test_documents(1, skip=Config.NUM_TO_INDEX)
    text  = texts[0]

    print(f"\nDocument: {len(text)} chars")
    t0 = time.time()
    cd, retrieval_ids = compressor.compress_text(text, top_k=top_k, context_mode=context_mode)
    print(f"Compressed to {cd.compressed_length} bytes in {time.time()-t0:.2f}s")
    print(f"Retrieval overhead: {cd.metadata.get('retrieval_overhead_bytes', 0)} bytes")

    t0 = time.time()
    recovered = compressor.decompress_text(cd, retrieval_ids, context_mode=context_mode)
    print(f"Decompressed in {time.time()-t0:.2f}s")

    if recovered == text:
        print("Round-trip OK.")
    else:
        print("MISMATCH between original and recovered text!")
        # Token-level diff for debugging
        orig_ids = tokenizer.encode(text)
        rec_ids  = tokenizer.encode(recovered)
        match = sum(a == b for a, b in zip(orig_ids, rec_ids))
        print(f"  Token match: {match}/{len(orig_ids)}")


# ---------------------------------------------------------------------------
# Benchmark: compression ratio
# ---------------------------------------------------------------------------

def benchmark_ratio(
    n_docs:       int = 10,
    top_k:        int = Config.TOP_K,
    context_mode: str = "tokens",
):
    """Measure compression ratio (with retrieval overhead) over n_docs documents."""
    retriever = setup_retriever()
    device    = torch.device(Config.DEVICE)
    model, tokenizer = load_model("auto")
    compressor = RAGLLMCompressor(model, tokenizer, retriever, device=device)

    texts = load_test_documents(n_docs, skip=Config.NUM_TO_INDEX)

    total_comp  = 0
    total_bytes = 0

    for text in tqdm.tqdm(texts, desc="Compressing with RAG"):
        cd, _ = compressor.compress_text(text, top_k=top_k, context_mode=context_mode)
        overhead = cd.metadata.get("retrieval_overhead_bytes", 0)
        total_comp  += cd.compressed_length + overhead
        total_bytes += len(text.encode())

    bpc   = (total_comp * 8) / max(total_bytes, 1)
    ratio = total_bytes / max(total_comp, 1)
    print(f"\nDocuments: {n_docs}")
    print(f"BPC (with retrieval overhead): {bpc:.4f}")
    print(f"Compression ratio:             {ratio:.4f}x")
    return bpc, ratio


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAG-LLM compression evaluation")
    p.add_argument("--mode", choices=["test", "benchmark"], default="benchmark",
                   help="'test' verifies a single round-trip; 'benchmark' measures compression ratio")
    p.add_argument("--model",        default=Config.LLM_MODEL_NAME,
                   help="Path to causal LM (default: %(default)s)")
    p.add_argument("--embed-model",  default=Config.EMBEDDING_MODEL_NAME,
                   help="Path to embedding model (default: %(default)s)")
    p.add_argument("--rag-dataset",  default=Config.RAG_DATASET,
                   help="Dataset to index for retrieval (default: %(default)s)")
    p.add_argument("--test-dataset", default=Config.TEST_DATASET,
                   help="Dataset to compress for evaluation (default: %(default)s)")
    p.add_argument("--retriever-path", default=Config.RETRIEVER_PATH,
                   help="Directory to persist the FAISS index (default: %(default)s)")
    p.add_argument("--num-to-index", type=int, default=Config.NUM_TO_INDEX,
                   help="Documents to index from rag-dataset (default: %(default)s)")
    p.add_argument("--n-docs",  type=int, default=10,
                   help="Documents to compress in benchmark mode (default: %(default)s)")
    p.add_argument("--top-k",   type=int, default=Config.TOP_K,
                   help="Retrieval top-k (default: %(default)s)")
    p.add_argument("--device",  default=Config.DEVICE,
                   help="Torch device, e.g. cuda:0 (default: %(default)s)")
    p.add_argument("--context-mode", default="tokens", choices=["tokens"],
                   help="Prompt context mode (default: %(default)s)")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # Override Config from CLI arguments
    Config.LLM_MODEL_NAME       = args.model
    Config.EMBEDDING_MODEL_NAME = args.embed_model
    Config.RAG_DATASET          = args.rag_dataset
    Config.TEST_DATASET         = args.test_dataset
    Config.RETRIEVER_PATH       = args.retriever_path
    Config.NUM_TO_INDEX         = args.num_to_index
    Config.TOP_K                = args.top_k
    Config.DEVICE               = args.device

    if args.mode == "test":
        test_workflow(top_k=args.top_k, context_mode=args.context_mode)
    else:
        benchmark_ratio(n_docs=args.n_docs, top_k=args.top_k, context_mode=args.context_mode)


if __name__ == "__main__":
    main()
