import gc
import math
from functools import partial
from itertools import islice
from typing import List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.naive_rag import SimpleRagRetriever
from evaluation.LLMCompress import Metric, compress


# ==================== Configuration ====================
class RAGCompressionConfig:
    """Configuration for RAG-enhanced compression"""
    # Dataset paths
    RAG_DATASET = "datasets/cosmopedia-100k"
    TEST_COMPRESSION_DATASET = "datasets/cosmopedia-100k"

    # Storage paths
    RETRIEVER_STORAGE_PATH = "retriever_cache/cosmopedia-100k-qwen3_embedding_0.6B-storage"

    # Model paths
    EMBEDDING_MODEL_NAME = "pretrained/Qwen3-Embedding-0.6B"
    LLM_MODEL_NAME = "pretrained/SmolLM2-135M"

    # Model parameters
    LLM_DTYPE = torch.float16

    # Retrieval parameters
    NUM_DOCUMENTS_TO_INDEX = 1000
    TOP_K_RETRIEVAL = 3

    # Compression parameters
    NUM_DOCUMENTS_TO_COMPRESS = 1
    VERBOSE_THRESHOLD = 5  # Print retrieval results if num_documents <= this


# ==================== Retriever Setup ====================
def setup_retriever(
    model_name: str = None,
    persist_path: str = None,
    rag_dataset: str = None,
    num_documents_to_index: int = None
) -> SimpleRagRetriever:
    """
    Setup RAG retriever, either by loading existing index or building new one

    :param model_name: embedding model name (default from config)
    :param persist_path: path to persist/load index (default from config)
    :param rag_dataset: dataset for building index (default from config)
    :param num_documents_to_index: number of documents to index (default from config)
    :return: initialized retriever
    """
    if model_name is None:
        model_name = RAGCompressionConfig.EMBEDDING_MODEL_NAME
    if persist_path is None:
        persist_path = RAGCompressionConfig.RETRIEVER_STORAGE_PATH
    if rag_dataset is None:
        rag_dataset = RAGCompressionConfig.RAG_DATASET
    if num_documents_to_index is None:
        num_documents_to_index = RAGCompressionConfig.NUM_DOCUMENTS_TO_INDEX

    print("\n=== Setting up RAG Retriever ===")
    retriever = SimpleRagRetriever(
        model_name=model_name,
        persist_path=persist_path
    )

    if retriever.index is None or retriever.index.ntotal == 0:
        print("\nIndex is empty. Building new index...")
        print(f"Loading {rag_dataset} dataset...")
        dataset = load_dataset(rag_dataset, split="train", streaming=True)

        documents_to_process = list(iter(dataset.take(num_documents_to_index)))
        print(f"Loaded {len(documents_to_process)} documents for indexing.")

        retriever.index_documents(documents_to_process)

        # Clean up memory
        del documents_to_process
        del dataset
        gc.collect()
        print("Index built successfully.")
    else:
        print("\nIndex loaded from disk. Skipping indexing.")

    print("--- Retriever is ready ---\n")
    return retriever


# ==================== Model Loading ====================
def load_llm_and_tokenizer(
    model_name: str = None,
    dtype: torch.dtype = None,
    device_map: str = "auto"
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load language model and tokenizer

    :param model_name: model name or path (default from config)
    :param dtype: model dtype (default from config)
    :param device_map: device map for model loading
    :return: tuple of (model, tokenizer)
    """
    if model_name is None:
        model_name = RAGCompressionConfig.LLM_MODEL_NAME
    if dtype is None:
        dtype = RAGCompressionConfig.LLM_DTYPE

    print("\n=== Loading LLM and Tokenizer ===")
    llm = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    llm.eval()
    print("LLM and Tokenizer loaded successfully.\n")

    return llm, tokenizer


# ==================== Data Loading ====================
def is_same_dataset(
    rag_dataset: Optional[str],
    test_dataset: Optional[str]
) -> bool:
    """
    Check whether RAG dataset and test compression dataset are the same source.
    Current logic compares dataset identifiers/paths directly.
    """
    if rag_dataset is None or test_dataset is None:
        return False
    return rag_dataset == test_dataset


def load_compression_documents(
    dataset_path: str = None,
    num_documents: int = None,
    rag_dataset: str = None,
    num_documents_to_index: int = None
) -> tuple[List[str], int]:
    """
    Load documents for compression testing.
    If dataset_path and rag_dataset are the same dataset, skip the documents
    already used for building the RAG index.

    :param dataset_path: path to compression dataset (default from config)
    :param num_documents: number of documents to load (default from config)
    :param rag_dataset: RAG dataset path (default from config)
    :param num_documents_to_index: number of indexed RAG documents to skip if same dataset
    :return: tuple of (list of document texts, total_length)
    """
    if dataset_path is None:
        dataset_path = RAGCompressionConfig.TEST_COMPRESSION_DATASET
    if num_documents is None:
        num_documents = RAGCompressionConfig.NUM_DOCUMENTS_TO_COMPRESS
    if rag_dataset is None:
        rag_dataset = RAGCompressionConfig.RAG_DATASET
    if num_documents_to_index is None:
        num_documents_to_index = RAGCompressionConfig.NUM_DOCUMENTS_TO_INDEX

    print("\n=== Loading Documents for Compression ===")
    print(f"Loading documents from dataset: {dataset_path}")

    same_dataset = is_same_dataset(rag_dataset, dataset_path)
    skip_count = num_documents_to_index if same_dataset else 0

    if same_dataset:
        print(
            f"Detected RAG_DATASET == TEST_COMPRESSION_DATASET. "
            f"Skipping first {skip_count} documents used for RAG indexing."
        )

    to_be_compressed = load_dataset(dataset_path, split="train", streaming=True)

    # Remove unnecessary columns if they exist
    removable_columns = [
        "prompt", "text_token_length", "seed_data", "format", "audience"
    ]
    existing_columns = getattr(to_be_compressed, "column_names", None)
    if existing_columns is not None:
        columns_to_remove = [c for c in removable_columns if c in existing_columns]
        if columns_to_remove:
            to_be_compressed = to_be_compressed.remove_columns(columns_to_remove)

    iterator = iter(to_be_compressed)
    documents_data = list(islice(iterator, skip_count, skip_count + num_documents))
    documents = [doc["text"] for doc in documents_data]
    total_length = sum(len(doc) for doc in documents)

    print(f"✓ Loaded {len(documents)} documents from dataset")
    print(f"Total length: {total_length} characters")
    return documents, total_length


# ==================== Retrieval Cost Helpers ====================
def compute_retrieval_extra_bytes(
    retrieval_pool_size: int,
    num_retrieved: int
) -> int:
    """
    Compute the number of extra bytes needed to encode retrieved document IDs.

    If the retriever can return one item from retrieval_pool_size candidates,
    each retrieved ID needs ceil(log2(retrieval_pool_size)) bits.

    Total extra bytes = ceil(num_retrieved * bits_per_id / 8)

    :param retrieval_pool_size: total number of retrievable entries
    :param num_retrieved: number of retrieved entries actually used
    :return: extra bytes needed to encode retrieval IDs
    """
    if retrieval_pool_size <= 1 or num_retrieved <= 0:
        return 0

    bits_per_id = math.ceil(math.log2(retrieval_pool_size))
    total_bits = bits_per_id * num_retrieved
    extra_bytes = math.ceil(total_bits / 8)
    return extra_bytes


# ==================== Compression with RAG ====================
def compress_with_rag_context(
    doc_text: str,
    retriever: SimpleRagRetriever,
    llm: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    metric: Metric,
    top_k: int = None,
    verbose: bool = False
) -> tuple:
    """
    Compress document using RAG context

    :param doc_text: document text to compress
    :param retriever: RAG retriever
    :param llm: language model
    :param tokenizer: tokenizer
    :param metric: compression metric tracker
    :param top_k: number of top documents to retrieve (default from config)
    :param verbose: whether to print retrieval results
    :return: tuple of:
        (
            compression_results,
            retrieval_extra_bytes,
            retrieved_count,
            bits_per_retrieved_id
        )
    """
    if top_k is None:
        top_k = RAGCompressionConfig.TOP_K_RETRIEVAL

    # Retrieve relevant context
    top_k_results = retriever.retrieve(doc_text, k=top_k)

    retrieval_pool_size = retriever.index.ntotal if retriever.index is not None else 0
    retrieved_count = len(top_k_results)
    bits_per_retrieved_id = (
        math.ceil(math.log2(retrieval_pool_size))
        if retrieval_pool_size > 1 else 0
    )
    retrieval_extra_bytes = compute_retrieval_extra_bytes(
        retrieval_pool_size=retrieval_pool_size,
        num_retrieved=retrieved_count
    )

    if verbose:
        print(f"\n--- Top {top_k} Retrieval Results ---")
        print(f"Retrieval pool size: {retrieval_pool_size}")
        print(f"Retrieved count: {retrieved_count}")
        print(f"Bits per retrieved ID: {bits_per_retrieved_id}")
        print(f"Extra retrieval overhead: {retrieval_extra_bytes} bytes")

        for i, result in enumerate(top_k_results, 1):
            print(f"\nResult {i}:")
            print(f"  ID: {result['id']}")
            score_direction = 'Higher is better' if retriever.normalize_embeddings else 'Lower is better'
            print(f"  Score: {result['score']:.4f} ({score_direction})")
            print(f"  Text preview: {result['text'][:200]}...")

    # Combine context documents
    context_docs = " ".join(result["text"] for result in top_k_results)

    # Tokenize context and document
    tokenized_context = tokenizer(context_docs, return_tensors="pt")
    tokenized_doc = tokenizer(doc_text, return_tensors="pt")
    prefix_length = tokenized_context["input_ids"].shape[1]

    # Concatenate context and document
    full_input_ids = torch.cat(
        [tokenized_context["input_ids"], tokenized_doc["input_ids"]], dim=1
    ).to(llm.device)
    full_attention_mask = torch.cat(
        [tokenized_context["attention_mask"], tokenized_doc["attention_mask"]], dim=1
    ).to(llm.device)

    # Generate logits
    with torch.inference_mode():
        logits = (
            llm(full_input_ids, attention_mask=full_attention_mask, use_cache=False)
            .logits[:, :-1]
            .to(torch.float32)
        )

    # Compress
    compression_results = compress(
        full_input_ids,
        logits,
        metric,
        prefix_length=prefix_length
    )

    return (
        compression_results,
        retrieval_extra_bytes,
        retrieved_count,
        bits_per_retrieved_id,
    )


# ==================== Main Workflow ====================
def run_rag_compression(
    dataset_path: str = None,
    num_documents: int = None
):
    """
    Main workflow for RAG-enhanced compression

    :param dataset_path: path to dataset (default from config)
    :param num_documents: number of documents to compress (default from config)
    """
    # Setup retriever
    retriever = setup_retriever()

    # Load LLM and tokenizer
    llm, tokenizer = load_llm_and_tokenizer()

    # Load documents for compression
    documents, total_length = load_compression_documents(
        dataset_path=dataset_path,
        num_documents=num_documents,
        rag_dataset=RAGCompressionConfig.RAG_DATASET,
        num_documents_to_index=RAGCompressionConfig.NUM_DOCUMENTS_TO_INDEX,
    )

    # Initialize metric
    metric = Metric()

    # Track retrieval-ID encoding overhead
    total_retrieval_extra_bytes = 0

    # Process each document
    print("\n=== Starting Compression ===")
    num_docs = len(documents)
    verbose = num_docs <= RAGCompressionConfig.VERBOSE_THRESHOLD

    for idx, doc_text in enumerate(documents, 1):
        print(f"\n{'='*60}")
        print(f"Processing document {idx}/{num_docs}")
        print(f"Document length: {len(doc_text)} characters")
        print(f"{'='*60}")

        (
            compression_results,
            retrieval_extra_bytes,
            retrieved_count,
            bits_per_retrieved_id,
        ) = compress_with_rag_context(
            doc_text=doc_text,
            retriever=retriever,
            llm=llm,
            tokenizer=tokenizer,
            metric=metric,
            verbose=verbose
        )

        total_retrieval_extra_bytes += retrieval_extra_bytes

        (
            compressed_bytes,
            num_padded_bits,
            start_symbol,
            sequence_array,
            pd,
            probs
        ) = compression_results

        print(f"\n✓ Document {idx} compressed: {len(compressed_bytes)} bytes")
        print(
            f"✓ Retrieval overhead: {retrieved_count} ids × "
            f"{bits_per_retrieved_id} bits -> {retrieval_extra_bytes} bytes"
        )

    # Compute final compression ratio
    print("\n" + "="*60)
    print("=== Final Compression Results ===")
    print("="*60)

    compute_ratio_with_retrieval_cost = partial(
        metric.compute_ratio,
        extra_bytes=total_retrieval_extra_bytes
    )

    compression_rate, compression_ratio = compute_ratio_with_retrieval_cost(
        set_total_length=total_length
    )

    print(f"Total original length: {total_length} characters")
    print(f"Base compressed length: {metric.compressed_length} bytes")
    print(f"Retrieval overhead: {total_retrieval_extra_bytes} bytes")
    print(f"Compression ratio: {compression_ratio:.6f}")
    print(f"Compression rate: {compression_rate:.6f}x")
    print("="*60)

    return metric, compression_rate, compression_ratio


# ==================== Main ====================
if __name__ == "__main__":
    # Run compression with default configuration
    run_rag_compression()