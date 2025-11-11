import gc
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Optional
from naive_rag import SimpleRagRetriever
from LLMCompress import Metric, compress


# ==================== Configuration ====================
class RAGCompressionConfig:
    """Configuration for RAG-enhanced compression"""
    # Dataset paths
    RAG_DATASET = "datasets/wiki20231101en"
    TEST_COMPRESSION_DATASET = "datasets/cosmopedia-100k"
    
    # Storage paths
    RETRIEVER_STORAGE_PATH = "retriever_cache/wikipedia-qwen3_embedding_0.6B-storage"
    TEST_SAMPLE_PATH = "datasets/test_workflow/test_sample.txt"
    
    # Model paths
    EMBEDDING_MODEL_NAME = "pretrained/Qwen3-Embedding-0.6B"
    LLM_MODEL_NAME = "pretrained/Qwen3-0.6B"
    
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
def load_test_sample(test_sample_path: str = None) -> Optional[str]:
    """
    Try to load test sample from file
    
    :param test_sample_path: path to test sample file (default from config)
    :return: test sample text if found, None otherwise
    """
    if test_sample_path is None:
        test_sample_path = RAGCompressionConfig.TEST_SAMPLE_PATH
    
    try:
        with open(test_sample_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        test_sample = '\n'.join(lines)
        print(f"✓ Loaded test sample from {test_sample_path}")
        print(f"  Test sample length: {len(test_sample)} characters")
        return test_sample
    except FileNotFoundError:
        print(f"✗ Test sample file not found at {test_sample_path}")
        return None
    except Exception as e:
        print(f"✗ Error loading test sample: {e}")
        return None


def load_compression_documents(
    dataset_path: str = None,
    num_documents: int = None,
    test_sample_path: str = None
) -> tuple[List[str], int]:
    """
    Load documents for compression testing
    
    :param dataset_path: path to compression dataset (default from config)
    :param num_documents: number of documents to load (default from config)
    :param test_sample_path: path to test sample file (default from config)
    :return: tuple of (list of document texts, total_length)
    """
    if dataset_path is None:
        dataset_path = RAGCompressionConfig.TEST_COMPRESSION_DATASET
    if num_documents is None:
        num_documents = RAGCompressionConfig.NUM_DOCUMENTS_TO_COMPRESS
    
    print("\n=== Loading Documents for Compression ===")
    
    # Try to load test sample first
    test_sample = load_test_sample(test_sample_path)
    
    if test_sample is not None:
        # Use test sample
        print(f"Using test sample as the document to compress")
        documents = [test_sample]
        total_length = len(test_sample)
        print(f"Total documents: 1 (from test sample)")
    else:
        # Load from dataset
        print(f"Loading documents from dataset: {dataset_path}")
        to_be_compressed = load_dataset(dataset_path, split="train", streaming=True)
        
        # Remove unnecessary columns
        to_be_compressed = to_be_compressed.remove_columns(
            ["prompt", "text_token_length", "seed_data", "format", "audience"]
        )
        
        documents_data = list(iter(to_be_compressed.take(num_documents)))
        documents = [doc["text"] for doc in documents_data]
        total_length = sum(len(doc) for doc in documents)
        
        print(f"✓ Loaded {len(documents)} documents from dataset")
    
    print(f"Total length: {total_length} characters")
    return documents, total_length


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
    :return: tuple of compression results
    """
    if top_k is None:
        top_k = RAGCompressionConfig.TOP_K_RETRIEVAL
    
    # Retrieve relevant context
    top_k_results = retriever.retrieve(doc_text, k=top_k)
    
    if verbose:
        print(f"\n--- Top {top_k} Retrieval Results ---")
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
    
    return compression_results


# ==================== Main Workflow ====================
def run_rag_compression(
    test_sample_path: str = None,
    dataset_path: str = None,
    num_documents: int = None
):
    """
    Main workflow for RAG-enhanced compression
    
    :param test_sample_path: path to test sample file (default from config)
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
        test_sample_path=test_sample_path
    )
    
    # Initialize metric
    metric = Metric()
    
    # Process each document
    print("\n=== Starting Compression ===")
    num_docs = len(documents)
    verbose = num_docs <= RAGCompressionConfig.VERBOSE_THRESHOLD
    
    for idx, doc_text in enumerate(documents, 1):
        print(f"\n{'='*60}")
        print(f"Processing document {idx}/{num_docs}")
        print(f"Document length: {len(doc_text)} characters")
        print(f"{'='*60}")
        
        compression_results = compress_with_rag_context(
            doc_text=doc_text,
            retriever=retriever,
            llm=llm,
            tokenizer=tokenizer,
            metric=metric,
            verbose=verbose
        )
        
        (compressed_bytes, num_padded_bits, start_symbol, 
         sequence_array, pd, probs) = compression_results
        
        print(f"\n✓ Document {idx} compressed: {len(compressed_bytes)} bytes")
    
    # Compute final compression ratio
    print("\n" + "="*60)
    print("=== Final Compression Results ===")
    print("="*60)
    compression_rate, compression_ratio = metric.compute_ratio(
        set_total_length=total_length
    )
    
    print(f"Total original length: {total_length} characters")
    print(f"Total compressed length: {metric.compressed_length} bytes")
    print(f"Compression ratio: {compression_ratio:.6f}")
    print(f"Compression rate: {compression_rate:.6f}x")
    print("="*60)
    
    return metric, compression_rate, compression_ratio


# ==================== Main ====================
if __name__ == "__main__":
    # Run compression with default configuration
    # Will use test_sample.txt if it exists, otherwise load from dataset
    run_rag_compression()
    
    # You can also specify custom paths:
    # run_rag_compression(
    #     test_sample_path="./my_custom_sample.txt",
    #     dataset_path="datasets/my_dataset",
    #     num_documents=5
    # )