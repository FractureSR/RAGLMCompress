import gc
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from naive_rag import SimpleRagRetriever
from LLMCompress import Metric, compress


rag_dataset = "datasets/wiki20231101en"
test_compression_dataset = "datasets/cosmopedia-100k"
storage_path = "retriever_cache/wikipedia-qwen3_embedding_0.6B-storage"


retriever = SimpleRagRetriever(
    model_name="pretrained/Qwen3-Embedding-0.6B", persist_path=storage_path
)
if retriever.index is None or retriever.index.ntotal == 0:
    print("\nIndex is empty. Proceeding to download data and build a new index.")
    print("Loading Wikipedia dataset sample...")
    dataset = load_dataset(rag_dataset, split="train", streaming=True)

    num_documents_to_index = 1000
    documents_to_process = list(iter(dataset.take(num_documents_to_index)))
    print(f"Loaded {len(documents_to_process)} documents for indexing.")

    retriever.index_documents(documents_to_process)

    del documents_to_process
    del dataset
    gc.collect()
else:
    print("\nIndex was loaded from disk. Skipping data download and indexing.")

print("\n--- Retriever is ready ---")

print("\nLoading LLM and Tokenizer ...")
llm = AutoModelForCausalLM.from_pretrained(
    "pretrained/Qwen3-0.6B", torch_dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("pretrained/Qwen3-0.6B", use_fast=False)
llm.eval()
print("LLM and Tokenizer loaded.")

# prepare data to be compressed
print("\nLoading documents for compression testing ...")
to_be_compressed = load_dataset(test_compression_dataset, split="train", streaming=True)
to_be_compressed = to_be_compressed.remove_columns(
    ["prompt", "text_token_length", "seed_data", "format", "audience"]
)
num_documents_to_compress = 1
documents_to_compress = list(iter(to_be_compressed.take(num_documents_to_compress)))
print(f"Loaded {len(documents_to_compress)} documents for compression testing.")

metric = Metric()
total_length = 0
with open('./test_sample.txt', 'r') as f:
    lines = f.readlines()
for doc in documents_to_compress:
    doc = doc["text"]
    doc = '\n'.join(lines)
    total_length += len(doc)
    top_k_results = retriever.retrieve(doc, k=3)
    if num_documents_to_compress <= 5:
        print("\n--- Top 3 Retrieval Results ---")
        for result in top_k_results:
            print(f"ID: {result['id']}")
            print(
                f"Score: {result['score']:.4f} ({'Higher is better' if retriever.normalize_embeddings else 'Lower is better'})"
            )
            print(f"Text: {result['text']}\n")

    context_docs = " ".join(result["text"] for result in top_k_results)

    tokenized_context = tokenizer(context_docs, return_tensors="pt")
    tokenized_doc = tokenizer(doc, return_tensors="pt")
    prefix_length = tokenized_context["input_ids"].shape[1]

    full_input_ids = torch.cat(
        [tokenized_context["input_ids"], tokenized_doc["input_ids"]], dim=1
    ).to(llm.device)
    full_attention_mask = torch.cat(
        [tokenized_context["attention_mask"], tokenized_doc["attention_mask"]], dim=1
    ).to(llm.device)

    with torch.inference_mode():
        logits = (
            llm(full_input_ids, attention_mask=full_attention_mask, use_cache=False)
            .logits[:, :-1]
            .to(torch.float32)
        )
    compressed_bytes, num_padded_bits, start_symbol, sequence_array, pd, probs = (
        compress(full_input_ids, logits, metric, prefix_length)
    )

print(
    metric.compute_ratio(
        # extra_bytes=3 * num_documents_to_compress * 10 / 8,
        set_total_length=total_length,
    )
)
