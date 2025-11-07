import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import time
import gc
import os
import pickle

class SimpleRagRetriever:

    def __init__(self, model_name, normalize_embeddings=True, persist_path="retriever_storage"):
        """
        :param model_name: embedding model name or path.
        :param normalize_embeddings: whether to normalize the embeddings.
        :param persist_path: Path to the directory where the index and doc_store will be saved/loaded.
        """
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(model_name)
        self.normalize_embeddings = normalize_embeddings
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"Embedding model loaded. Embedding dimension: {self.embedding_dim}")

        self.persist_path = persist_path
        self.index_file = os.path.join(persist_path, "faiss_index.bin")
        self.doc_store_file = os.path.join(persist_path, "doc_store.pkl")

        if os.path.exists(self.index_file) and os.path.exists(self.doc_store_file):
            print(f"Found existing index at '{self.persist_path}'. Loading...")
            self._load_index()
            print(f"Index loaded successfully. Total indexed documents: {self.index.ntotal}")
        else:
            print("No existing index found. Initializing a new one.")
            self.doc_store = {}
            self.next_id = 0
            self.index = None
    
    def _save_index(self):
        """Saves the FAISS index and the document store to disk."""
        if not self.persist_path:
            return

        print(f"Saving index to '{self.persist_path}'...")
        os.makedirs(self.persist_path, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, self.index_file)

        # Save doc_store and next_id using pickle
        with open(self.doc_store_file, "wb") as f:
            pickle.dump({
                "doc_store": self.doc_store,
                "next_id": self.next_id
            }, f)
        print("Index saved successfully.")

    def _load_index(self):
        """Loads the FAISS index and the document store from disk."""
        # Load FAISS index
        self.index = faiss.read_index(self.index_file)

        # Load doc_store and next_id from pickle file
        with open(self.doc_store_file, "rb") as f:
            data = pickle.load(f)
            self.doc_store = data["doc_store"]
            self.next_id = data["next_id"]

    def _chunk_text(self, text, chunk_size=256, chunk_overlap=32):
        words = text.split()
        if not words:
            return []
        
        chunks = []
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk = words[i:i + chunk_size]
            chunks.append(" ".join(chunk))
        return chunks

    def index_documents(self, documents):
        """
        Indexes a list of documents. If the index already contains documents,
        this will add new documents to the existing index.
        :param documents: Iterable, each element is a dictionary containing 'title' and 'text'.
        """
        if not documents:
            print("No new documents to index.")
            return

        print("Starting indexing...")
        all_chunks = []
        
        # Store starting ID to correctly map new chunks to their IDs
        start_id = self.next_id

        for doc in documents:
            doc_text = f"Title: {doc['title']}\n{doc['text']}"
            chunks = self._chunk_text(doc_text)
            for chunk in chunks:
                current_id = self.next_id
                self.doc_store[current_id] = chunk
                all_chunks.append(chunk)
                self.next_id += 1
        
        if not all_chunks:
            print("Warning: No chunks to index.")
            return
        print(f"Chunking complete, got {len(all_chunks)} new chunks.")

        print("Generating embeddings for new chunks...")
        start_time = time.time()
        embeddings = self.embedding_model.encode(
            all_chunks, 
            show_progress_bar=True, 
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )
        end_time = time.time()
        print(f"Embedding generation complete, took: {end_time - start_time:.2f} seconds.")

        print("Building or updating FAISS index...")
        if self.index is None:
            # Create new FAISS index if it doesn't exist
            if self.normalize_embeddings:
                # Use Inner Product for normalized embeddings (cosine similarity)
                self.index = faiss.IndexFlatIP(self.embedding_dim)
            else:
                # Use L2 distance for non-normalized embeddings
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            # Wrap with IndexIDMap to allow custom integer IDs
            self.index = faiss.IndexIDMap(self.index)

        # FAISS requires IDs to be int64
        ids_to_add = np.arange(start_id, self.next_id, dtype=np.int64)
        self.index.add_with_ids(embeddings.astype('float32'), ids_to_add)

        print(f"Indexing complete. Total indexed chunks: {self.index.ntotal}")

        # --- NEW: Save the updated index to disk ---
        self._save_index()

    def retrieve(self, query, k=5):
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Index is empty. Please call index_documents() with some data first.")

        print(f"\nSearching: '{query}'")

        start_time = time.time()
        query_embedding = self.embedding_model.encode(
            [query], 
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )
        
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        end_time = time.time()
        print(f"Search time: {end_time - start_time:.4f} seconds.")

        results = []
        retrieved_ids = indices[0]
        retrieved_scores = distances[0]
        
        for i in range(len(retrieved_ids)):
            doc_id = retrieved_ids[i]
            if doc_id != -1:
                score = retrieved_scores[i]
                text = self.get_text_by_id(doc_id)
                results.append({
                    "id": int(doc_id),
                    "text": text,
                    "score": float(score)
                })
        
        return results

    def get_text_by_id(self, doc_id):
        return self.doc_store.get(doc_id)


if __name__ == '__main__':
    
    # Define the path for our persistent storage
    storage_path = "retriever_cache/wikipedia-qwen3_embedding_0.6B-storage"

    # 1. Initialize retriever. It will automatically load from `storage_path` if it exists.
    retriever = SimpleRagRetriever(model_name='pretrained/Qwen3-Embedding-0.6B', persist_path=storage_path)

    # 2. Check if the index is empty. If so, load data and index it.
    if retriever.index is None or retriever.index.ntotal == 0:
        print("\nIndex is empty. Proceeding to download data and build a new index.")
        print("Loading Wikipedia dataset sample...")
        dataset = load_dataset("datasets/wiki20231101en", split='train', streaming=True)

        num_documents_to_index = 1000
        documents_to_process = list(iter(dataset.take(num_documents_to_index)))
        print(f"Loaded {len(documents_to_process)} documents for indexing.")
        
        # Index the documents. This will also save the index to disk automatically.
        retriever.index_documents(documents_to_process)

        # Clean up memory
        del documents_to_process
        del dataset
        gc.collect()
    else:
        print("\nIndex was loaded from disk. Skipping data download and indexing.")

    # 3. Now, the retriever is ready to be used, either newly built or loaded from disk.
    print("\n--- Retriever is ready. Executing queries. ---")
    
    query1 = "What is the theory of relativity?"
    top_k_results1 = retriever.retrieve(query1, k=3)

    print("\n--- Top 3 Retrieval Results ---")
    for result in top_k_results1:
        print(f"ID: {result['id']}")
        # For Inner Product (normalized), higher score is better. 1.0 is perfect match.
        # For L2 Distance, lower score is better. 0.0 is perfect match.
        print(f"Score: {result['score']:.4f} ({'Higher is better' if retriever.normalize_embeddings else 'Lower is better'})")
        print(f"Text: {result['text']}\n")

    # To demonstrate, let's try another query
    query2 = "Who was the first person on the moon?"
    top_k_results2 = retriever.retrieve(query2, k=3)

    print("\n--- Top 3 Retrieval Results ---")
    for result in top_k_results2:
        print(f"ID: {result['id']}")
        print(f"Score: {result['score']:.4f} ({'Higher is better' if retriever.normalize_embeddings else 'Lower is better'})")
        print(f"Text: {result['text']}\n")