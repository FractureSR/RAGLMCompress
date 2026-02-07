import faiss
import numpy as np
import os
import pickle
import time
import gc
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


# ==================== Configuration ====================
class RAGRetrieverConfig:
    """Configuration for RAG retriever"""
    # Model configuration
    DEFAULT_MODEL_NAME = "pretrained/Qwen3-Embedding-0.6B"
    NORMALIZE_EMBEDDINGS = True
    
    # Storage configuration
    DEFAULT_PERSIST_PATH = "retriever_cache/wikipedia-qwen3_embedding_0.6B-storage"
    INDEX_FILENAME = "faiss_index.bin"
    DOC_STORE_FILENAME = "doc_store.pkl"
    
    # Chunking configuration
    CHUNK_SIZE = 256
    CHUNK_OVERLAP = 32
    
    # Dataset configuration
    DEFAULT_DATASET_PATH = "datasets/wiki20231101en"
    DEFAULT_NUM_DOCUMENTS = 1000
    
    # Retrieval configuration
    DEFAULT_TOP_K = 5


# ==================== RAG Retriever Class ====================
class SimpleRagRetriever:
    """Simple RAG retriever using FAISS and SentenceTransformer"""
    
    def __init__(
        self,
        model_name: str = None,
        normalize_embeddings: bool = None,
        persist_path: str = None
    ):
        """
        Initialize RAG retriever
        
        :param model_name: embedding model name or path (default from config)
        :param normalize_embeddings: whether to normalize embeddings (default from config)
        :param persist_path: path to persist/load index (default from config)
        """
        if model_name is None:
            model_name = RAGRetrieverConfig.DEFAULT_MODEL_NAME
        if normalize_embeddings is None:
            normalize_embeddings = RAGRetrieverConfig.NORMALIZE_EMBEDDINGS
        if persist_path is None:
            persist_path = RAGRetrieverConfig.DEFAULT_PERSIST_PATH
        
        print("="*60)
        print("Initializing RAG Retriever")
        print("="*60)
        print(f"Loading embedding model: {model_name}")
        
        self.embedding_model = SentenceTransformer(model_name)
        self.normalize_embeddings = normalize_embeddings
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        print(f"✓ Embedding model loaded")
        print(f"  Embedding dimension: {self.embedding_dim}")
        print(f"  Normalize embeddings: {self.normalize_embeddings}")

        self.persist_path = persist_path
        self.index_file = os.path.join(persist_path, RAGRetrieverConfig.INDEX_FILENAME)
        self.doc_store_file = os.path.join(persist_path, RAGRetrieverConfig.DOC_STORE_FILENAME)

        # Try to load existing index
        if os.path.exists(self.index_file) and os.path.exists(self.doc_store_file):
            print(f"\n✓ Found existing index at '{self.persist_path}'")
            self._load_index()
            print(f"✓ Index loaded successfully")
            print(f"  Total indexed documents: {self.index.ntotal}")
        else:
            print(f"\n✗ No existing index found at '{self.persist_path}'")
            print("  Initializing a new empty index")
            self.doc_store = {}
            self.next_id = 0
            self.index = None
        
        print("="*60)
    
    def _save_index(self):
        """Save the FAISS index and document store to disk"""
        if not self.persist_path:
            return

        print(f"\nSaving index to '{self.persist_path}'...")
        os.makedirs(self.persist_path, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, self.index_file)

        # Save doc_store and next_id using pickle
        with open(self.doc_store_file, "wb") as f:
            pickle.dump({
                "doc_store": self.doc_store,
                "next_id": self.next_id
            }, f)
        
        print(f"✓ Index saved successfully")
        print(f"  Index file: {self.index_file}")
        print(f"  Document store file: {self.doc_store_file}")

    def _load_index(self):
        """Load the FAISS index and document store from disk"""
        # Load FAISS index
        self.index = faiss.read_index(self.index_file)

        # Load doc_store and next_id from pickle file
        with open(self.doc_store_file, "rb") as f:
            data = pickle.load(f)
            self.doc_store = data["doc_store"]
            self.next_id = data["next_id"]

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> List[str]:
        """
        Split text into overlapping chunks
        
        :param text: text to chunk
        :param chunk_size: size of each chunk in words (default from config)
        :param chunk_overlap: overlap between chunks in words (default from config)
        :return: list of text chunks
        """
        if chunk_size is None:
            chunk_size = RAGRetrieverConfig.CHUNK_SIZE
        if chunk_overlap is None:
            chunk_overlap = RAGRetrieverConfig.CHUNK_OVERLAP
        
        words = text.split()
        if not words:
            return []
        
        chunks = []
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk = words[i:i + chunk_size]
            chunks.append(" ".join(chunk))
        
        return chunks

    def index_documents(self, documents: List[Dict[str, str]]):
        """
        Index a list of documents
        If the index already exists, new documents will be added to it
        
        :param documents: list of documents, each with 'title' and 'text' fields
        """
        if not documents:
            print("✗ No documents to index")
            return

        print("\n" + "="*60)
        print("Starting Document Indexing")
        print("="*60)
        print(f"Number of documents to index: {len(documents)}")
        
        all_chunks = []
        start_id = self.next_id

        # Chunk all documents
        for idx, doc in enumerate(documents, 1):
            doc_text = f"Title: {doc['title']}\n{doc['text']}"
            chunks = self._chunk_text(doc_text)
            
            for chunk in chunks:
                current_id = self.next_id
                self.doc_store[current_id] = chunk
                all_chunks.append(chunk)
                self.next_id += 1
            
            if idx % 100 == 0:
                print(f"  Processed {idx}/{len(documents)} documents...")
        
        if not all_chunks:
            print("✗ Warning: No chunks generated from documents")
            return
        
        print(f"✓ Chunking complete")
        print(f"  Total chunks: {len(all_chunks)}")
        print(f"  Average chunks per document: {len(all_chunks)/len(documents):.1f}")

        # Generate embeddings
        print(f"\nGenerating embeddings for {len(all_chunks)} chunks...")
        start_time = time.time()
        embeddings = self.embedding_model.encode(
            all_chunks, 
            show_progress_bar=True, 
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )
        end_time = time.time()
        print(f"✓ Embedding generation complete")
        print(f"  Time taken: {end_time - start_time:.2f} seconds")
        print(f"  Speed: {len(all_chunks)/(end_time - start_time):.1f} chunks/second")

        # Build or update FAISS index
        print(f"\nBuilding/updating FAISS index...")
        if self.index is None:
            # Create new FAISS index
            if self.normalize_embeddings:
                # Use Inner Product for normalized embeddings (cosine similarity)
                base_index = faiss.IndexFlatIP(self.embedding_dim)
                print("  Using Inner Product index (cosine similarity)")
            else:
                # Use L2 distance for non-normalized embeddings
                base_index = faiss.IndexFlatL2(self.embedding_dim)
                print("  Using L2 distance index")
            
            # Wrap with IndexIDMap to allow custom integer IDs
            self.index = faiss.IndexIDMap(base_index)

        # FAISS requires IDs to be int64
        ids_to_add = np.arange(start_id, self.next_id, dtype=np.int64)
        self.index.add_with_ids(embeddings.astype('float32'), ids_to_add)

        print(f"✓ Indexing complete")
        print(f"  Total indexed chunks: {self.index.ntotal}")
        print("="*60)

        # Save the updated index to disk
        self._save_index()

    def retrieve(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant documents for a query
        
        :param query: search query
        :param k: number of results to return (default from config)
        :return: list of retrieval results with id, text, and score
        """
        if k is None:
            k = RAGRetrieverConfig.DEFAULT_TOP_K
        
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError(
                "Index is empty. Please call index_documents() with some data first."
            )

        print(f"\nSearching for: '{query[:100]}{'...' if len(query) > 100 else ''}'")

        start_time = time.time()
        
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query], 
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        end_time = time.time()
        print(f"✓ Search completed in {end_time - start_time:.4f} seconds")

        # Format results
        results = []
        retrieved_ids = indices[0]
        retrieved_scores = distances[0]
        
        for i in range(len(retrieved_ids)):
            doc_id = retrieved_ids[i]
            if doc_id != -1:  # -1 indicates no result found
                score = retrieved_scores[i]
                text = self.get_text_by_id(doc_id)
                results.append({
                    "id": int(doc_id),
                    "text": text,
                    "score": float(score)
                })
        
        print(f"✓ Retrieved {len(results)} results")
        return results

    def get_text_by_id(self, doc_id: int) -> Optional[str]:
        """
        Get document text by ID
        
        :param doc_id: document ID
        :return: document text or None if not found
        """
        return self.doc_store.get(doc_id)


# ==================== Helper Functions ====================
def load_and_index_documents(
    retriever: SimpleRagRetriever,
    dataset_path: str = None,
    num_documents: int = None,
    force_reindex: bool = False
):
    """
    Load documents from dataset and index them
    
    :param retriever: RAG retriever instance
    :param dataset_path: path to dataset (default from config)
    :param num_documents: number of documents to index (default from config)
    :param force_reindex: force reindexing even if index exists
    """
    if dataset_path is None:
        dataset_path = RAGRetrieverConfig.DEFAULT_DATASET_PATH
    if num_documents is None:
        num_documents = RAGRetrieverConfig.DEFAULT_NUM_DOCUMENTS
    
    # Check if we need to index
    if not force_reindex and retriever.index is not None and retriever.index.ntotal > 0:
        print("\n✓ Index already exists and is not empty")
        print("  Skipping data download and indexing")
        return

    print("\n" + "="*60)
    print("Loading Dataset for Indexing")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Number of documents to load: {num_documents}")
    
    dataset = load_dataset(dataset_path, split='train', streaming=True)
    documents_to_process = list(iter(dataset.take(num_documents)))
    
    print(f"✓ Loaded {len(documents_to_process)} documents")
    print("="*60)
    
    # Index the documents
    retriever.index_documents(documents_to_process)

    # Clean up memory
    del documents_to_process
    del dataset
    gc.collect()
    print("\n✓ Memory cleaned up")


def display_retrieval_results(
    results: List[Dict[str, Any]],
    normalize_embeddings: bool = True,
    max_text_length: int = 200
):
    """
    Display retrieval results in a formatted way
    
    :param results: list of retrieval results
    :param normalize_embeddings: whether embeddings are normalized
    :param max_text_length: maximum length of text to display
    """
    if not results:
        print("No results found")
        return
    
    print(f"\n{'='*60}")
    print(f"Top {len(results)} Retrieval Results")
    print(f"{'='*60}")
    
    for idx, result in enumerate(results, 1):
        print(f"\nResult {idx}:")
        print(f"  ID: {result['id']}")
        
        # Score interpretation
        score_direction = 'Higher is better' if normalize_embeddings else 'Lower is better'
        print(f"  Score: {result['score']:.4f} ({score_direction})")
        
        # Text preview
        text = result['text']
        if len(text) > max_text_length:
            text_display = text[:max_text_length] + "..."
        else:
            text_display = text
        print(f"  Text: {text_display}")
    
    print(f"\n{'='*60}")


# ==================== Main ====================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("RAG Retriever Demo")
    print("="*60)
    
    # 1. Initialize retriever
    storage_path = RAGRetrieverConfig.DEFAULT_PERSIST_PATH
    retriever = SimpleRagRetriever(persist_path=storage_path)

    # 2. Load and index documents if needed
    load_and_index_documents(retriever)

    # 3. The retriever is ready - execute queries
    print("\n" + "="*60)
    print("Executing Sample Queries")
    print("="*60)
    
    # Query 1
    query1 = "What is the theory of relativity?"
    top_k_results1 = retriever.retrieve(query1, k=3)
    display_retrieval_results(
        top_k_results1,
        normalize_embeddings=retriever.normalize_embeddings
    )

    # Query 2
    query2 = "Who was the first person on the moon?"
    top_k_results2 = retriever.retrieve(query2, k=3)
    display_retrieval_results(
        top_k_results2,
        normalize_embeddings=retriever.normalize_embeddings
    )
    
    print("\n" + "="*60)
    print("Demo Complete")
    print("="*60)