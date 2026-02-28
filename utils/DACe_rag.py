import faiss
import numpy as np
import os
import pickle
import time
import gc
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import dac
import torch 
import torchaudio
import tqdm 

# ==================== Configuration ====================
class DACERAGRetrieverConfig:
    """Configuration for DACe RAG retriever"""
    # Model configuration
    DACE_MODEL_TYPE = "16khz"
    DACE_HIDDEN_SIZE = 1024
    
    # Storage configuration
    DEFAULT_PERSIST_PATH = "retriever_cache/audio_dace_storage"
    INDEX_FILENAME = "audio_faiss_index.bin"
    DOC_STORE_FILENAME = "audio_path_store.pkl"

    SAMPLE_RATE = 16000
    
    # Retrieval configuration
    DEFAULT_TOP_K = 1


# ==================== RAG Retriever Class ====================
class DACERagRetriever:
    """Simple DACe RAG retriever using FAISS and SentenceTransformer"""
    
    def __init__(
        self,
        model_type: str = None, 
        persist_path: str = None
    ):
        """
        Initialize RAG retriever
        
        :param model_name: embedding model name or path (default from config)
        :param normalize_embeddings: whether to normalize embeddings (default from config)
        :param persist_path: path to persist/load index (default from config)
        """
        if model_type is None:
            model_type = DACERAGRetrieverConfig.DEFAULT_MODEL_NAME
        if persist_path is None:
            persist_path = DACERAGRetrieverConfig.DEFAULT_PERSIST_PATH
        
        print("="*60)
        print("Initializing DACe RAG Retriever")
        print("="*60)
        print(f"Loading embedding model: {model_type}")
        
        # Load DACe
        model_path = dac.utils.download(model_type=model_type)
        self.encoder_model = dac.DAC.load(model_path)
        self.encoder_model.to('cuda').eval()

        self.embedding_dim = DACERAGRetrieverConfig.DACE_HIDDEN_SIZE
        self.persist_path = persist_path
        
        print(f"✓ Embedding model loaded")
        print(f"  Embedding dimension: {self.embedding_dim}")

        self.index_file = os.path.join(persist_path, DACERAGRetrieverConfig.INDEX_FILENAME)
        self.doc_store_file = os.path.join(persist_path, DACERAGRetrieverConfig.DOC_STORE_FILENAME)

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
            # use L2 for raw latent vectors
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.embedding_dim))
        
        print("="*60)

    def _get_dace_embedding(self, waveform: torch.Tensor) -> np.ndarray:
        """Helper to extract single fixed size vector from DACe"""
        with torch.no_grad():
            # z shape: [Batch, 1024, Frames]
            z = self.encoder_model.encoder(waveform.to('cuda'))
            # Mean pool acros time to get single vector per clip 
            embedding = z.mean(dim=-1).cpu().numpy()
        return embedding.astype('float32')

    def _index_audio_files(self, file_paths: List[str]):
        """Build the FAISS index from a list of audio file paths"""
        print(f"Indexing {len(file_paths)} files...")
        embeddings_list = []
        ids_to_add = []

        for path in tqdm(file_paths, desc="Indexing Audio"):
            # load and resample if needed 
            signal, sr = torchaudio.load(path)
            if sr != DACERAGRetrieverConfig.SAMPLE_RATE:
                signal = torchaudio.transforms.Resample(sr, DACERAGRetrieverConfig.SAMPLE_RATE)(signal)
            
            # Ensure 1-channel for DACe
            if signal.shape[0] > 1: 
                signal = signal.mean(dim=0, keepdim=True)
            
            emb = self._get_dace_embedding(signal.unsqueeze(0))
            self.doc_store[self.next_id] = path 
            embeddings_list.append(emb)
            ids_to_add.append(self.next_id)
            self.next_id += 1
        
        # update FAISS
        self.index.add_with_ids(np.vstack(embeddings_list), np.array(ids_to_add, dtype=np.int64))
        self._save_index()



    
    def _save_index(self):
        """Save the FAISS index and audio store to disk"""
        if not self.persist_path:
            return

        print(f"\nSaving index to '{self.persist_path}'...")
        os.makedirs(self.persist_path, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, self.index_file)

        # Save doc_store and next_id using pickle
        with open(self.doc_store_file, "rb") as f:
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


    def retrieve(self, query_waveform: torch.Tensor, k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant audio clips for a query
        
        :param query: search query
        :param k: number of results to return (default from config)
        :return: list of retrieval results with id, text, and score
        """
        if k is None:
            k = RAGRetrieverConfig.DEFAULT_TOP_K
        
        query_emb = self._get_dace_embedding(query_waveform)
        distances, indices = self.index.search(query_emb, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append({"id": int(idx), 
                                "path": self.doc_store[int(idx)], 
                                "score": float(dist)
                            })
        return results

    
# ==================== Helper Functions ====================
def load_and_index_documents(
    retriever: DACERagRetriever,
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