import faiss
import numpy as np
import os
import pickle
import torchaudio
from typing import List, Dict, Any
import torch
from transformers import EncodecModel, AutoProcessor
from tqdm import tqdm


# ==================== Configuration ====================
class EnCodecRAGConfig:
    ENCODEC_MODEL = "facebook/encodec_16khz"
    SAMPLE_RATE = 16000
    # EnCodec 24kHz encoder outputs 128-channel features
    ENCODER_DIM = 128
    DEFAULT_PERSIST_PATH = "retriever_cache/audio_encodec_storage"
    INDEX_FILENAME = "encodec_faiss_index.bin"
    DOC_STORE_FILENAME = "encodec_doc_store.pkl"
    DEFAULT_TOP_K = 1


# ==================== RAG Retriever ====================
class EnCodecRAGRetriever:
    """Audio RAG retriever using EnCodec encoder embeddings and FAISS."""

    def __init__(self, model_name: str = None, persist_path: str = None):
        if model_name is None:
            model_name = EnCodecRAGConfig.ENCODEC_MODEL
        if persist_path is None:
            persist_path = EnCodecRAGConfig.DEFAULT_PERSIST_PATH

        print("=" * 60)
        print("Initializing EnCodec RAG Retriever")
        print(f"Model: {model_name}")
        print("=" * 60)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.encodec = EncodecModel.from_pretrained(model_name)
        self.encodec.to(self.device).eval()
        for p in self.encodec.parameters():
            p.requires_grad = False

        self.embedding_dim = EnCodecRAGConfig.ENCODER_DIM
        self.persist_path = persist_path
        self.index_file = os.path.join(persist_path, EnCodecRAGConfig.INDEX_FILENAME)
        self.doc_store_file = os.path.join(persist_path, EnCodecRAGConfig.DOC_STORE_FILENAME)

        if os.path.exists(self.index_file) and os.path.exists(self.doc_store_file):
            print(f"Loading existing index from '{persist_path}'")
            self._load_index()
            print(f"Index loaded: {self.index.ntotal} entries")
        else:
            print(f"No existing index at '{persist_path}', initializing empty index")
            self.doc_store = {}
            self.next_id = 0
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.embedding_dim))

        print("=" * 60)

    @torch.no_grad()
    def _get_embedding(self, waveform: torch.Tensor) -> np.ndarray:
        """
        Get a fixed-size embedding for an audio waveform using EnCodec's encoder.
        waveform: [1, T] or [B, 1, T]
        returns: [B, ENCODER_DIM] as float32
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)  # [1, 1, T]
        waveform = waveform.to(self.device)
        # EncodecModel.encoder returns continuous features [B, D, T]
        enc_out = self.encodec.encoder(waveform)
        # Mean-pool over time → [B, D]
        embedding = enc_out.mean(dim=-1)
        return embedding.cpu().numpy().astype("float32")

    def index_audio_files(self, file_paths: List[str]):
        """Build FAISS index from a list of audio file paths."""
        print(f"Indexing {len(file_paths)} audio files...")
        embeddings_list = []
        ids_to_add = []

        for path in tqdm(file_paths, desc="Indexing audio"):
            signal, sr = torchaudio.load(path)
            if sr != EnCodecRAGConfig.SAMPLE_RATE:
                signal = torchaudio.functional.resample(signal, sr, EnCodecRAGConfig.SAMPLE_RATE)
            if signal.shape[0] > 1:
                signal = signal.mean(dim=0, keepdim=True)

            emb = self._get_embedding(signal)
            self.doc_store[self.next_id] = path
            embeddings_list.append(emb)
            ids_to_add.append(self.next_id)
            self.next_id += 1

        self.index.add_with_ids(
            np.vstack(embeddings_list), np.array(ids_to_add, dtype=np.int64)
        )
        self._save_index()
        print(f"Indexed {len(file_paths)} files. Total: {self.index.ntotal}")

    def retrieve(self, waveform: torch.Tensor, k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most similar audio files.
        waveform: [1, T] or [B, 1, T]
        returns: list of dicts with 'id', 'path', 'score'
        """
        if k is None:
            k = EnCodecRAGConfig.DEFAULT_TOP_K
        emb = self._get_embedding(waveform)
        distances, indices = self.index.search(emb, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append({
                    "id": int(idx),
                    "path": self.doc_store[int(idx)],
                    "score": float(dist),
                })
        return results

    def _save_index(self):
        os.makedirs(self.persist_path, exist_ok=True)
        faiss.write_index(self.index, self.index_file)
        with open(self.doc_store_file, "wb") as f:
            pickle.dump({"doc_store": self.doc_store, "next_id": self.next_id}, f)
        print(f"Index saved to '{self.persist_path}'")

    def _load_index(self):
        self.index = faiss.read_index(self.index_file)
        with open(self.doc_store_file, "rb") as f:
            data = pickle.load(f)
        self.doc_store = data["doc_store"]
        self.next_id = data["next_id"]
