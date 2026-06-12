"""RAGLLMCompressor: extends LLMCompressor with FAISS retrieval context.

Usage:
    compressor = RAGLLMCompressor(model, tokenizer, retriever)

    # compress
    cd, retrieval_ids = compressor.compress_text(text, top_k=3)

    # decompress (retrieval_ids must be stored alongside cd)
    recovered_text = compressor.decompress_text(cd, retrieval_ids)

Retrieval overhead (bits to encode the document IDs) is computed and stored in
cd.metadata so callers can account for it in compression ratio calculations.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from compression.llm_compressor import LLMCompressor
from compression.types import CompressedData, PromptContext


class RAGLLMCompressor(LLMCompressor):

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        retriever,                          # utils.rag_utils.SimpleRagRetriever
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(model, tokenizer, device)
        self.retriever = retriever

    # ------------------------------------------------------------------
    # Text-level API (handles tokenisation internally)
    # ------------------------------------------------------------------

    def compress_text(
        self,
        text: str,
        top_k: int = 3,
        context_mode: str = "tokens",
    ) -> Tuple[CompressedData, List[int]]:
        """Compress a plain-text document with RAG context.

        Returns (CompressedData, retrieval_ids).
        retrieval_ids must be stored by the caller for decompression.
        """
        results = self.retriever.retrieve(text, k=top_k)
        retrieval_ids = [r["id"] for r in results]

        prompt_ctx = self._build_prompt_ctx(results, context_mode)
        input_ids = self._tokenize(text)
        attention_mask = torch.ones_like(input_ids)

        cd = self.compress_batch(input_ids, attention_mask, prompt_ctx=prompt_ctx)[0]

        # annotate with retrieval overhead
        pool_size = self.retriever.index.ntotal if self.retriever.index else 0
        cd.metadata["retrieval_ids"] = retrieval_ids
        cd.metadata["retrieval_overhead_bytes"] = _retrieval_overhead_bytes(
            pool_size, len(retrieval_ids)
        )
        return cd, retrieval_ids

    def decompress_text(
        self,
        compressed: CompressedData,
        retrieval_ids: List[int],
        context_mode: str = "tokens",
    ) -> str:
        """Decompress using stored retrieval IDs to reconstruct the same context."""
        results = [
            {"id": rid, "text": self.retriever.doc_store[rid]}
            for rid in retrieval_ids
            if rid in self.retriever.doc_store
        ]
        prompt_ctx = self._build_prompt_ctx(results, context_mode)
        token_ids = self.decompress_batch([compressed], prompt_ctx=prompt_ctx)[0]
        return self.tokenizer.decode(token_ids[0].tolist(), skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> torch.Tensor:
        enc = self.tokenizer(text, return_tensors="pt")
        return enc["input_ids"].to(self.device)

    def _build_prompt_ctx(self, results: list, context_mode: str) -> PromptContext:
        if context_mode == "tokens":
            ctx_text = " ".join(r["text"] for r in results)
            ctx_ids = self.tokenizer(ctx_text, return_tensors="pt")["input_ids"].to(self.device)
            return PromptContext(mode="tokens", token_ids=ctx_ids)
        elif context_mode == "embeds":
            # Caller is responsible for providing an encoder; for now raise a
            # clear error so future experiments know the hook point.
            raise NotImplementedError(
                "Latent embedding context mode is not yet implemented. "
                "Provide a context encoder and override _build_prompt_ctx."
            )
        raise ValueError(f"Unknown context_mode: {context_mode!r}")


def _retrieval_overhead_bytes(pool_size: int, num_retrieved: int) -> int:
    if pool_size <= 1 or num_retrieved <= 0:
        return 0
    bits_per_id = math.ceil(math.log2(pool_size))
    return math.ceil(bits_per_id * num_retrieved / 8)
