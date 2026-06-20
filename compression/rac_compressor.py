"""RACCompressor: retrieval-augmented compression on top of a frozen-LM compressor.

Wraps :class:`LLMCompressor` (the generic "backbone compressor") and adds the
retrieval-conditioning prefix:

    retrieved base chunks ──► ChunkEncoder ──► latent prefix ──► PromptContext(embeds)

Each data chunk gets its *own* prefix (its own retrieved context), so the prefix
tensor is per-sequence ``[B, m*n_latents, hidden]`` — the backbone's embeds path
accepts batched prefixes as long as every sequence shares the same prefix length,
which holds because we always condition on a fixed ``m`` chunks.

Codec note
----------
Retrieval is conditioned on the data being compressed, which the decoder does not
have.  So the selected base chunk ids are stored in ``CompressedData.metadata
["ctx_ids"]`` and transmitted as side information; the decoder re-encodes those
same base chunks (shared codec state) to reconstruct an identical prefix.  The id
overhead is tiny relative to the chunk but must be counted for honest ratios —
see ``eval_rac.py`` / :meth:`ctx_overhead_bytes`.

The wrapper is deliberately backbone-agnostic (encoder + base tokens + an embeds
capable compressor) so the planned bGPT/audio/image variant reuses it unchanged
once BGPTCompressor grows an ``inputs_embeds`` path.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch

from compression.llm_compressor import LLMCompressor
from compression.rac_encoder import ChunkEncoder
from compression.types import CompressedData, PromptContext
from utils.text_utils import pad_token_ids


class RACCompressor:

    def __init__(
        self,
        llm_compressor: LLMCompressor,
        encoder: ChunkEncoder,
        base_tokens: List[List[int]],   # base_tokens[chunk_id] = token ids
        m: int = 4,
        reranker=None,                  # optional: .rank(query_ids, cand_ids) -> ordered ids
        device: Optional[torch.device] = None,
    ) -> None:
        self.llm = llm_compressor
        self.model = llm_compressor.model
        self.tokenizer = llm_compressor.tokenizer
        self.device = device or llm_compressor.device
        self.encoder = encoder.to(self.device).eval()
        self.base_tokens = base_tokens
        self.m = m
        self.reranker = reranker
        self.pad_id = self.tokenizer.pad_token_id or 0
        self._model_dtype = next(self.model.parameters()).dtype

    @property
    def n_latents(self) -> int:
        return self.encoder.n_latents

    # ------------------------------------------------------------------
    # Context selection + encoding
    # ------------------------------------------------------------------

    def _select(self, cand_ids: Sequence[int], query_ids: Optional[List[int]]) -> List[int]:
        """Pick m base ids from the k candidates (rerank if available), pad to m."""
        ordered = list(cand_ids)
        if self.reranker is not None and query_ids is not None:
            ordered = self.reranker.rank(query_ids, ordered)
        sel = ordered[: self.m]
        if not sel:
            raise ValueError("RACCompressor: empty candidate list for a query")
        while len(sel) < self.m:        # repeat to keep a uniform prefix length
            sel.append(sel[-1])
        return sel

    @torch.no_grad()
    def _prefix_embeds(self, ctx_ids_lists: List[List[int]]) -> torch.Tensor:
        """Encode B*m base chunks into a [B, m*n_latents, hidden] prefix tensor."""
        B = len(ctx_ids_lists)
        flat = [self.base_tokens[cid] for ctx in ctx_ids_lists for cid in ctx]
        ids, mask = pad_token_ids(flat, self.pad_id, device=self.device)
        emb = self.model.get_input_embeddings()(ids).float()
        latents = self.encoder(emb, mask)                     # [B*m, n_lat, H]
        H = latents.shape[-1]
        return latents.view(B, self.m * self.n_latents, H).to(self._model_dtype)

    def ctx_overhead_bytes(self, n_base: int) -> int:
        """Bytes to transmit one chunk's m context ids (fixed-width per id)."""
        return self.m * max(1, math.ceil(math.log2(max(n_base, 2)) / 8))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(
        self,
        data_token_lists: List[List[int]],
        cand_ids_lists: List[Sequence[int]],
    ) -> List[CompressedData]:
        """Compress B data chunks, each conditioned on its own retrieved context.

        ``cand_ids_lists[b]`` are the top-k retrieval candidates for chunk b
        (reranked down to m internally).
        """
        ctx_ids_lists = [
            self._select(cands, data_token_lists[b])
            for b, cands in enumerate(cand_ids_lists)
        ]
        embeds = self._prefix_embeds(ctx_ids_lists)
        prompt_ctx = PromptContext(mode="embeds", embeds=embeds)

        input_ids, attn = pad_token_ids(data_token_lists, self.pad_id, device=self.device)
        cds = self.llm.compress_batch(input_ids, attn, prompt_ctx)
        for cd, ctx in zip(cds, ctx_ids_lists):
            cd.metadata["ctx_ids"] = list(ctx)
        return cds

    def decompress(
        self,
        compressed_list: List[CompressedData],
        show_progress: bool = False,
    ) -> List[torch.Tensor]:
        """Reconstruct B chunks; the prefix is rebuilt from metadata ctx_ids."""
        ctx_ids_lists = [cd.metadata["ctx_ids"] for cd in compressed_list]
        embeds = self._prefix_embeds(ctx_ids_lists)
        prompt_ctx = PromptContext(mode="embeds", embeds=embeds)
        return self.llm.decompress_batch(compressed_list, prompt_ctx, show_progress=show_progress)
