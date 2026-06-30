"""LLMCompressor: token-level compression using a causal language model.

Supports two prompt context modes (see PromptContext):
  tokens — retrieved or hand-crafted token IDs prepended before the data.
  embeds — pre-computed latent embeddings injected before data embeddings;
            model is called with inputs_embeds so no token IDs are needed
            for the context portion.

Compress path (prefill):
  1. Build full input [prompt | data] (tokens or embeds).
  2. Single forward pass → logits for every position.
  3. Arithmetic-code only the data portion (skip prefix_length positions).

Decompress path (iterative, with padding trick):
  At step i:
    - Pad decoded-so-far to [prompt | sym_0 … sym_{i-1} | 0 … 0] (full length).
    - Single batched forward pass → use logits at position prefix_length+i-1.
    - Decode next symbol per sequence.
  This keeps the numerical computation identical to the prefill pass.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from compression.base_compressor import BaseCompressor
from compression.types import CompressedData, LMScore, PromptContext


class LLMCompressor(BaseCompressor):

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.eval()

    # ------------------------------------------------------------------
    # Shared prefill (one forward over [prompt | data]) — used by both the
    # arithmetic-coding path (compress_batch) and the score-only path
    # (score_batch), so the bits the latter reports match what the former emits.
    # ------------------------------------------------------------------

    def _prefill(
        self,
        input_ids: torch.Tensor,                  # [B, max_data_len]  right-padded data
        prompt_ctx: Optional[PromptContext],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Single forward over ``[prompt | data]``.

        Returns ``(full_ids, logits, prefix_length)`` where ``full_ids`` is the
        token tensor handed to the coder (the context portion is dummy zeros in
        ``embeds`` mode) and ``logits[b, j]`` already predicts ``full_ids[b, j+1]``
        (i.e. it is sliced to ``[:, :-1, :]``).

        No attention mask is passed to the model so the forward pass mirrors the
        decode loop (which also runs without a mask, relying on causal attention).
        Causal attention ensures ``logits[b, j, :]`` depends only on
        ``full_ids[b, 0:j+1]``, making right-padding safe. In ``tokens`` mode
        ``prompt_ctx.token_ids`` may be ``[1, P]`` (shared prompt) or ``[B, P]``
        (a different prompt per row).
        """
        B = input_ids.shape[0]
        input_ids = input_ids.to(self.device)

        if prompt_ctx is None:
            prefix = torch.full((B, 1), self._dummy_token(), dtype=torch.long, device=self.device)
            prefix_length = 1
            full_ids = torch.cat([prefix, input_ids], dim=1)
            with torch.inference_mode():
                logits = self.model(full_ids, use_cache=False).logits[:, :-1, :].float()

        elif prompt_ctx.mode == "tokens":
            ctx_ids = prompt_ctx.token_ids.to(self.device).expand(B, -1)
            prefix_length = prompt_ctx.prefix_length()
            full_ids = torch.cat([ctx_ids, input_ids], dim=1)
            with torch.inference_mode():
                logits = self.model(full_ids, use_cache=False).logits[:, :-1, :].float()

        elif prompt_ctx.mode == "embeds":
            ctx_embeds = prompt_ctx.embeds.to(self.device).expand(B, -1, -1)
            prefix_length = prompt_ctx.prefix_length()
            with torch.inference_mode():
                data_embeds = self.model.get_input_embeddings()(input_ids)
                full_embeds = torch.cat([ctx_embeds, data_embeds], dim=1)
                logits = self.model(inputs_embeds=full_embeds, use_cache=False).logits[:, :-1, :].float()
            # Reconstruct token tensor for _encode_sequence (context portion is dummy zeros)
            dummy_prefix = torch.zeros(B, prefix_length, dtype=torch.long, device=self.device)
            full_ids = torch.cat([dummy_prefix, input_ids], dim=1)

        else:
            raise ValueError(f"Unknown prompt mode: {prompt_ctx.mode!r}")

        return full_ids, logits, prefix_length

    # ------------------------------------------------------------------
    # Batch API  (bs=1 is the degenerate case — pass a [1, L] tensor)
    # ------------------------------------------------------------------

    def compress_batch(
        self,
        input_ids: torch.Tensor,       # [B, max_seq_len]  — right-padded data tokens
        attention_mask: torch.Tensor,  # [B, max_seq_len]  — used to read per-seq lengths
        prompt_ctx: Optional[PromptContext] = None,
    ) -> List[CompressedData]:
        """Compress B sequences in a single model forward pass.

        Each sequence gets its own CompressedData.  The model is called once on
        the full right-padded batch (via :meth:`_prefill`); per-sequence logits
        are extracted afterwards.
        """
        seq_lens: List[int] = attention_mask.sum(dim=1).long().tolist()
        full_ids, logits, prefix_length = self._prefill(input_ids, prompt_ctx)

        results: List[CompressedData] = []
        for b in range(input_ids.shape[0]):
            L = int(seq_lens[b])
            seq_ids    = full_ids[b:b+1, :prefix_length + L]
            seq_logits = logits[b:b+1, :prefix_length + L - 1, :]
            cd = self._encode_sequence(seq_ids, seq_logits, prefix_length)
            cd.metadata["prefix_length"] = prefix_length
            cd.metadata["prompt_mode"] = prompt_ctx.mode if prompt_ctx else "none"
            results.append(cd)
        return results

    def score_batch(
        self,
        input_ids: torch.Tensor,       # [B, max_seq_len]  — right-padded data tokens
        attention_mask: torch.Tensor,  # [B, max_seq_len]  — used to read per-seq lengths
        prompt_ctx: Optional[PromptContext] = None,
    ) -> List[LMScore]:
        """Code length (bits) + per-token NLL for each sequence — no coding.

        The score-only counterpart of :meth:`compress_batch`: it shares
        :meth:`_prefill`, so each ``LMScore.bits`` equals the data code length the
        coder would emit (up to the few bytes of range-coder overhead). Use it to
        *rank* prompt contexts by exact code length (RAC's oracle) and to inspect
        per-token surprise (RAC's cascade) without paying for the range coder.
        """
        seq_lens: List[int] = attention_mask.sum(dim=1).long().tolist()
        full_ids, logits, prefix_length = self._prefill(input_ids, prompt_ctx)

        results: List[LMScore] = []
        for b in range(input_ids.shape[0]):
            L = int(seq_lens[b])
            target = full_ids[b, prefix_length: prefix_length + L]               # [L]
            lg = logits[b, prefix_length - 1: prefix_length - 1 + L, :]          # [L, V]
            nll = -lg.log_softmax(dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
            results.append(LMScore(bits=float(nll.sum().item()) / math.log(2), token_nll=nll))
        return results

    def decompress_batch(
        self,
        compressed_list: List[CompressedData],
        prompt_ctx: Optional[PromptContext] = None,
        show_progress: bool = False,
    ) -> List[torch.Tensor]:
        """Decode B compressed sequences, one batched forward pass per token step."""
        B = len(compressed_list)

        if prompt_ctx is None:
            start_tokens = torch.full(
                (B, 1), self._dummy_token(), dtype=torch.long, device=self.device,
            )
            def get_logits_fn(buf: torch.Tensor) -> torch.Tensor:
                with torch.inference_mode():
                    return self.model(buf, use_cache=False).logits[:, :-1, :].float()

        elif prompt_ctx.mode == "tokens":
            start_tokens = prompt_ctx.token_ids.to(self.device).expand(B, -1)
            def get_logits_fn(buf: torch.Tensor) -> torch.Tensor:
                with torch.inference_mode():
                    return self.model(buf, use_cache=False).logits[:, :-1, :].float()

        elif prompt_ctx.mode == "embeds":
            ctx_embeds = prompt_ctx.embeds.to(self.device).expand(B, -1, -1)
            prefix_length = prompt_ctx.prefix_length()
            start_tokens = torch.zeros(B, prefix_length, dtype=torch.long, device=self.device)
            def get_logits_fn(buf: torch.Tensor) -> torch.Tensor:
                data_ids = buf[:, prefix_length:]
                with torch.inference_mode():
                    data_embeds = self.model.get_input_embeddings()(data_ids)
                    full_embeds = torch.cat([ctx_embeds, data_embeds], dim=1)
                    return self.model(inputs_embeds=full_embeds, use_cache=False).logits[:, :-1, :].float()

        else:
            raise ValueError(f"Unknown prompt mode: {prompt_ctx.mode!r}")

        decoded = self._decode_batch(compressed_list, get_logits_fn, start_tokens, self.device,
                                     show_progress=show_progress)
        return [
            torch.tensor(d, dtype=torch.long, device=self.device).unsqueeze(0)
            for d in decoded
        ]

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _dummy_token(self) -> int:
        tid = self.tokenizer.bos_token_id
        return tid if tid is not None else 0
