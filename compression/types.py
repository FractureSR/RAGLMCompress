"""Shared data types for the compression module."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class CompressedData:
    """Output of any compressor's compress() call."""
    compressed_bytes: bytes
    original_length: int        # number of tokens / byte-tokens to reconstruct
    metadata: dict = field(default_factory=dict)  # e.g. ext, retrieval_ids

    @property
    def compressed_length(self) -> int:
        return len(self.compressed_bytes)


@dataclass
class PromptContext:
    """Describes the prompt prepended before the data to be compressed.

    Two modes are supported:

    tokens  — plain token IDs prepended in the token sequence.
              prefix_length is set to the number of prompt tokens so the coder
              skips them.

    embeds  — pre-computed embedding tensor injected before the data embeddings.
              The model is called with inputs_embeds instead of input_ids.
              prefix_length is set to the number of context embedding slots.

    In both cases the same PromptContext must be used for compress and decompress
    to guarantee identical model conditioning.
    """
    mode: str                               # "tokens" or "embeds"
    # --- tokens mode ---
    token_ids: Optional[torch.Tensor] = None   # [1, prompt_len] or [B, prompt_len]
    # --- embeds mode ---
    embeds: Optional[torch.Tensor] = None      # [1, ctx_len, hidden] or [B, ...]

    def prefix_length(self) -> int:
        if self.mode == "tokens":
            assert self.token_ids is not None
            return self.token_ids.shape[1]
        elif self.mode == "embeds":
            assert self.embeds is not None
            return self.embeds.shape[1]
        raise ValueError(f"Unknown PromptContext mode: {self.mode!r}")
