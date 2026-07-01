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
class LMScore:
    """Code-length estimate for one sequence, *without* arithmetic coding.

    Returned by ``score_batch`` (the cheap counterpart of ``compress_batch``):
    the same forward pass and per-token NLL the coder would use, exposed so a
    caller can *rank* prompt contexts by exact code length and inspect where the
    model is still surprised — without paying for the range coder. ``bits`` is the
    ideal data code length (``sum(token_nll) / ln 2``); ``token_nll`` is the
    per-token negative log-likelihood in nats (used e.g. to find high-entropy
    residual tokens for RAC's cascade).
    """
    bits: float
    token_nll: torch.Tensor     # [data_len], nats


@dataclass
class PromptContext:
    """Describes the prompt prepended before the data to be compressed.

    tokens  — plain token IDs prepended in the token sequence. ``prefix_length``
              is the number of prompt tokens so the coder skips them. The same
              PromptContext must be used for compress and decompress to guarantee
              identical model conditioning.
    """
    mode: str                                   # "tokens"
    token_ids: Optional[torch.Tensor] = None    # [1, prompt_len] or [B, prompt_len]

    def prefix_length(self) -> int:
        if self.mode == "tokens":
            assert self.token_ids is not None
            return self.token_ids.shape[1]
        raise ValueError(f"Unknown PromptContext mode: {self.mode!r}")
