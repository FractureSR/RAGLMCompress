"""ChunkEncoder: Perceiver-style pooling of a retrieved chunk into a latent prefix.

A small set of learnable *latent queries* cross-attend over a chunk's token (or
byte/patch) embeddings and are refined by self-attention, producing a fixed
``n_latents`` vectors in the frozen backbone's embedding space.  Those vectors
are concatenated across the retrieved chunks and injected as the conditioning
prefix via ``PromptContext(mode="embeds")`` — see :class:`RACCompressor`.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import torch
import torch.nn as nn


class _PerceiverBlock(nn.Module):
    """Pre-norm cross-attention (latents attend to inputs) + self-attention + FFN."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.cross_q_norm = nn.LayerNorm(d_model)
        self.cross_kv_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.self_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, latents: torch.Tensor, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        q = self.cross_q_norm(latents)
        kv = self.cross_kv_norm(x)
        attn, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        latents = latents + attn

        s = self.self_norm(latents)
        sa, _ = self.self_attn(s, s, s, need_weights=False)
        latents = latents + sa

        latents = latents + self.ff(self.ff_norm(latents))
        return latents


class ChunkEncoder(nn.Module):
    """Encode a (padded) batch of chunk embeddings into ``n_latents`` prefix vectors."""

    def __init__(
        self,
        hidden_size: int,
        n_latents: int = 8,
        n_heads: int = 8,
        n_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.config = dict(
            hidden_size=hidden_size, n_latents=n_latents,
            n_heads=n_heads, n_layers=n_layers,
            ff_mult=ff_mult, dropout=dropout,
        )

        self.latents = nn.Parameter(torch.randn(n_latents, hidden_size) * 0.02)
        self.blocks = nn.ModuleList(
            [_PerceiverBlock(hidden_size, n_heads, ff_mult, dropout) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        embeds: torch.Tensor,                           # [B, L, hidden_size]
        attention_mask: Optional[torch.Tensor] = None,  # [B, L]  1 = keep, 0 = pad
    ) -> torch.Tensor:                                  # [B, n_latents, hidden_size]
        B = embeds.shape[0]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        # MultiheadAttention expects True where a key should be *ignored*.
        kpm = (attention_mask == 0) if attention_mask is not None else None
        for block in self.blocks:
            latents = block(latents, embeds, kpm)
        return self.out_norm(latents)

    @property
    def n_latents(self) -> int:
        return self.config["n_latents"]

    # -- persistence -----------------------------------------------------

    def save_pretrained(self, dirpath: str) -> None:
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, "encoder_config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
        torch.save(self.state_dict(), os.path.join(dirpath, "encoder.pt"))

    @classmethod
    def from_pretrained(cls, dirpath: str, map_location=None) -> "ChunkEncoder":
        with open(os.path.join(dirpath, "encoder_config.json")) as f:
            config = json.load(f)
        model = cls(**config)
        state = torch.load(os.path.join(dirpath, "encoder.pt"), map_location=map_location)
        model.load_state_dict(state)
        return model
