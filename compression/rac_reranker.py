"""Reranker: predict a base chunk's compression value for a query, to pick top-m.

After the encoder is trained, retrieval returns a large top-k.  Conditioning on
all k is expensive and noisy, so a reranker scores each candidate and keeps the m
most useful.  "Useful" is defined by the encoder-training by-product: a chunk's
*marginal CE reduction* ``Δ_j = L_0 - L_j`` (how much conditioning on chunk j
alone lowers the data's coding cost).  The reranker regresses to (per-query
standardised) Δ and is used at inference to order candidates.

The scoring model is a light bi-encoder interaction MLP over pooled feature
vectors, so it is modality-agnostic: any per-chunk feature (here, mean-pooled
frozen-LM input embeddings) works.  :class:`RerankerScorer` wires the trained
module to a feature source and exposes the ``.rank(query_ids, cand_ids)`` method
that :class:`RACCompressor` expects.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from utils.text_utils import pad_token_ids


# ---------------------------------------------------------------------------
# Feature extraction (mean-pooled frozen-LM input embeddings)
# ---------------------------------------------------------------------------

@torch.no_grad()
def mean_pool_embeddings(
    model, token_id_lists: List[List[int]], pad_id: int, device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Mean-pool a frozen LM's input embeddings over each token list -> [N, hidden]."""
    embed = model.get_input_embeddings()
    out: List[torch.Tensor] = []
    for s in range(0, len(token_id_lists), batch_size):
        chunk = token_id_lists[s:s + batch_size]
        ids, mask = pad_token_ids(chunk, pad_id, device=device)
        emb = embed(ids).float()                       # [b, L, H]
        m = mask.unsqueeze(-1).float()
        pooled = (emb * m).sum(1) / m.sum(1).clamp(min=1)
        out.append(pooled)
    return torch.cat(out, dim=0)


# ---------------------------------------------------------------------------
# Reranker model
# ---------------------------------------------------------------------------

class Reranker(nn.Module):
    """Bi-encoder interaction scorer: (query_feat, cand_feat) -> scalar value."""

    def __init__(self, dim: int, proj_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.config = dict(dim=dim, proj_dim=proj_dim, dropout=dropout)
        self.q_proj = nn.Linear(dim, proj_dim)
        self.c_proj = nn.Linear(dim, proj_dim)
        self.score = nn.Sequential(
            nn.Linear(4 * proj_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 1),
        )

    def forward(self, q_feat: torch.Tensor, c_feat: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(q_feat)
        c = self.c_proj(c_feat)
        inter = torch.cat([q, c, q * c, (q - c).abs()], dim=-1)
        return self.score(inter).squeeze(-1)           # [N]

    # -- persistence -----------------------------------------------------

    def save_pretrained(self, dirpath: str) -> None:
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, "reranker_config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
        torch.save(self.state_dict(), os.path.join(dirpath, "reranker.pt"))

    @classmethod
    def from_pretrained(cls, dirpath: str, map_location=None) -> "Reranker":
        with open(os.path.join(dirpath, "reranker_config.json")) as f:
            config = json.load(f)
        model = cls(**config)
        model.load_state_dict(
            torch.load(os.path.join(dirpath, "reranker.pt"), map_location=map_location))
        return model


# ---------------------------------------------------------------------------
# Inference-time scorer used by RACCompressor
# ---------------------------------------------------------------------------

class RerankerScorer:
    """Wraps a trained :class:`Reranker` with a feature source for inference.

    Base-chunk features are cached lazily across calls.
    """

    def __init__(self, reranker: Reranker, model, base_tokens: List[List[int]],
                 pad_id: int, device):
        self.reranker = reranker.to(device).eval()
        self.model = model
        self.base_tokens = base_tokens
        self.pad_id = pad_id
        self.device = device
        self._base_feat: Dict[int, torch.Tensor] = {}

    def _query_feat(self, query_ids: List[int]) -> torch.Tensor:
        return mean_pool_embeddings(self.model, [query_ids], self.pad_id, self.device)[0]

    def _cand_feats(self, cand_ids: Sequence[int]) -> torch.Tensor:
        missing = [c for c in cand_ids if c not in self._base_feat]
        if missing:
            feats = mean_pool_embeddings(
                self.model, [self.base_tokens[c] for c in missing], self.pad_id, self.device)
            for c, f in zip(missing, feats):
                self._base_feat[c] = f
        return torch.stack([self._base_feat[c] for c in cand_ids], dim=0)

    @torch.no_grad()
    def rank(self, query_ids: List[int], cand_ids: Sequence[int]) -> List[int]:
        if len(cand_ids) <= 1:
            return list(cand_ids)
        qf = self._query_feat(query_ids).unsqueeze(0).expand(len(cand_ids), -1)
        cf = self._cand_feats(cand_ids)
        scores = self.reranker(qf, cf)
        order = torch.argsort(scores, descending=True).tolist()
        return [cand_ids[i] for i in order]
