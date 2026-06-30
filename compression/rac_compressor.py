"""Oracle retrieval-augmented compression over token prefixes."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, TypeVar

import torch

from compression.llm_compressor import LLMCompressor
from compression.rac_index import FixedIndexCoder
from compression.types import CompressedData, LMScore, PromptContext
from utils.text_utils import pad_token_ids


T = TypeVar("T")


@dataclass
class _Piece:
    idx: int
    data: List[int]
    pool: List[int]
    choices: List[int] = field(default_factory=list)
    used: Set[int] = field(default_factory=set)
    gains: List[float] = field(default_factory=list)
    net_gains: List[float] = field(default_factory=list)
    id_bits: List[float] = field(default_factory=list)
    baseline_bits: float = 0.0
    bits: float = 0.0
    nll: Optional[torch.Tensor] = None


class RACCompressor:
    """Select retrieved token prefixes by exact LM code length, then encode."""

    def __init__(
        self,
        llm_compressor: LLMCompressor,
        base_tokens: List[List[int]],
        index_coder=None,
        max_ctx: int = 1024,
        margin_bits: float = 0.0,
        batch_size: Optional[int] = 1,
        cascade: bool = False,
        cascade_max_cond: int = 2,
        cascade_nll_thresh: float = 4.0,
        cascade_min_frac: float = 0.05,
        cascade_top_k: int = 16,
        retriever=None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.llm = llm_compressor
        self.model = llm_compressor.model
        self.tokenizer = llm_compressor.tokenizer
        self.device = device or llm_compressor.device
        self.base_tokens = base_tokens
        self.index_coder = index_coder or FixedIndexCoder(len(base_tokens))
        self.max_ctx = max_ctx
        self.margin_bits = margin_bits
        self.batch_size = max(1, batch_size or 1)
        self.cascade = cascade
        self.cascade_max_cond = max(1, cascade_max_cond)
        self.cascade_nll_thresh = cascade_nll_thresh
        self.cascade_min_frac = cascade_min_frac
        self.cascade_top_k = cascade_top_k
        self.retriever = retriever
        self.pad_id = self.tokenizer.pad_token_id or 0

    def compress_batch(
        self,
        data_token_lists: List[List[int]],
        cand_ids_lists: Sequence[Sequence[int]],
    ) -> List[CompressedData]:
        if not data_token_lists:
            return []

        pieces = [
            _Piece(idx=i, data=list(data), pool=list(cands))
            for i, (data, cands) in enumerate(zip(data_token_lists, cand_ids_lists))
        ]

        baseline = self._score_jobs([[]] * len(pieces), [p.data for p in pieces])
        for p, score in zip(pieces, baseline):
            p.baseline_bits = score.bits
            p.bits = score.bits
            p.nll = score.token_nll

        active = [p for p in pieces if p.pool]
        max_cond = self.cascade_max_cond if self.cascade else 1
        for level in range(max_cond):
            if not active:
                break
            active = self._oracle_step(active, level, max_cond)

        return self._encode_pieces(pieces)

    def decompress_batch(
        self,
        compressed_list: List[CompressedData],
    ) -> List[torch.Tensor]:
        if not compressed_list:
            return []

        out: List[Optional[torch.Tensor]] = [None] * len(compressed_list)
        by_plen: Dict[int, list] = defaultdict(list)
        for i, cd in enumerate(compressed_list):
            prefix = self._build_prefix(cd.metadata.get("ctx_ids", []))
            by_plen[len(prefix)].append((i, cd, prefix))

        for plen, group in by_plen.items():
            for start in range(0, len(group), self.batch_size):
                batch = group[start:start + self.batch_size]
                cds = [cd for _, cd, _ in batch]
                if plen == 0:
                    recs = self.llm.decompress_batch(cds)
                else:
                    prompt = PromptContext(
                        mode="tokens",
                        token_ids=torch.tensor(
                            [prefix for _, _, prefix in batch], dtype=torch.long,
                        ),
                    )
                    recs = self.llm.decompress_batch(cds, prompt)
                for (i, _, _), rec in zip(batch, recs):
                    out[i] = rec

        return self._filled(out)

    def index_overhead_bits(self, cd: CompressedData) -> float:
        return float(cd.metadata.get("index_bits", 0.0))

    def _build_prefix(self, choices: Sequence[int]) -> List[int]:
        prefix: List[int] = []
        for cid in choices:
            prefix.extend(self.base_tokens[cid])
        return prefix[: self.max_ctx]

    def _index_bits(self, choices: Sequence[int]) -> float:
        return (
            sum(self.index_coder.cost_bits(cid) for cid in choices)
            + self.index_coder.cost_bits(None)
        )

    @torch.no_grad()
    def _score_jobs(
        self,
        prefixes: List[List[int]],
        datas: List[List[int]],
    ) -> List[LMScore]:
        out: List[Optional[LMScore]] = [None] * len(prefixes)
        by_plen: Dict[int, List[int]] = defaultdict(list)
        for i, prefix in enumerate(prefixes):
            by_plen[len(prefix)].append(i)

        # by_plen is for batching by prefix length
        for plen, idxs in by_plen.items():
            idxs.sort(key=lambda i: len(datas[i]))
            for start in range(0, len(idxs), self.batch_size):
                batch = idxs[start:start + self.batch_size]
                input_ids, attn_mask = pad_token_ids(
                    [datas[i] for i in batch], self.pad_id, device=self.device,
                )
                if plen == 0:
                    scores = self.llm.score_batch(input_ids, attn_mask)
                else:
                    prompt = PromptContext(
                        mode="tokens",
                        token_ids=torch.tensor(
                            [prefixes[i] for i in batch], dtype=torch.long,
                        ),
                    )
                    scores = self.llm.score_batch(input_ids, attn_mask, prompt)
                for i, score in zip(batch, scores):
                    out[i] = score

        return self._filled(out)

    def _oracle_step(
        self,
        active: List[_Piece],
        level: int,
        max_cond: int,
    ) -> List[_Piece]:
        prefixes: List[List[int]] = []
        datas: List[List[int]] = []
        for p in active:
            for cid in p.pool:
                prefixes.append(self._build_prefix(p.choices + [cid]))
                datas.append(p.data)

        scores = self._score_jobs(prefixes, datas)

        next_active: List[_Piece] = []
        retrieve_pieces: List[_Piece] = []
        retrieve_queries: List[str] = []
        retrieve_excludes: List[Set[int]] = []
        cursor = 0
        for p in active:
            k = len(p.pool)
            piece_scores = scores[cursor:cursor + k]
            cursor += k

            best = min(range(k), key=lambda i: piece_scores[i].bits)
            cid = p.pool[best]
            gain = p.bits - piece_scores[best].bits
            id_bits = self.index_coder.cost_bits(cid)
            if gain <= id_bits + self.margin_bits:
                continue

            p.pool.pop(best)
            p.choices.append(cid)
            p.used.add(cid)
            p.gains.append(gain)
            p.net_gains.append(gain - id_bits)
            p.id_bits.append(id_bits)
            p.bits = piece_scores[best].bits
            p.nll = piece_scores[best].token_nll

            if self.cascade and level + 1 < max_cond:
                query = self._cascade_query(p.data, p.nll)
                if query is None:
                    continue
                if self.retriever is None or not query:
                    if p.pool:
                        next_active.append(p)
                else:
                    retrieve_pieces.append(p)
                    retrieve_queries.append(query)
                    retrieve_excludes.append(set(p.used))

        if retrieve_queries:
            hits_many = self.retriever.retrieve_many(
                retrieve_queries,
                top_k=self.cascade_top_k,
                exclude_ids=retrieve_excludes,
            )
            for p, hits in zip(retrieve_pieces, hits_many):
                p.pool = [cid for cid, _ in hits if cid not in p.used]
                if p.pool:
                    next_active.append(p)

        return next_active

    def _cascade_query(
        self,
        data_ids: List[int],
        nll: torch.Tensor,
    ) -> Optional[str]:
        high = (nll > self.cascade_nll_thresh).cpu()
        if float(high.float().mean().item()) < self.cascade_min_frac:
            return None

        hi_tokens = [
            token_id for token_id, is_high in zip(data_ids, high.tolist()) if is_high
        ]
        query = self.tokenizer.decode(hi_tokens, skip_special_tokens=True)
        return query if query.strip() else ""

    def _encode_pieces(self, pieces: List[_Piece]) -> List[CompressedData]:
        out: List[Optional[CompressedData]] = [None] * len(pieces)
        by_plen: Dict[int, List[_Piece]] = defaultdict(list)
        for p in pieces:
            by_plen[len(self._build_prefix(p.choices))].append(p)

        for plen, group in by_plen.items():
            group.sort(key=lambda p: len(p.data))
            for start in range(0, len(group), self.batch_size):
                batch = group[start:start + self.batch_size]
                input_ids, attn_mask = pad_token_ids(
                    [p.data for p in batch], self.pad_id, device=self.device,
                )
                if plen == 0:
                    cds = self.llm.compress_batch(input_ids, attn_mask)
                else:
                    prompt = PromptContext(
                        mode="tokens",
                        token_ids=torch.tensor(
                            [self._build_prefix(p.choices) for p in batch],
                            dtype=torch.long,
                        ),
                    )
                    cds = self.llm.compress_batch(input_ids, attn_mask, prompt)

                for p, cd in zip(batch, cds):
                    cd.metadata["ctx_ids"] = list(p.choices)
                    cd.metadata["index_bits"] = self._index_bits(p.choices)
                    cd.metadata["ctx_gain_bits"] = list(p.gains)
                    cd.metadata["ctx_net_gain_bits"] = list(p.net_gains)
                    cd.metadata["ctx_id_bits"] = list(p.id_bits)
                    cd.metadata["baseline_bits"] = p.baseline_bits
                    cd.metadata["data_bits"] = p.bits
                    out[p.idx] = cd

        return self._filled(out)

    @staticmethod
    def _filled(items: List[Optional[T]]) -> List[T]:
        filled: List[T] = []
        for item in items:
            assert item is not None
            filled.append(item)
        return filled
