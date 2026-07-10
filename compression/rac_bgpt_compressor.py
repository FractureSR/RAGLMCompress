"""Oracle retrieval-augmented compression over byte prefixes (bGPT).

The byte-domain analog of ``rac_llm_compressor.RACLLMCompressor``: it is to
``BGPTCompressor`` exactly what ``RACLLMCompressor`` is to ``LLMCompressor``.
Structurally it mirrors ``RACLLMCompressor`` step for step (baseline score → oracle
levels → encode; group jobs by prefix length; keep the best of top-k only if it
beats its transmitted-id cost, else no condition). The only differences are the
ones that separate ``BGPTCompressor`` from ``LLMCompressor``:

  - A data piece is a byte ``segment`` ``(payload_bytes, ext)`` rather than a
    list of LM token ids.
  - The retrieved prefix is a run of **byte tokens** (whole base patches) passed
    through ``BGPTCompressor``'s ``prefixes=`` path instead of a token
    ``PromptContext``.
  - bGPT pads a batch itself (PAD_TOKEN), so there is no tokenizer / pad-id
    bookkeeping.
  - The cascade query built from high-surprise residual is raw ``bytes`` (for a
    byte-domain retriever) instead of detokenised text.

No training. The chosen base-chunk ids are transmitted as side information; their
bit cost is charged by the same ``rac_index`` coders used for text.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple, TypeVar

import torch

from compression.bgpt_compressor import BGPTCompressor
from compression.rac_index import FixedIndexCoder
from compression.types import CompressedData, LMScore
from utils.bgpt_codec_utils import PAD_TOKEN, bytes_to_padded_tokens


T = TypeVar("T")


@dataclass
class _Piece:
    idx: int
    data: bytes
    ext: str
    pool: List[int]
    choices: List[int] = field(default_factory=list)
    used: Set[int] = field(default_factory=set)
    gains: List[float] = field(default_factory=list)
    net_gains: List[float] = field(default_factory=list)
    id_bits: List[float] = field(default_factory=list)
    baseline_bits: float = 0.0
    bits: float = 0.0
    nll: Optional[torch.Tensor] = None


class RACBGPTCompressor:
    """Select retrieved byte prefixes by exact bGPT code length, then encode."""

    def __init__(
        self,
        bgpt_compressor: BGPTCompressor,
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
        chunk_size: Optional[int] = None,
        show_progress: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        self.bgpt = bgpt_compressor
        self.model = bgpt_compressor.model
        self.patch_size = bgpt_compressor.patch_size
        self.device = device or bgpt_compressor.device
        self.show_progress = show_progress
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
        # At most this many conditions are ever prepended per piece (one per
        # cascade level; exactly one when cascade is off).
        self.max_cond = self.cascade_max_cond if self.cascade else 1
        # Each retrieved base chunk is a whole number of patches, so any prefix
        # the oracle builds starts the payload on a patch boundary (BGPTCompressor
        # asserts this). A chunk split mid-patch would misalign every later patch.
        for cid, chunk in enumerate(base_tokens):
            assert chunk and len(chunk) % self.patch_size == 0, (
                f"base chunk {cid} has length {len(chunk)}; every base chunk must "
                f"be a positive multiple of patch_size ({self.patch_size})")
        # The prefix budget must fit every condition at full size, otherwise
        # ``_build_prefix`` would silently truncate later conditions (they'd yield
        # zero gain and be rejected after a wasted forward pass). Mirrors the text
        # RAC invariant ``max_ctx == chunk_size * max_cond``.
        assert max_ctx % self.patch_size == 0, (
            f"max_ctx ({max_ctx}) must be a multiple of patch_size "
            f"({self.patch_size})")
        if chunk_size is not None:
            assert max_ctx == chunk_size * self.max_cond, (
                f"max_ctx ({max_ctx}) must equal chunk_size ({chunk_size}) * "
                f"max_cond ({self.max_cond}) = {chunk_size * self.max_cond} so "
                f"every retrieved condition fits the prefix budget without "
                f"truncation. Set --max-ctx accordingly (or lower "
                f"--cascade-max-cond / --chunk-size)."
            )

    def _pbar(self, total: int, desc: str):
        """A per-device tqdm over model-forward sequences, or None if disabled."""
        if not self.show_progress:
            return None
        import tqdm as _tqdm
        return _tqdm.tqdm(
            total=total, desc=f"{desc} [{self.device}]",
            position=getattr(self.device, "index", 0) or 0,
            leave=False, unit="seq", unit_scale=True,
        )

    def compress_batch(
        self,
        segments: List[Tuple[bytes, str]],
        cand_ids_lists: Sequence[Sequence[int]],
    ) -> List[CompressedData]:
        if not segments:
            return []

        pieces = [
            _Piece(idx=i, data=bytes(raw), ext=ext, pool=list(cands))
            for i, ((raw, ext), cands) in enumerate(zip(segments, cand_ids_lists))
        ]

        # Sequences fed to the model: baseline (1/piece) + level-0 candidates
        # (pool size/piece) + encode (1/piece). Cascade grows the total live.
        n_pieces = len(pieces)
        pbar = self._pbar(2 * n_pieces + sum(len(p.pool) for p in pieces),
                          "RAC-bGPT compress")
        try:
            baseline = self._score_jobs([[]] * n_pieces,
                                        [(p.data, p.ext) for p in pieces], pbar=pbar)
            for p, score in zip(pieces, baseline):
                p.baseline_bits = score.bits
                p.bits = score.bits
                p.nll = score.token_nll

            active = [p for p in pieces if p.pool]
            for level in range(self.max_cond):
                if not active:
                    break
                active = self._oracle_step(active, level, self.max_cond, pbar=pbar)

            return self._encode_pieces(pieces, pbar=pbar)
        finally:
            if pbar is not None:
                pbar.close()

    def decompress_batch(
        self,
        compressed_list: List[CompressedData],
    ) -> List[bytes]:
        if not compressed_list:
            return []

        out: List[Optional[bytes]] = [None] * len(compressed_list)
        by_plen: Dict[int, list] = defaultdict(list)
        for i, cd in enumerate(compressed_list):
            prefix = self._build_prefix(cd.metadata.get("ctx_ids", []))
            by_plen[len(prefix)].append((i, cd, prefix))

        pbar = self._pbar(len(compressed_list), "RAC-bGPT decompress")
        try:
            for plen, group in by_plen.items():
                # Rebuild the exact batches _encode_pieces used: same plen group,
                # same (payload token length, piece index) order, same chunking.
                # Unlike the text LM, bGPT's iterative decode is batch-composition
                # sensitive (a float-determinism property), so a piece must be
                # decoded in the same company it was coded in or the coder's logits
                # won't bit-match and the round-trip breaks.
                group.sort(key=lambda t: (t[1].original_length, t[0]))
                for start in range(0, len(group), self.batch_size):
                    batch = group[start:start + self.batch_size]
                    cds = [cd for _, cd, _ in batch]
                    if plen == 0:
                        recs = self.bgpt.decompress_batch(cds)
                    else:
                        recs = self.bgpt.decompress_batch(
                            cds, prefixes=[prefix for _, _, prefix in batch])
                    for (i, _, _), rec in zip(batch, recs):
                        out[i] = rec
                    if pbar is not None:
                        pbar.update(len(batch))
        finally:
            if pbar is not None:
                pbar.close()

        return self._filled(out)

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
        segments: List[Tuple[bytes, str]],
        pbar=None,
    ) -> List[LMScore]:
        out: List[Optional[LMScore]] = [None] * len(prefixes)
        by_plen: Dict[int, List[int]] = defaultdict(list)
        for i, prefix in enumerate(prefixes):
            by_plen[len(prefix)].append(i)

        # by_plen is for batching by prefix length
        for plen, idxs in by_plen.items():
            idxs.sort(key=lambda i: len(segments[i][0]))
            for start in range(0, len(idxs), self.batch_size):
                batch = idxs[start:start + self.batch_size]
                seg_batch = [segments[i] for i in batch]
                if plen == 0:
                    scores = self.bgpt.score_batch(seg_batch)
                else:
                    scores = self.bgpt.score_batch(
                        seg_batch, prefixes=[prefixes[i] for i in batch])
                for i, score in zip(batch, scores):
                    out[i] = score
                if pbar is not None:
                    pbar.update(len(batch))

        return self._filled(out)

    def _oracle_step(
        self,
        active: List[_Piece],
        level: int,
        max_cond: int,
        pbar=None,
    ) -> List[_Piece]:
        prefixes: List[List[int]] = []
        segments: List[Tuple[bytes, str]] = []
        for p in active:
            for cid in p.pool:
                prefixes.append(self._build_prefix(p.choices + [cid]))
                segments.append((p.data, p.ext))

        scores = self._score_jobs(prefixes, segments, pbar=pbar)

        next_active: List[_Piece] = []
        retrieve_pieces: List[_Piece] = []
        retrieve_queries: List[bytes] = []
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
                        if pbar is not None:
                            pbar.total += len(p.pool)
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
                    if pbar is not None:
                        pbar.total += len(p.pool)

        return next_active

    def _cascade_query(
        self,
        data: bytes,
        nll: torch.Tensor,
    ) -> Optional[bytes]:
        tokens = bytes_to_padded_tokens(data, self.patch_size)
        high = (nll > self.cascade_nll_thresh).cpu()
        if float(high.float().mean().item()) < self.cascade_min_frac:
            return None

        hi_bytes = bytes(
            tok for tok, is_high in zip(tokens, high.tolist())
            if is_high and tok != PAD_TOKEN
        )
        return hi_bytes if hi_bytes else b""

    def _encode_pieces(self, pieces: List[_Piece], pbar=None) -> List[CompressedData]:
        out: List[Optional[CompressedData]] = [None] * len(pieces)
        by_plen: Dict[int, List[_Piece]] = defaultdict(list)
        for p in pieces:
            by_plen[len(self._build_prefix(p.choices))].append(p)

        for plen, group in by_plen.items():
            # Sort by (payload token length, idx) so decompress_batch can rebuild
            # the identical batches from CompressedData.original_length — bGPT's
            # decode must see each piece in the same batch it was coded in.
            group.sort(key=lambda p: (len(bytes_to_padded_tokens(p.data, self.patch_size)), p.idx))
            for start in range(0, len(group), self.batch_size):
                batch = group[start:start + self.batch_size]
                seg_batch = [(p.data, p.ext) for p in batch]
                if plen == 0:
                    cds = self.bgpt.compress_batch(seg_batch)
                else:
                    cds = self.bgpt.compress_batch(
                        seg_batch,
                        prefixes=[self._build_prefix(p.choices) for p in batch])

                for p, cd in zip(batch, cds):
                    cd.metadata["ctx_ids"] = list(p.choices)
                    cd.metadata["index_bits"] = self._index_bits(p.choices)
                    cd.metadata["ctx_gain_bits"] = list(p.gains)
                    cd.metadata["ctx_net_gain_bits"] = list(p.net_gains)
                    cd.metadata["ctx_id_bits"] = list(p.id_bits)
                    cd.metadata["baseline_bits"] = p.baseline_bits
                    cd.metadata["data_bits"] = p.bits
                    out[p.idx] = cd
                if pbar is not None:
                    pbar.update(len(batch))

        return self._filled(out)

    @staticmethod
    def _filled(items: List[Optional[T]]) -> List[T]:
        filled: List[T] = []
        for item in items:
            assert item is not None
            filled.append(item)
        return filled
