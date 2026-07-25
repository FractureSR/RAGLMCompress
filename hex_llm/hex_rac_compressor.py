"""Oracle RAC over the byte/hex LLM compressor.

The LLM analog of RACBGPTCompressor, but over AlphabetLLMCompressor's prefix
path. For each payload window:

  1. retrieve the top-m most byte-similar base chunks (query = the payload bytes);
  2. score the payload under the model with each candidate prepended as a free
     prefix (AlphabetLLMCompressor.score);
  3. keep the single best candidate only if its data-bit gain beats the cost of
     transmitting its id (FixedIndexCoder), else no condition;
  4. code the payload with the chosen prefix; charge the index bits.

The decoder can't re-run retrieval (the query is the unknown data), so the chosen
base-chunk id is side information; its bit cost is added to the compressed size.
No training. Single condition per window (no cascade) — the minimal RAC.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from compression.rac_index import FixedIndexCoder
from hex_llm.hex_llm_compressor import AlphabetLLMCompressor


@dataclass
class RACResult:
    """One window's RAC outcome."""

    compressed: bytes            # arithmetic-coded payload bytes
    original_bytes: int
    index_bits: float            # transmitted id(s) + stop
    cond_id: Optional[int]       # chosen base chunk, or None
    base_bits: float             # ideal no-condition payload code length
    data_bits: float             # ideal payload code length with the chosen prefix
    gain_bits: float             # base_bits - data_bits (0 if no condition)
    id_bits: float               # index_coder.cost_bits(cond_id) (0 if none)

    @property
    def total_bits(self) -> float:
        return len(self.compressed) * 8 + self.index_bits


class HexRACCompressor:
    """Retrieve -> oracle-select one byte prefix -> code the payload."""

    def __init__(
        self,
        base: AlphabetLLMCompressor,
        base_chunks: Sequence[bytes],
        retriever,
        index_coder=None,
        max_ctx: int = 512,
        m: int = 8,
        margin_bits: float = 0.0,
    ) -> None:
        self.base = base
        self.base_chunks = list(base_chunks)
        self.retriever = retriever
        self.index_coder = index_coder or FixedIndexCoder(len(self.base_chunks))
        self.max_ctx = int(max_ctx)
        self.m = int(m)
        self.margin_bits = float(margin_bits)

    def compress(self, data: bytes) -> RACResult:
        data = bytes(data)
        base_bits = self.base.score(data)                     # no-condition ideal length

        hits = self.retriever.retrieve(data, top_k=self.m) if self.base_chunks else []
        best_id: Optional[int] = None
        best_bits = base_bits
        best_gain = 0.0
        best_id_bits = 0.0
        for cid, _score in hits:
            cid = int(cid)
            prefix = self.base_chunks[cid][: self.max_ctx]
            cond_bits = self.base.score(data, prefix=prefix)
            gain = base_bits - cond_bits
            id_bits = self.index_coder.cost_bits(cid)
            # Accept the best candidate that also clears its own id cost.
            if gain > id_bits + self.margin_bits and cond_bits < best_bits:
                best_id, best_bits, best_gain, best_id_bits = cid, cond_bits, gain, id_bits

        stop_bits = self.index_coder.cost_bits(None)
        if best_id is None:
            comp, n = self.base.compress(data)
            return RACResult(comp, n, stop_bits, None, base_bits, base_bits, 0.0, 0.0)
        prefix = self.base_chunks[best_id][: self.max_ctx]
        comp, n = self.base.compress(data, prefix=prefix)
        return RACResult(
            comp, n, best_id_bits + stop_bits, best_id,
            base_bits, best_bits, best_gain, best_id_bits)


def chunk_bytes(data: bytes, size: int, keep_partial: bool = False) -> List[bytes]:
    """Split *data* into ``size``-byte chunks (drop the short tail by default)."""
    chunks = [data[i:i + size] for i in range(0, len(data), size)]
    if not keep_partial and chunks and len(chunks[-1]) < size:
        chunks.pop()
    return chunks
