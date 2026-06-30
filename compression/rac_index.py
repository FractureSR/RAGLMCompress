"""Index side-info coders for oracle RAC.

The decoder cannot re-run retrieval — the retrieval query *is* the unknown data
``x`` — so the chosen base-chunk id(s) must be transmitted as side information.
These coders define how many bits that costs; the cost is used both for the
in-method break-even decision (is a condition worth its id?) and for honest size
accounting.

A ``choice`` is an int base id, or ``None`` ("no condition"). Per piece the
compressor emits a sequence ``[id_1, ..., id_t, None]`` — the trailing ``None`` is
the stop symbol — so the per-piece index cost is
``sum(cost_bits(id_i)) + cost_bits(None)``.

  FixedIndexCoder       flag-style: every id costs ``ceil(log2 n_base)`` bits,
                        ``None`` (stop) costs 1 bit.
  CalibratedIndexCoder  entropy code against a static frequency table calibrated
                        on a held-out split — popular "template" chunks and the
                        common stop become cheap. ``-log2 p(choice)`` is exactly
                        the arithmetic-code length under the static model, so we
                        charge it directly (paper feature 3.3.3).

Following the codebase convention, the chosen ids travel in
``CompressedData.metadata`` and their *bit cost* is added to the compressed size;
we do not emit a separate index bitstream.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from typing import Dict, Optional, Sequence

Choice = Optional[int]   # base id, or None = no condition / stop


class FixedIndexCoder:
    """Fixed-width id + 1-bit stop symbol."""

    kind = "fixed"

    def __init__(self, n_base: int):
        self.n_base = int(n_base)
        self.id_bits = max(1, math.ceil(math.log2(max(self.n_base, 2))))

    def cost_bits(self, choice: Choice) -> float:
        return 1.0 if choice is None else float(self.id_bits)

    def config(self) -> dict:
        return {"kind": self.kind, "n_base": self.n_base}

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.config(), f)

    @classmethod
    def load(cls, path: str) -> "FixedIndexCoder":
        with open(path) as f:
            cfg = json.load(f)
        return cls(cfg["n_base"])


class CalibratedIndexCoder:
    """Static entropy model over ``{None} ∪ {0..n_base-1}`` from calibration choices.

    Add-``alpha`` smoothing keeps every id codeable (an id never seen in
    calibration still costs the floor ``-log2(alpha / denom)``).
    """

    kind = "calibrated"

    def __init__(self, counts: Dict[int, int], none_count: int, total: int,
                 n_base: int, alpha: float = 0.5):
        self.counts = {int(k): int(v) for k, v in counts.items()}
        self.none_count = int(none_count)
        self.total = int(total)
        self.n_base = int(n_base)
        self.alpha = float(alpha)
        # denominator over the stop symbol + n_base ids, each with +alpha pseudocount
        self._denom = self.total + self.alpha * (self.n_base + 1)

    def _p(self, choice: Choice) -> float:
        cnt = self.none_count if choice is None else self.counts.get(int(choice), 0)
        return (cnt + self.alpha) / self._denom

    def cost_bits(self, choice: Choice) -> float:
        return -math.log2(self._p(choice))

    @classmethod
    def calibrate(cls, choice_seqs: Sequence[Sequence[int]], n_base: int,
                  alpha: float = 0.5) -> "CalibratedIndexCoder":
        """Build the table from per-piece chosen-id sequences.

        Each piece contributes its used ids plus one stop (``None``), so the
        stop probability reflects the per-piece "no more conditions" frequency.
        """
        counts: Counter = Counter()
        none_count = 0
        total = 0
        for ids in choice_seqs:
            for cid in ids:
                counts[int(cid)] += 1
                total += 1
            none_count += 1   # one stop per piece
            total += 1
        return cls(dict(counts), none_count, total, n_base, alpha)

    def config(self) -> dict:
        return {"kind": self.kind, "counts": self.counts, "none_count": self.none_count,
                "total": self.total, "n_base": self.n_base, "alpha": self.alpha}

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.config(), f)

    @classmethod
    def load(cls, path: str) -> "CalibratedIndexCoder":
        with open(path) as f:
            cfg = json.load(f)
        return cls({int(k): v for k, v in cfg["counts"].items()},
                   cfg["none_count"], cfg["total"], cfg["n_base"], cfg.get("alpha", 0.5))


def load_index_coder(path: str):
    """Load whichever coder kind was saved at ``path``."""
    with open(path) as f:
        kind = json.load(f).get("kind")
    return {"fixed": FixedIndexCoder, "calibrated": CalibratedIndexCoder}[kind].load(path)
