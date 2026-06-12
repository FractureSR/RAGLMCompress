"""Range coder backed by constriction.stream.queue.

Provides RangeEncoder / RangeDecoder with the same per-symbol interface as the
original DeepMind arithmetic_coder so callers need no changes:

    encoder = RangeEncoder()
    for sym, pdf in zip(symbols, pdfs):
        encoder.encode(pdf, sym)
    compressed: bytes = encoder.terminate()

    decoder = RangeDecoder(compressed)
    for pdf in pdfs:
        sym: int = decoder.decode(pdf)

PDFs are accepted as float32/float64 numpy arrays over the full alphabet.
They are normalised internally before passing to constriction.
"""
from __future__ import annotations

from typing import List

import numpy as np
import constriction

from arithmetic_coder.ac_utils import normalize_pdf


def _to_categorical(pdf: np.ndarray) -> constriction.stream.model.Categorical:
    return constriction.stream.model.Categorical(
        normalize_pdf(pdf, data_type=np.float32),
        perfect=False,
    )


class RangeEncoder:
    def __init__(self) -> None:
        self._encoder = constriction.stream.queue.RangeEncoder()

    def encode(self, pdf: np.ndarray, symbol: int) -> None:
        model = _to_categorical(pdf)
        self._encoder.encode(np.array([symbol], dtype=np.int32), model)

    def terminate(self) -> bytes:
        """Flush and return compressed data as bytes."""
        words = self._encoder.get_compressed()   # uint32 array
        return words.astype("<u4").tobytes()


class RangeDecoder:
    def __init__(self, compressed: bytes) -> None:
        words = np.frombuffer(compressed, dtype="<u4").copy()
        self._decoder = constriction.stream.queue.RangeDecoder(words)

    def decode(self, pdf: np.ndarray) -> int:
        model = _to_categorical(pdf)
        result = self._decoder.decode(model, 1)
        return int(result[0])
