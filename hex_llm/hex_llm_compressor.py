"""Masked byte compressor over a fixed symbol alphabet (Delétang-faithful).

Two routes, one implementation:
  byte : each byte -> its single vocab token (256-symbol alphabet). Needs a
         tokenizer with all 256 single-byte tokens (Qwen3). 1 token/byte.
  hex  : each byte -> 2 hex chars (16-symbol alphabet). Works with any tokenizer.
         2 tokens/byte.

Each payload symbol is arithmetic-coded against the model's next-token
distribution **restricted and renormalized to the alphabet's token ids**. The
decoder applies the same mask, so it is lossless.

An optional ``prefix`` (raw bytes) is prepended as free model context before the
payload — the coder skips it. This is the RAC condition path: a retrieved base
chunk conditions the payload's prediction without being coded. ``score`` returns
the payload's ideal code length given a prefix, for the RAC oracle.

Self-contained: reuses only the repo's verified range coder. Encode is one
prefill forward; decode mirrors LLMCompressor's padding trick (re-forward the
full-length buffer each step) so decode numerics match encode — O(n^2).
"""
from __future__ import annotations

import math
import os
import sys
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from arithmetic_coder.ac_utils import normalize_pdf
from arithmetic_coder.range_coder import RangeDecoder, RangeEncoder


class AlphabetLLMCompressor:
    """Masked N-symbol arithmetic compressor backed by a causal LM."""

    def __init__(
        self,
        model,
        tokenizer,
        alphabet_ids: Sequence[int],
        to_symbols: Callable[[bytes], List[int]],
        from_symbols: Callable[[Sequence[int]], bytes],
        symbols_per_byte: int,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.alphabet_ids = list(alphabet_ids)
        self.alphabet_t = torch.tensor(self.alphabet_ids, dtype=torch.long, device=self.device)
        self._to_symbols = to_symbols
        self._from_symbols = from_symbols
        self.symbols_per_byte = int(symbols_per_byte)
        bos = tokenizer.bos_token_id
        self.bos = int(bos) if bos is not None else 0

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _payload_logprobs(
        self, payload_symbols: List[int], prefix_symbols: List[int]
    ) -> torch.Tensor:
        """One forward over [bos | prefix | payload]; log-probs for the PAYLOAD.

        Returns ``[L, N]`` log-probabilities over the N alphabet symbols, row i =
        the distribution predicting payload symbol i. The prefix positions are
        context only. Full = [bos] + prefix + payload; logits[j] predicts
        full[j+1], so payload symbol i (full position 1+P+i) is predicted by
        logits[P+i].
        """
        pre_ids = [self.alphabet_ids[s] for s in prefix_symbols]
        pay_ids = [self.alphabet_ids[s] for s in payload_symbols]
        full = torch.tensor([[self.bos] + pre_ids + pay_ids], dtype=torch.long, device=self.device)
        logits = self.model(full, use_cache=False).logits[0, :-1, :].float()  # [P+L, V]
        p = len(prefix_symbols)
        pay_logits = logits[p:p + len(payload_symbols), :]                    # [L, V]
        return pay_logits[:, self.alphabet_t].log_softmax(dim=-1)             # [L, N]

    def score(self, data: bytes, prefix: bytes = b"") -> float:
        """Ideal payload code length in BITS given *prefix* (no coding).

        The RAC oracle ranks conditions by this; it matches what compress emits
        up to a few bytes of range-coder flush.
        """
        payload = self._to_symbols(bytes(data))
        if not payload:
            return 0.0
        logp = self._payload_logprobs(payload, self._to_symbols(bytes(prefix)))
        idx = torch.tensor(payload, dtype=torch.long, device=self.device)
        nll = -logp[torch.arange(len(payload), device=self.device), idx]     # nats
        return float(nll.sum().item()) / math.log(2)

    def compress(self, data: bytes, prefix: bytes = b"") -> Tuple[bytes, int]:
        """Return (compressed_bytes, original_byte_count); *prefix* is free context."""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"data must be bytes, got {type(data).__name__}")
        data = bytes(data)
        payload = self._to_symbols(data)
        if not payload:
            return b"", 0
        probs = self._payload_logprobs(payload, self._to_symbols(bytes(prefix))).exp().cpu().numpy()
        encoder = RangeEncoder()
        for sym, pmf in zip(payload, probs):
            encoder.encode(normalize_pdf(pmf, data_type=np.float32), int(sym))
        return encoder.terminate(), len(data)

    @torch.inference_mode()
    def decompress(self, compressed: bytes, n_bytes: int, prefix: bytes = b"") -> bytes:
        """Inverse of :meth:`compress` under the same *prefix*."""
        n_pay = n_bytes * self.symbols_per_byte
        if n_pay == 0:
            return b""
        pre_ids = [self.alphabet_ids[s] for s in self._to_symbols(bytes(prefix))]
        p = len(pre_ids)
        decoder = RangeDecoder(compressed)
        buf = torch.full((1, 1 + p + n_pay), self.bos, dtype=torch.long, device=self.device)
        for j, tid in enumerate(pre_ids):
            buf[0, 1 + j] = tid
        symbols: List[int] = []
        for i in range(n_pay):
            logits = self.model(buf, use_cache=False).logits[0, :-1, :].float()
            alpha_logits = logits[p + i, self.alphabet_t]                    # [N]
            pmf = alpha_logits.log_softmax(dim=-1).exp().cpu().numpy()
            sym = decoder.decode(normalize_pdf(pmf, data_type=np.float32))
            symbols.append(int(sym))
            buf[0, 1 + p + i] = self.alphabet_t[sym]
        return self._from_symbols(symbols)

    def roundtrip_ok(self, data: bytes, prefix: bytes = b"") -> bool:
        comp, n = self.compress(data, prefix)
        return self.decompress(comp, n, prefix) == bytes(data)


def build_compressor(
    model,
    tokenizer,
    route: str = "byte",
    device: Optional[torch.device] = None,
) -> AlphabetLLMCompressor:
    """Build a byte- or hex-route compressor for *tokenizer*."""
    if route == "byte":
        from hex_llm.byte_codec import byte_token_ids, bytes_to_symbols, symbols_to_bytes
        return AlphabetLLMCompressor(
            model, tokenizer, byte_token_ids(tokenizer),
            bytes_to_symbols, symbols_to_bytes, symbols_per_byte=1, device=device)
    if route == "hex":
        from hex_llm.hex_codec import ordered_hex_ids, bytes_to_symbols, symbols_to_bytes
        return AlphabetLLMCompressor(
            model, tokenizer, ordered_hex_ids(tokenizer),
            bytes_to_symbols, symbols_to_bytes, symbols_per_byte=2, device=device)
    raise ValueError(f"unknown route {route!r}; use 'byte' or 'hex'")
