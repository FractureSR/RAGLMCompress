"""Lossless bytes <-> hex-token-id codec for feeding arbitrary bytes to a text LLM.

Delétang-style byte compression needs a lossless, deterministic map from raw
bytes to a fixed small symbol alphabet the LLM can predict. We use hex:

    each byte -> 2 hex chars (0-9a-f)

and map every hex char to its **single** vocabulary token id *char-by-char*
(bypassing BPE merges), so the whole stream lives in a fixed 16-symbol alphabet.
That is what lets the compressor mask the model distribution to those 16 tokens.

Cost: 2 tokens/byte -> the token sequence is 2x the byte length (context/compute
cost, halves the effective byte window). It does NOT inflate the compressed size:
arithmetic coding charges the model's -log p, and "the next token is one of 16
hex chars" is structure the model predicts away.

This module has no torch dependency; it only needs a HF tokenizer.
"""
from __future__ import annotations

from typing import Dict, List

HEX_CHARS = "0123456789abcdef"


def hex_char_token_ids(tokenizer) -> Dict[str, int]:
    """Map each of 0-9a-f to its single vocab token id; fail if any is not 1 token.

    A byte-level BPE tokenizer represents every hex char as one token; we verify
    that here so a tokenizer that does not is rejected loudly rather than silently
    producing a non-hex-alphabet stream.
    """
    ids: Dict[str, int] = {}
    for char in HEX_CHARS:
        enc = tokenizer(char, add_special_tokens=False)["input_ids"]
        if len(enc) != 1:
            raise ValueError(
                f"hex char {char!r} does not map to a single token (got {enc}); "
                "this tokenizer cannot be used for the char-level hex scheme")
        ids[char] = int(enc[0])
    if len(set(ids.values())) != len(HEX_CHARS):
        raise ValueError(f"hex chars collide onto <16 distinct token ids: {ids}")
    return ids


def ordered_hex_ids(tokenizer) -> List[int]:
    """The 16 token ids in HEX_CHARS order, so list index == hex symbol value."""
    char_ids = hex_char_token_ids(tokenizer)
    return [char_ids[char] for char in HEX_CHARS]


def bytes_to_symbols(data: bytes) -> List[int]:
    """Raw bytes -> a list of hex symbols in 0..15 (2 per byte)."""
    return [HEX_CHARS.index(char) for char in data.hex()]


def symbols_to_bytes(symbols) -> bytes:
    """Inverse of :func:`bytes_to_symbols`."""
    return bytes.fromhex("".join(HEX_CHARS[int(sym)] for sym in symbols))


def _selftest() -> None:
    # Codec is tokenizer-independent for the symbol<->byte half; check that.
    import os
    import random

    for _ in range(1000):
        n = random.randint(0, 64)
        data = bytes(random.randrange(256) for _ in range(n))
        syms = bytes_to_symbols(data)
        assert len(syms) == 2 * n, (len(syms), n)
        assert all(0 <= s < 16 for s in syms)
        assert symbols_to_bytes(syms) == data
    # full 0..255 coverage
    every = bytes(range(256))
    assert symbols_to_bytes(bytes_to_symbols(every)) == every
    print("hex_codec: byte<->symbol round-trip OK (1000 random + full 0..255)")

    model = os.environ.get("HEX_TEST_MODEL")
    if model:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, use_fast=False)
        ids = ordered_hex_ids(tok)
        print(f"hex_codec: {model} -> 16 hex token ids {ids}")
        # decode each id back to its char
        for sym, tid in enumerate(ids):
            assert tok.decode([tid]) == HEX_CHARS[sym], (sym, tid, tok.decode([tid]))
        print("hex_codec: token-id -> hex char decode OK")


if __name__ == "__main__":
    _selftest()
