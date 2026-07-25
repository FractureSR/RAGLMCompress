"""Lossless bytes <-> byte-token-id codec: each byte -> its single vocab token.

The "byte route" (vs the hex route in hex_codec.py). Requires a byte-level BPE
tokenizer whose vocabulary contains all 256 single-byte tokens — true for Qwen3
(verified 256/256), not for SmolLM2. Advantages over hex:

  * 1 token/byte (hex is 2) -> half the sequence, ~2x faster;
  * with Qwen3's 40960 context a window holds ~40k bytes, so a whole EuroSAT
    image (12288 B) or audio chunk fits in ONE window (no cold-start splitting).

Symbols are the byte values 0..255 directly, so bytes<->symbols is trivial and
tokenizer-independent; only the 256 token ids come from the tokenizer.
"""
from __future__ import annotations

from typing import List


def byte_token_ids(tokenizer) -> List[int]:
    """The 256 token ids in byte order (index == byte value 0..255).

    Uses the GPT-2 byte<->unicode table that every byte-level BPE tokenizer
    (GPT-2, Llama, Qwen) is built on. Fails loudly if any byte is not a single
    token, i.e. the vocab lacks a complete 256-byte alphabet — use the hex route
    with that tokenizer instead.
    """
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

    byte_to_char = bytes_to_unicode()
    ids: List[int] = []
    for value in range(256):
        enc = tokenizer(byte_to_char[value], add_special_tokens=False)["input_ids"]
        if len(enc) != 1:
            raise ValueError(
                f"byte {value} maps to {len(enc)} tokens ({enc}); this tokenizer "
                "lacks a complete 256-byte vocab — use the hex route instead")
        ids.append(int(enc[0]))
    if len(set(ids)) != 256:
        raise ValueError("byte tokens are not 256 distinct ids")
    return ids


def bytes_to_symbols(data: bytes) -> List[int]:
    """Raw bytes -> symbols in 0..255 (1 per byte)."""
    return list(data)


def symbols_to_bytes(symbols) -> bytes:
    """Inverse of :func:`bytes_to_symbols`."""
    return bytes(int(sym) for sym in symbols)


def _selftest() -> None:
    import os
    import random

    for _ in range(1000):
        data = bytes(random.randrange(256) for _ in range(random.randint(0, 64)))
        assert symbols_to_bytes(bytes_to_symbols(data)) == data
    assert symbols_to_bytes(bytes_to_symbols(bytes(range(256)))) == bytes(range(256))
    print("byte_codec: byte<->symbol round-trip OK (1000 random + full 0..255)")

    model = os.environ.get("BYTE_TEST_MODEL")
    if model:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
        ids = byte_token_ids(tok)
        assert len(ids) == 256
        for value, tid in enumerate(ids):
            assert tok.decode([tid]).encode("latin-1", "ignore")[:1] in (bytes([value]), b"")
        print(f"byte_codec: {model} -> 256/256 byte token ids OK")


if __name__ == "__main__":
    _selftest()
