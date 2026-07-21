"""Shared bGPT byte-token preparation helpers."""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch


PAD_TOKEN = 256
VOCAB_SIZE = 257
RAC_BGPT_DB_FORMAT_VERSION = 1


def extension_tokens(ext: str, patch_size: int) -> List[int]:
    ext = ext.lower().lstrip(".")
    return list(ext.encode("utf-8"))[:patch_size]


def tokens_to_bytes(tokens: Sequence[int]) -> bytes:
    tokens = list(tokens)
    while tokens and tokens[-1] == PAD_TOKEN:
        tokens.pop()
    return bytes(int(x) for x in tokens)


def pad_input_for_bgpt(
    segments: Sequence[Sequence[int]],
    ext_list: Sequence[Sequence[int]],
    device: torch.device,
    patch_size: int,
    pad_to_length: Optional[int] = None,
) -> dict:
    """Build bGPT (patches, masks) tensors for a batch of byte-token segments."""
    prepared: List[List[int]] = []
    valid_lengths: List[int] = []

    for segment, ext in zip(segments, ext_list):
        payload = list(segment)
        if pad_to_length is not None:
            if len(payload) > pad_to_length:
                payload = payload[:pad_to_length]
            else:
                payload = payload + [PAD_TOKEN] * (pad_to_length - len(payload))

        ext_patch = (list(ext)[:patch_size] + [PAD_TOKEN] * patch_size)[:patch_size]
        full = ext_patch + payload + [PAD_TOKEN] * patch_size
        prepared.append(full)
        valid_lengths.append(len(full))

    max_len = max(valid_lengths)
    if max_len % patch_size:
        max_len += patch_size - (max_len % patch_size)

    padded_bytes, patch_masks = [], []
    total_patches = max_len // patch_size

    for full, vlen in zip(prepared, valid_lengths):
        padded_bytes.append(full + [PAD_TOKEN] * (max_len - vlen))
        active = (vlen + patch_size - 1) // patch_size
        patch_masks.append([1] * active + [0] * (total_patches - active))

    return {
        "patches": torch.tensor(padded_bytes, dtype=torch.long, device=device),
        "masks": torch.tensor(patch_masks, dtype=torch.long, device=device),
    }


# ---------------------------------------------------------------------------
# Byte-domain retrieval (the bGPT instantiation of utils/rag_utils.py)
# ---------------------------------------------------------------------------
#
# Mirrors the text instantiation in utils/text_utils.py (bm25_tokenize +
# make_text_retriever): a modality supplies a tokenizer + a retriever factory
# built from rag_utils' generic Scorer/Retriever. Here the "items" are raw byte
# chunks (audio/image), so retrieval is syntactic over byte k-grams — the byte
# analog of lexical text retrieval (RAC claim 2.1.1: re-encoding x favours
# syntactic matches). Used by prepare_rac_data_bgpt.py (build) and
# eval_rac_bgpt.py (load); pass the same signals/kgram to both since the
# tokenizer callable is not persisted.

def bgpt_bytes_tokenize(data: bytes, kgram: int = 4) -> List[str]:
    """Tokenise a byte payload into overlapping k-gram words for BM25.

    The byte-domain counterpart of ``text_utils.bm25_tokenize``: each sliding
    window of ``kgram`` bytes becomes one hashable token (hex-encoded so ``bm25s``
    sees plain strings), so chunks with locally similar byte sequences score high.
    """
    if kgram <= 0:
        raise ValueError(f"kgram must be positive, got {kgram}")
    if not isinstance(data, bytes):
        raise TypeError(f"byte retriever items must be bytes, got {type(data)!r}")
    if len(data) <= kgram:
        return [data.hex()] if data else []
    return [
        data[i:i + kgram].hex()
        for i in range(len(data) - kgram + 1)
    ]


def make_bgpt_retriever(signals: str = "bm25", kgram: int = 4, rrf_k: int = 60):
    """Construct an *unbuilt* ``Retriever`` over byte chunks (its items are bytes).

    ``signals``:
      * ``"bm25"`` — lexical/syntactic over byte k-grams (default; no model
        loaded, the RAC default — cf. ``text_utils.make_text_retriever``).

    A dense byte scorer (e.g. bGPT's own patch encoder as the embedder) is the
    natural ``"dense"``/``"hybrid"`` extension but is not implemented yet. The
    same factory (same ``signals``/``kgram``) must be used to build and to load.
    """
    if kgram <= 0:
        raise ValueError(f"kgram must be positive, got {kgram}")
    from functools import partial
    from utils.rag_utils import Retriever, BM25Scorer
    if signals != "bm25":
        raise ValueError(
            f"bgpt retriever supports signals='bm25' only (got {signals!r}); "
            f"a dense byte/patch-embedding scorer is a future extension")
    scorers = [BM25Scorer(tokenize=partial(bgpt_bytes_tokenize, kgram=kgram))]
    return Retriever(scorers, rrf_k=rrf_k)
