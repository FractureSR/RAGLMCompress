"""Shared bGPT byte-token preparation helpers."""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch


PAD_TOKEN = 256
VOCAB_SIZE = 257


def extension_tokens(ext: str, patch_size: int) -> List[int]:
    ext = ext.lower().lstrip(".")
    return list(ext.encode("utf-8"))[:patch_size]


def bytes_to_padded_tokens(raw_bytes: bytes, patch_size: int) -> List[int]:
    tokens = list(raw_bytes)
    remainder = len(tokens) % patch_size
    if remainder:
        tokens.extend([PAD_TOKEN] * (patch_size - remainder))
    return tokens


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
