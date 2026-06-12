"""BaseCompressor: shared encode/decode kernels used by LLM and bGPT compressors.

Both compressors share:
  - _encode_sequence : arithmetic-code a token sequence given pre-computed logits
  - _decode_batch    : batched iterative decode with the padding trick

They differ only in how they prepare inputs and call the model, which is
implemented in the subclass via a get_logits_fn callback passed to _decode_batch.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import torch

from arithmetic_coder.ac_utils import normalize_pdf
from arithmetic_coder.range_coder import RangeEncoder, RangeDecoder
from compression.types import CompressedData


def _normalize(pdf: np.ndarray) -> np.ndarray:
    """Backward-compatible wrapper used by notebooks and diagnostics."""
    return normalize_pdf(pdf, data_type=np.float32)


class BaseCompressor:

    # ------------------------------------------------------------------
    # Shared encode kernel
    # ------------------------------------------------------------------

    def _encode_sequence(
        self,
        input_ids: torch.Tensor,     # [1, seq_len]  (includes dummy prefix token)
        logits: torch.Tensor,        # [1, seq_len-1, vocab]  — logits[i] predicts input_ids[i+1]
        prefix_length: int,
    ) -> CompressedData:
        """Arithmetic-code input_ids[prefix_length:] using the given logits."""
        target = input_ids[0, prefix_length:].cpu().numpy()           # (data_len,)
        prob_matrix = (
            logits[0, prefix_length - 1:, :]                          # (data_len, vocab)
            .softmax(dim=-1).float().cpu().numpy()
        )

        encoder = RangeEncoder()
        for sym, pdf in zip(target, prob_matrix):
            encoder.encode(_normalize(pdf), int(sym))
        compressed = encoder.terminate()

        return CompressedData(
            compressed_bytes=compressed,
            original_length=len(target),
        )

    # ------------------------------------------------------------------
    # Shared batch decode loop
    # ------------------------------------------------------------------

    def _decode_batch(
        self,
        compressed_list: List[CompressedData],
        get_logits_fn: Callable[[torch.Tensor], torch.Tensor],
        start_tokens: torch.Tensor,   # [B, prefix_length]
        device: torch.device,
        pad_fill: int = 0,
        max_tokens: Optional[int] = None,
        show_progress: bool = False,
    ) -> List[List[int]]:
        """Batch decode: one forward pass per token step across B sequences.

        At each step i a single call to get_logits_fn covers all B sequences.
        Sequences shorter than max_len are frozen once their length is reached.

        get_logits_fn(buf) receives [B, prefix_length + max_orig_len] and returns
        logits [B, prefix_length + max_orig_len - 1, vocab]. The logit at position
        prefix_length + i - 1 predicts the symbol at decode step i.
        """
        B = len(compressed_list)
        orig_lengths = [cd.original_length for cd in compressed_list]
        decode_lengths = [
            (min(L, max_tokens) if max_tokens is not None else L)
            for L in orig_lengths
        ]
        max_orig_len = max(orig_lengths)
        n_decode = max(decode_lengths)
        prefix_length = start_tokens.shape[1]

        decoders = [RangeDecoder(cd.compressed_bytes) for cd in compressed_list]

        buf = torch.full(
            (B, prefix_length + max_orig_len), pad_fill,
            dtype=torch.long, device=device,
        )
        buf[:, :prefix_length] = start_tokens

        decoded: List[List[int]] = [[] for _ in range(B)]

        steps = range(n_decode)
        if show_progress:
            import tqdm as _tqdm
            steps = _tqdm.tqdm(steps, desc=f"Decode B={B}", total=n_decode,
                               leave=False, unit="tok", unit_scale=B)

        for i in steps:
            logits = get_logits_fn(buf)   # [B, prefix_length + max_orig_len - 1, vocab]
            for b in range(B):
                if i < decode_lengths[b]:
                    probs = (
                        logits[b, prefix_length + i - 1, :]
                        .softmax(dim=-1).float().cpu().numpy()
                    )
                    token = decoders[b].decode(_normalize(probs))
                    decoded[b].append(token)
                    buf[b, prefix_length + i] = token

        return decoded
