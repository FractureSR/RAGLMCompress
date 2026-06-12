"""BGPTCompressor: byte-level compression using the bGPT hierarchical model.

Structurally mirrors LLMCompressor:  both use _encode_sequence and _decode_batch
from BaseCompressor.  The only differences are:
  - Input domain: raw bytes → byte tokens 0-255, with PAD=256.
  - Model input: (patches, masks) tensors built by pad_input_for_bgpt.
  - Logits shape: the bGPT output needs a reshape before use.

The padding trick is preserved identically: the decode loop pads the partial
decoded token list to the original payload length before each forward pass,
mirroring the prefill pass.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from compression.base_compressor import BaseCompressor
from compression.types import CompressedData
from utils.bgpt_codec_utils import (
    PAD_TOKEN, VOCAB_SIZE,
    bytes_to_padded_tokens, extension_tokens,
    pad_input_for_bgpt, tokens_to_bytes,
)


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------

class BGPTCompressor(BaseCompressor):

    def __init__(self, model, patch_size: int = 16, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.patch_size = patch_size
        self.device = device or next(model.parameters()).device
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress_batch(
        self, segments: List[Tuple[bytes, str]]
    ) -> List[CompressedData]:
        """Compress multiple segments in a single forward pass.

        Segments with different byte lengths (e.g. the last audio chunk) are
        padded to the longest payload in the batch with PAD_TOKEN.  Each
        CompressedData stores the actual original_length so the decoder knows
        how many tokens to recover.
        """
        if not segments:
            return []

        ext_ids_list  = [extension_tokens(ext, self.patch_size) for _, ext in segments]
        payload_list  = [bytes_to_padded_tokens(raw, self.patch_size) for raw, _ in segments]
        orig_lengths  = [len(p) for p in payload_list]
        max_orig_len  = max(orig_lengths)

        # Pad shorter payloads to the longest in the batch.
        payload_list = [
            p + [PAD_TOKEN] * (max_orig_len - len(p))
            for p in payload_list
        ]

        B      = len(segments)
        padded = pad_input_for_bgpt(
            payload_list, ext_ids_list,
            device=self.device, patch_size=self.patch_size,
        )

        with torch.inference_mode():
            out        = self.model(patches=padded["patches"], masks=padded["masks"])
            logits_raw = out.logits   # (B * pairs_per_sample, patch_size+1, VOCAB)

        pairs_per_sample  = logits_raw.shape[0] // B
        logits_4d         = logits_raw.reshape(B, pairs_per_sample, self.patch_size + 1, VOCAB_SIZE)
        # (B, max_orig_len, VOCAB) — slice per sample below to its actual length
        logits_per_sample = logits_4d[:, :-1, :-1, :].reshape(B, -1, VOCAB_SIZE)

        results = []
        for b, (_, ext) in enumerate(segments):
            orig_len  = orig_lengths[b]
            # Extract only the actual (non-padding) payload tokens and logits.
            payload_t = padded["patches"][b:b+1, self.patch_size : self.patch_size + orig_len]
            prefix    = torch.full((1, 1), PAD_TOKEN, dtype=torch.long, device=self.device)
            ac_input  = torch.cat([prefix, payload_t], dim=1)
            logits_b  = logits_per_sample[b:b+1, :orig_len, :]   # (1, orig_len, VOCAB)

            cd = self._encode_sequence(ac_input, logits_b, prefix_length=1)
            cd.metadata["ext"] = ext.lower().lstrip(".")
            results.append(cd)

        return results

    def decompress_batch(
        self,
        compressed_list: List[CompressedData],
        max_tokens: Optional[int] = None,
        show_progress: bool = False,
    ) -> List[bytes]:
        """Decompress N samples with one forward pass per token step.

        Samples with different original_length are handled by padding all
        decoded buffers to the longest sequence; _decode_batch freezes each
        sequence once its own original_length is reached.
        """
        if not compressed_list:
            return []

        B            = len(compressed_list)
        max_orig_len = max(cd.original_length for cd in compressed_list)
        ext_ids_list = [
            extension_tokens(cd.metadata.get("ext", "bin"), self.patch_size)
            for cd in compressed_list
        ]

        def get_logits_fn(buf: torch.Tensor) -> torch.Tensor:
            # buf: [B, 1 + max_orig_len]
            decoded_so_far = [buf[b, 1:].tolist() for b in range(B)]
            padded = pad_input_for_bgpt(
                decoded_so_far, ext_ids_list,
                device=self.device, patch_size=self.patch_size,
                pad_to_length=max_orig_len,
            )
            with torch.inference_mode():
                out        = self.model(patches=padded["patches"], masks=padded["masks"])
                logits_raw = out.logits   # (B * pairs_per_sample, patch_size+1, VOCAB)
            pairs_per_sample = logits_raw.shape[0] // B
            logits_4d        = logits_raw.reshape(B, pairs_per_sample, self.patch_size + 1, VOCAB_SIZE)
            return logits_4d[:, :-1, :-1, :].reshape(B, -1, VOCAB_SIZE)

        prefix = torch.full((B, 1), PAD_TOKEN, dtype=torch.long, device=self.device)
        decoded_lists = self._decode_batch(
            compressed_list, get_logits_fn, prefix, self.device,
            pad_fill=PAD_TOKEN, max_tokens=max_tokens,
            show_progress=show_progress,
        )
        return [tokens_to_bytes(d) for d in decoded_lists]
