"""Byte-level arithmetic compression with bGPT.

For the audio and image checkpoints, bGPT was trained on complete WAV/BMP byte
streams.  The public pipeline stores and measures only media payload bytes, so
this compressor restores the familiar container header as *free context*:

    extension patch + header + optional RAC condition + payload

The header and condition are fed to bGPT but are never arithmetic-coded.  Only
the original payload contributes to ``CompressedData.original_length`` and the
reported compression ratio.  The combined stream is padded once, at its end,
to satisfy bGPT's byte-patch layout.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch

from compression.base_compressor import BaseCompressor
from compression.types import CompressedData, LMScore
from utils.bgpt_codec_utils import (
    PAD_TOKEN,
    VOCAB_SIZE,
    extension_tokens,
    pad_input_for_bgpt,
    tokens_to_bytes,
)


IMAGE_BODY_BYTES = 32 * 32 * 3
AUDIO_BODY_BYTES = 8000


# Canonical headers for the bGPT training-sized files. They are model-only
# context, not part of the stored or measured byte stream. In particular, the
# WAV header remains the checkpoint's 8 kHz template even though input clips
# retain their native sample rate; only their header-free PCM payload is coded.
_BMP_32X32_RGB24_HEADER = bytes.fromhex(
    "424d360c00000000000036000000280000002000000020000000"
    "0100180000000000000c0000c40e0000c40e00000000000000000000"
)
_WAV_8K_MONO_U8_8000_HEADER = bytes.fromhex(
    "52494646641f000057415645666d74201000000001000100401f0000"
    "401f00000100080064617461401f0000"
)
_FREE_CONTEXT_HEADERS = {
    "bmp": _BMP_32X32_RGB24_HEADER,
    "wav": _WAV_8K_MONO_U8_8000_HEADER,
}


def free_context_header(ext: str) -> bytes:
    """Return fixed decoder-known model context for a bGPT modality."""
    if not isinstance(ext, str):
        raise TypeError(f"bGPT media extension must be str, got {type(ext)!r}")
    key = ext.lower().lstrip(".")
    try:
        return _FREE_CONTEXT_HEADERS[key]
    except KeyError as exc:
        raise ValueError(
            f"unsupported bGPT media extension {ext!r}; expected 'bmp' or 'wav'"
        ) from exc


class BGPTCompressor(BaseCompressor):

    def __init__(self, model, patch_size: int = 16,
                 device: Optional[torch.device] = None) -> None:
        self.model = model
        self.patch_size = patch_size
        self.device = device or next(model.parameters()).device
        self.model.eval()

    def _normalise_prefixes(
        self,
        prefixes: Optional[Sequence[Sequence[int]]],
        batch_size: int,
    ) -> List[List[int]]:
        if prefixes is None:
            return [[] for _ in range(batch_size)]
        if len(prefixes) != batch_size:
            raise ValueError(
                f"got {len(prefixes)} prefixes for {batch_size} segments")

        result: List[List[int]] = []
        for sample_idx, prefix in enumerate(prefixes):
            values = [int(token) for token in prefix]
            invalid = next((token for token in values if not 0 <= token <= 255), None)
            if invalid is not None:
                raise ValueError(
                    f"prefix {sample_idx} contains non-byte token {invalid}; "
                    "RAC conditions must contain raw bytes, not PAD_TOKEN")
            result.append(values)
        return result

    def _check_context_length(self, content_length: int) -> None:
        total_patches = content_length // self.patch_size + 2
        patch_decoder = getattr(self.model, "patch_level_decoder", None)
        config = getattr(patch_decoder, "config", None)
        limit = getattr(config, "max_position_embeddings", None)
        if limit is not None and total_patches > limit:
            raise ValueError(
                f"bGPT input needs {total_patches} patches (including extension "
                f"and end patches), but the model supports {limit}")

    def _prepare_input(
        self,
        payloads: Sequence[Sequence[int]],
        exts: Sequence[str],
        prefixes: Optional[Sequence[Sequence[int]]] = None,
    ) -> Tuple[dict, List[int]]:
        """Build one uniformly padded batch and return payload byte offsets."""
        batch_size = len(payloads)
        prefix_list = self._normalise_prefixes(prefixes, batch_size)
        if len(exts) != batch_size:
            raise ValueError(f"got {len(exts)} extensions for {batch_size} payloads")

        contents: List[List[int]] = []
        payload_offsets: List[int] = []
        for sample_idx, (payload, ext, prefix) in enumerate(
                zip(payloads, exts, prefix_list)):
            payload_tokens = [int(token) for token in payload]
            invalid = next(
                (token for token in payload_tokens if not 0 <= token <= PAD_TOKEN),
                None,
            )
            if invalid is not None:
                raise ValueError(
                    f"payload {sample_idx} contains invalid bGPT token {invalid}")

            header = list(free_context_header(ext))
            payload_offsets.append(len(header) + len(prefix))
            content = header + prefix + payload_tokens
            remainder = len(content) % self.patch_size
            if remainder:
                content.extend([PAD_TOKEN] * (self.patch_size - remainder))
            contents.append(content)

        max_content_length = max(len(content) for content in contents)
        contents = [
            content + [PAD_TOKEN] * (max_content_length - len(content))
            for content in contents
        ]
        self._check_context_length(max_content_length)

        ext_ids = [extension_tokens(ext, self.patch_size) for ext in exts]
        padded = pad_input_for_bgpt(
            contents,
            ext_ids,
            device=self.device,
            patch_size=self.patch_size,
        )
        return padded, payload_offsets

    def _model_logits(self, padded: dict, batch_size: int) -> torch.Tensor:
        """Return one next-byte logit row per prepared content token."""
        with torch.inference_mode():
            output = self.model(
                patches=padded["patches"], masks=padded["masks"])
            logits_raw = output.logits

        if logits_raw.shape[0] % batch_size:
            raise RuntimeError(
                "bGPT returned a non-uniform number of patch pairs per sample")
        pairs_per_sample = logits_raw.shape[0] // batch_size
        logits_4d = logits_raw.reshape(
            batch_size,
            pairs_per_sample,
            self.patch_size + 1,
            VOCAB_SIZE,
        )
        # The final pair predicts the explicit end patch. It is model framing,
        # not part of header/condition/payload content.
        return logits_4d[:, :-1, :-1, :].reshape(
            batch_size, -1, VOCAB_SIZE)

    def _prefill(
        self,
        segments: List[Tuple[bytes, str]],
        prefixes: Optional[Sequence[Sequence[int]]] = None,
    ) -> Tuple[dict, torch.Tensor, List[int], List[int]]:
        """Run one teacher-forced batch over header + condition + payload.

        Returns ``(model_input, logits, raw_payload_lengths, payload_offsets)``.
        Offsets are relative to the prepared content, after the extension patch.
        """
        for sample_idx, (raw, _) in enumerate(segments):
            if not isinstance(raw, bytes):
                raise TypeError(
                    f"payload {sample_idx} must be bytes, got {type(raw)!r}")
        payloads = [list(raw) for raw, _ in segments]
        exts = [ext for _, ext in segments]
        padded, payload_offsets = self._prepare_input(payloads, exts, prefixes)
        logits = self._model_logits(padded, len(segments))
        return padded, logits, [len(raw) for raw, _ in segments], payload_offsets

    def compress_batch(
        self,
        segments: List[Tuple[bytes, str]],
        prefixes: Optional[Sequence[Sequence[int]]] = None,
    ) -> List[CompressedData]:
        """Compress only payload bytes, with header/prefix supplied for free."""
        if not segments:
            return []

        padded, logits, lengths, offsets = self._prefill(segments, prefixes)
        results: List[CompressedData] = []
        for sample_idx, ((_, ext), payload_length, offset) in enumerate(
                zip(segments, lengths, offsets)):
            start = self.patch_size + offset
            target = padded["patches"][
                sample_idx:sample_idx + 1, start:start + payload_length]
            dummy = torch.full(
                (1, 1), PAD_TOKEN, dtype=torch.long, device=self.device)
            coder_input = torch.cat([dummy, target], dim=1)
            payload_logits = logits[
                sample_idx:sample_idx + 1, offset:offset + payload_length, :]

            compressed = self._encode_sequence(
                coder_input, payload_logits, prefix_length=1)
            compressed.metadata.update({
                "ext": ext.lower().lstrip("."),
                "free_header_bytes": len(free_context_header(ext)),
            })
            results.append(compressed)
        return results

    def score_batch(
        self,
        segments: List[Tuple[bytes, str]],
        prefixes: Optional[Sequence[Sequence[int]]] = None,
    ) -> List[LMScore]:
        """Return payload code length and byte NLL without range coding."""
        if not segments:
            return []

        padded, logits, lengths, offsets = self._prefill(segments, prefixes)
        results: List[LMScore] = []
        for sample_idx, (payload_length, offset) in enumerate(zip(lengths, offsets)):
            start = self.patch_size + offset
            target = padded["patches"][
                sample_idx, start:start + payload_length]
            payload_logits = logits[
                sample_idx, offset:offset + payload_length, :]
            nll = -payload_logits.log_softmax(dim=-1).gather(
                1, target.unsqueeze(1)).squeeze(1)
            results.append(LMScore(
                bits=float(nll.sum().item()) / math.log(2),
                token_nll=nll,
            ))
        return results

    def decompress_batch(
        self,
        compressed_list: List[CompressedData],
        prefixes: Optional[Sequence[Sequence[int]]] = None,
        max_tokens: Optional[int] = None,
        show_progress: bool = False,
    ) -> List[bytes]:
        """Decode payloads under the same free header and RAC conditions."""
        if not compressed_list:
            return []

        batch_size = len(compressed_list)
        lengths = [compressed.original_length for compressed in compressed_list]
        max_payload_length = max(lengths)
        exts: List[str] = []
        for sample_idx, compressed in enumerate(compressed_list):
            if "ext" not in compressed.metadata:
                raise ValueError(
                    f"compressed sample {sample_idx} has no media extension metadata")
            ext = compressed.metadata["ext"]
            if not isinstance(ext, str):
                raise ValueError(
                    f"compressed sample {sample_idx} has invalid media extension "
                    f"metadata {ext!r}")
            exts.append(ext)

        def get_logits_fn(buffer: torch.Tensor) -> torch.Tensor:
            payloads = [
                buffer[sample_idx, 1:1 + payload_length].tolist()
                for sample_idx, payload_length in enumerate(lengths)
            ]
            padded, offsets = self._prepare_input(payloads, exts, prefixes)
            logits = self._model_logits(padded, batch_size)

            # BaseCompressor expects payload-relative rows shared in one dense
            # tensor, while free-header/prefix offsets may differ by sample.
            aligned = logits.new_zeros(
                (batch_size, max_payload_length, VOCAB_SIZE))
            for sample_idx, (payload_length, offset) in enumerate(
                    zip(lengths, offsets)):
                aligned[sample_idx, :payload_length] = logits[
                    sample_idx, offset:offset + payload_length]
            return aligned

        dummy = torch.full(
            (batch_size, 1), PAD_TOKEN, dtype=torch.long, device=self.device)
        decoded = self._decode_batch(
            compressed_list,
            get_logits_fn,
            dummy,
            self.device,
            pad_fill=PAD_TOKEN,
            max_tokens=max_tokens,
            show_progress=show_progress,
        )
        return [tokens_to_bytes(tokens) for tokens in decoded]
