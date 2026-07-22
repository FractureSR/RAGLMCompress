"""Whole-target delta-compression baseline adapters.

The patch bytes returned here intentionally do *not* include a reference ID.
Evaluators must add the shared-reference index cost using the same index coder
as RAC, then compare that total against the no-reference fallback.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from .base import (
    DecodeResult,
    DeltaCodec,
    EncodeResult,
    ExternalDeltaCodec,
    Metadata,
    _as_bytes,
    _metadata,
    discover_binary,
    read_file,
    run_command,
)


class ZstdPatchCodec(ExternalDeltaCodec):
    """zstd ``--patch-from`` at the same ``--ultra -22 -T1`` setting."""

    name = "zstd-patch"
    binary_candidates = ("zstd",)

    def __init__(
        self,
        *,
        binary: Optional[str | os.PathLike[str]] = None,
        timeout_s: float = 300.0,
        temp_dir: Optional[str | os.PathLike[str]] = None,
    ) -> None:
        super().__init__(binary=binary, timeout_s=timeout_s)
        self.temp_dir = os.fspath(temp_dir) if temp_dir is not None else None

    def encode(
        self,
        target: bytes,
        reference: bytes,
        metadata: Optional[Metadata] = None,
    ) -> EncodeResult:
        del metadata
        target_data = _as_bytes(target, label="delta target")
        reference_data = _as_bytes(reference, label="delta reference")
        with tempfile.TemporaryDirectory(
            prefix="raglm-zstd-patch-", dir=self.temp_dir
        ) as work_dir:
            target_path = Path(work_dir, "target.bin")
            reference_path = Path(work_dir, "reference.bin")
            artifact_path = Path(work_dir, "patch.zst")
            target_path.write_bytes(target_data)
            reference_path.write_bytes(reference_data)
            result = self._run(
                (
                    "--ultra",
                    "-22",
                    "-T1",
                    "-q",
                    f"--patch-from={reference_path}",
                    target_path,
                    "-o",
                    artifact_path,
                ),
                operation="zstd patch-from encode",
            )
            artifact_data = read_file(artifact_path, operation="zstd patch encode")
        return EncodeResult(artifact_data, result.elapsed_s)

    def decode(
        self,
        artifact: bytes,
        reference: bytes,
        metadata: Optional[Metadata] = None,
    ) -> DecodeResult:
        del metadata
        artifact_data = _as_bytes(artifact, label="zstd patch artifact")
        reference_data = _as_bytes(reference, label="delta reference")
        with tempfile.TemporaryDirectory(
            prefix="raglm-zstd-patch-", dir=self.temp_dir
        ) as work_dir:
            artifact_path = Path(work_dir, "patch.zst")
            reference_path = Path(work_dir, "reference.bin")
            output_path = Path(work_dir, "decoded.bin")
            artifact_path.write_bytes(artifact_data)
            reference_path.write_bytes(reference_data)
            result = self._run(
                (
                    "-d",
                    "-q",
                    f"--patch-from={reference_path}",
                    artifact_path,
                    "-o",
                    output_path,
                ),
                operation="zstd patch-from decode",
            )
            output = read_file(output_path, operation="zstd patch decode")
        return DecodeResult(output, result.elapsed_s)


class BSDiffCodec(DeltaCodec):
    """Reference implementation adapter for ``bsdiff`` and ``bspatch``."""

    name = "bsdiff"

    def __init__(
        self,
        *,
        binary: Optional[str | os.PathLike[str]] = None,
        decoder_binary: Optional[str | os.PathLike[str]] = None,
        timeout_s: float = 300.0,
        temp_dir: Optional[str | os.PathLike[str]] = None,
    ) -> None:
        super().__init__(timeout_s=timeout_s)
        self.binary = discover_binary(("bsdiff",), binary)
        self.decoder_binary = discover_binary(("bspatch",), decoder_binary)
        self.temp_dir = os.fspath(temp_dir) if temp_dir is not None else None

    def encode(
        self,
        target: bytes,
        reference: bytes,
        metadata: Optional[Metadata] = None,
    ) -> EncodeResult:
        del metadata
        target_data = _as_bytes(target, label="delta target")
        reference_data = _as_bytes(reference, label="delta reference")
        with tempfile.TemporaryDirectory(
            prefix="raglm-bsdiff-", dir=self.temp_dir
        ) as work_dir:
            reference_path = Path(work_dir, "reference.bin")
            target_path = Path(work_dir, "target.bin")
            artifact_path = Path(work_dir, "patch.bsdiff")
            reference_path.write_bytes(reference_data)
            target_path.write_bytes(target_data)
            result = run_command(
                (self.binary, reference_path, target_path, artifact_path),
                timeout_s=self.timeout_s,
                operation="bsdiff encode",
            )
            artifact_data = read_file(artifact_path, operation="bsdiff encode")
        return EncodeResult(artifact_data, result.elapsed_s)

    def decode(
        self,
        artifact: bytes,
        reference: bytes,
        metadata: Optional[Metadata] = None,
    ) -> DecodeResult:
        del metadata
        artifact_data = _as_bytes(artifact, label="bsdiff artifact")
        reference_data = _as_bytes(reference, label="delta reference")
        with tempfile.TemporaryDirectory(
            prefix="raglm-bsdiff-", dir=self.temp_dir
        ) as work_dir:
            reference_path = Path(work_dir, "reference.bin")
            artifact_path = Path(work_dir, "patch.bsdiff")
            output_path = Path(work_dir, "decoded.bin")
            reference_path.write_bytes(reference_data)
            artifact_path.write_bytes(artifact_data)
            result = run_command(
                (self.decoder_binary, reference_path, output_path, artifact_path),
                timeout_s=self.timeout_s,
                operation="bspatch decode",
            )
            output = read_file(output_path, operation="bspatch decode")
        return DecodeResult(output, result.elapsed_s)

    def version(self) -> str:
        # The canonical bsdiff binaries expose no portable version flag.
        return "bsdiff/bspatch CLI (version flag unavailable)"

    def is_available(self) -> bool:
        return all(
            os.path.isfile(path) and os.access(path, os.X_OK)
            for path in (self.binary, self.decoder_binary)
        )


class OpenVCDiffCodec(ExternalDeltaCodec):
    """Google open-vcdiff's ``vcdiff`` command-line frontend."""

    name = "open-vcdiff"
    binary_candidates = ("vcdiff", "open-vcdiff")

    def __init__(
        self,
        *,
        binary: Optional[str | os.PathLike[str]] = None,
        timeout_s: float = 300.0,
        temp_dir: Optional[str | os.PathLike[str]] = None,
    ) -> None:
        super().__init__(binary=binary, timeout_s=timeout_s)
        self.temp_dir = os.fspath(temp_dir) if temp_dir is not None else None

    def encode(
        self,
        target: bytes,
        reference: bytes,
        metadata: Optional[Metadata] = None,
    ) -> EncodeResult:
        del metadata
        target_data = _as_bytes(target, label="delta target")
        reference_data = _as_bytes(reference, label="delta reference")
        with tempfile.TemporaryDirectory(
            prefix="raglm-vcdiff-", dir=self.temp_dir
        ) as work_dir:
            reference_path = Path(work_dir, "reference.bin")
            target_path = Path(work_dir, "target.bin")
            artifact_path = Path(work_dir, "patch.vcdiff")
            reference_path.write_bytes(reference_data)
            target_path.write_bytes(target_data)
            result = self._run(
                (
                    "encode",
                    "-target_matches",
                    "-dictionary",
                    reference_path,
                    "-target",
                    target_path,
                    "-delta",
                    artifact_path,
                ),
                operation="open-vcdiff encode",
            )
            artifact_data = read_file(artifact_path, operation="open-vcdiff encode")
        return EncodeResult(artifact_data, result.elapsed_s)

    def decode(
        self,
        artifact: bytes,
        reference: bytes,
        metadata: Optional[Metadata] = None,
    ) -> DecodeResult:
        artifact_data = _as_bytes(artifact, label="open-vcdiff artifact")
        reference_data = _as_bytes(reference, label="delta reference")
        values = _metadata(metadata)
        declared_target_size = values.get("target_size", 1 << 26)
        if (
            not isinstance(declared_target_size, int)
            or isinstance(declared_target_size, bool)
            or declared_target_size < 0
        ):
            raise ValueError(
                "open-vcdiff metadata['target_size'] must be a non-negative int"
            )
        # The upstream CLI defaults both decoder limits to 64 MiB.  Whole
        # documents/clips can exceed that, so use the evaluator-known exact
        # target length (with the upstream default as a floor).
        decode_limit = max(1 << 26, declared_target_size)
        with tempfile.TemporaryDirectory(
            prefix="raglm-vcdiff-", dir=self.temp_dir
        ) as work_dir:
            reference_path = Path(work_dir, "reference.bin")
            artifact_path = Path(work_dir, "patch.vcdiff")
            output_path = Path(work_dir, "decoded.bin")
            reference_path.write_bytes(reference_data)
            artifact_path.write_bytes(artifact_data)
            result = self._run(
                (
                    "decode",
                    f"-max_target_file_size={decode_limit}",
                    f"-max_target_window_size={decode_limit}",
                    "-dictionary",
                    reference_path,
                    "-delta",
                    artifact_path,
                    "-target",
                    output_path,
                ),
                operation="open-vcdiff decode",
            )
            output = read_file(output_path, operation="open-vcdiff decode")
        return DecodeResult(output, result.elapsed_s)


__all__ = ["BSDiffCodec", "OpenVCDiffCodec", "ZstdPatchCodec"]
