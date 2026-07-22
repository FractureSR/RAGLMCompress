"""Whole-sample, modality-agnostic baseline codecs."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .base import (
    DecodeResult,
    EncodeResult,
    ExternalCodec,
    Metadata,
    _as_bytes,
    read_file,
    run_command,
)


class ZstdCodec(ExternalCodec):
    """Zstandard's strongest preset, one thread, one frame per sample."""

    name = "zstd"
    binary_candidates = ("zstd",)

    def encode(
        self, payload: bytes, metadata: Optional[Metadata] = None
    ) -> EncodeResult:
        del metadata
        data = _as_bytes(payload, label="payload")
        result = self._run(
            ("--ultra", "-22", "-T1", "-q", "--stdout"),
            input_data=data,
            operation="zstd-22 encode",
        )
        return EncodeResult(result.stdout, result.elapsed_s)

    def decode(
        self, artifact: bytes, metadata: Optional[Metadata] = None
    ) -> DecodeResult:
        del metadata
        encoded = _as_bytes(artifact, label="artifact")
        result = self._run(
            ("-d", "-q", "--stdout"),
            input_data=encoded,
            operation="zstd decode",
        )
        return DecodeResult(result.stdout, result.elapsed_s)


class OpenZLCodec(ExternalCodec):
    """OpenZL serial-profile adapter.

    The adapter uses explicit input and output paths because the OpenZL CLI is
    intentionally stream-format agnostic.  ``serial`` is the untrained generic
    profile; dataset-trained graphs should be evaluated as separate shared
    state, rather than silently substituted here.
    """

    name = "openzl"
    # ``zli`` is the CLI built by release-tagged OpenZL versions.  Do not
    # silently pick an unrelated executable named ``openzl`` from PATH.
    binary_candidates = ("zli",)

    def __init__(
        self,
        *,
        binary: Optional[str | os.PathLike[str]] = None,
        timeout_s: float = 300.0,
        profile: str = "serial",
        temp_dir: Optional[str | os.PathLike[str]] = None,
    ) -> None:
        super().__init__(binary=binary, timeout_s=timeout_s)
        if not profile or profile.startswith("-"):
            raise ValueError(f"Invalid OpenZL profile: {profile!r}")
        self.profile = profile
        self.temp_dir = os.fspath(temp_dir) if temp_dir is not None else None

    def encode(
        self, payload: bytes, metadata: Optional[Metadata] = None
    ) -> EncodeResult:
        del metadata
        data = _as_bytes(payload, label="payload")
        with tempfile.TemporaryDirectory(
            prefix="raglm-openzl-", dir=self.temp_dir
        ) as temp_dir:
            source = Path(temp_dir, "source.bin")
            artifact = Path(temp_dir, "artifact.openzl")
            source.write_bytes(data)
            result = self._run(
                (
                    "compress",
                    "--profile",
                    self.profile,
                    source,
                    "--output",
                    artifact,
                ),
                operation=f"OpenZL {self.profile} encode",
            )
            output = read_file(artifact, operation="OpenZL encode")
        return EncodeResult(output, result.elapsed_s)

    def decode(
        self, artifact: bytes, metadata: Optional[Metadata] = None
    ) -> DecodeResult:
        del metadata
        encoded = _as_bytes(artifact, label="artifact")
        with tempfile.TemporaryDirectory(
            prefix="raglm-openzl-", dir=self.temp_dir
        ) as temp_dir:
            source = Path(temp_dir, "artifact.openzl")
            output = Path(temp_dir, "decoded.bin")
            source.write_bytes(encoded)
            result = self._run(
                ("decompress", source, "--output", output),
                operation="OpenZL decode",
            )
            decoded = read_file(output, operation="OpenZL decode")
        return DecodeResult(decoded, result.elapsed_s)


class CmixCodec(ExternalCodec):
    """cmix adapter (extremely slow and memory intensive)."""

    name = "cmix"
    binary_candidates = ("cmix",)

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
        self, payload: bytes, metadata: Optional[Metadata] = None
    ) -> EncodeResult:
        del metadata
        data = _as_bytes(payload, label="payload")
        with tempfile.TemporaryDirectory(
            prefix="raglm-cmix-", dir=self.temp_dir
        ) as temp_dir:
            source = Path(temp_dir, "source.bin")
            artifact = Path(temp_dir, "artifact.cmix")
            source.write_bytes(data)
            result = self._run(
                ("-c", source, artifact), operation="cmix encode"
            )
            output = read_file(artifact, operation="cmix encode")
        return EncodeResult(output, result.elapsed_s)

    def decode(
        self, artifact: bytes, metadata: Optional[Metadata] = None
    ) -> DecodeResult:
        del metadata
        encoded = _as_bytes(artifact, label="artifact")
        with tempfile.TemporaryDirectory(
            prefix="raglm-cmix-", dir=self.temp_dir
        ) as temp_dir:
            source = Path(temp_dir, "artifact.cmix")
            output = Path(temp_dir, "decoded.bin")
            source.write_bytes(encoded)
            result = self._run(
                ("-d", source, output), operation="cmix decode"
            )
            decoded = read_file(output, operation="cmix decode")
        return DecodeResult(decoded, result.elapsed_s)

    def version(self) -> str:
        # The reference cmix CLI has no stable, side-effect-free version flag.
        return f"{Path(self.binary).name} CLI (version flag unavailable)"


class ZstdDictionaryCodec(ExternalCodec):
    """zstd-22 with an externally shared, trained dictionary."""

    name = "zstd-dict"
    binary_candidates = ("zstd",)

    def __init__(
        self,
        dictionary: bytes,
        *,
        binary: Optional[str | os.PathLike[str]] = None,
        timeout_s: float = 300.0,
        temp_dir: Optional[str | os.PathLike[str]] = None,
    ) -> None:
        super().__init__(binary=binary, timeout_s=timeout_s)
        self.dictionary = _as_bytes(dictionary, label="dictionary")
        if not self.dictionary:
            raise ValueError("dictionary must not be empty")
        self.temp_dir = os.fspath(temp_dir) if temp_dir is not None else None

    def _run_with_dictionary(
        self, args: Sequence[str], data: bytes, *, operation: str
    ):
        with tempfile.TemporaryDirectory(
            prefix="raglm-zstd-dict-", dir=self.temp_dir
        ) as temp_dir:
            dictionary_path = Path(temp_dir, "dictionary.zdict")
            dictionary_path.write_bytes(self.dictionary)
            return self._run(
                (*args, "-D", dictionary_path, "--stdout"),
                input_data=data,
                operation=operation,
            )

    def encode(
        self, payload: bytes, metadata: Optional[Metadata] = None
    ) -> EncodeResult:
        del metadata
        data = _as_bytes(payload, label="payload")
        result = self._run_with_dictionary(
            ("--ultra", "-22", "-T1", "-q"), data, operation="zstd dictionary encode"
        )
        return EncodeResult(result.stdout, result.elapsed_s)

    def decode(
        self, artifact: bytes, metadata: Optional[Metadata] = None
    ) -> DecodeResult:
        del metadata
        encoded = _as_bytes(artifact, label="artifact")
        result = self._run_with_dictionary(
            ("-d", "-q"), encoded, operation="zstd dictionary decode"
        )
        return DecodeResult(result.stdout, result.elapsed_s)


@dataclass(frozen=True)
class DictionaryTrainResult:
    dictionary: bytes
    train_s: float

    @property
    def dictionary_bytes(self) -> bytes:
        return self.dictionary

    @property
    def elapsed_s(self) -> float:
        return self.train_s


def train_zstd_dictionary(
    samples: Iterable[bytes],
    *,
    dictionary_size: int = 112_640,
    binary: Optional[str | os.PathLike[str]] = None,
    timeout_s: float = 600.0,
    temp_dir: Optional[str | os.PathLike[str]] = None,
) -> DictionaryTrainResult:
    """Train shared zstd state exclusively from caller-provided base samples."""
    if dictionary_size <= 0:
        raise ValueError(f"dictionary_size must be positive, got {dictionary_size}")
    codec_binary = ZstdCodec(binary=binary, timeout_s=timeout_s).binary
    with tempfile.TemporaryDirectory(
        prefix="raglm-zstd-train-",
        dir=os.fspath(temp_dir) if temp_dir is not None else None,
    ) as temp_dir:
        samples_dir = Path(temp_dir, "samples")
        samples_dir.mkdir()
        sample_count = 0
        for index, sample in enumerate(samples):
            data = _as_bytes(sample, label="dictionary sample")
            path = Path(samples_dir, f"sample-{index:08d}.bin")
            path.write_bytes(data)
            sample_count += 1
        if not sample_count:
            raise ValueError("At least one dictionary training sample is required")
        dictionary_path = Path(temp_dir, "trained.zdict")
        result = run_command(
            (
                codec_binary,
                "--train",
                "-T1",
                f"--maxdict={dictionary_size}",
                "-q",
                "-r",
                "-o",
                dictionary_path,
                samples_dir,
            ),
            timeout_s=timeout_s,
            operation="zstd dictionary training",
        )
        dictionary = read_file(dictionary_path, operation="zstd dictionary training")
    return DictionaryTrainResult(dictionary, result.elapsed_s)


__all__ = [
    "CmixCodec",
    "DictionaryTrainResult",
    "OpenZLCodec",
    "ZstdCodec",
    "ZstdDictionaryCodec",
    "train_zstd_dictionary",
]
