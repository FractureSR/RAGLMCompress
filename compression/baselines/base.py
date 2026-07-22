"""Common interfaces and subprocess helpers for conventional baselines.

The baseline package deliberately does not depend on the model compressor
classes.  A codec consumes the exact byte payload selected by an evaluator and
returns the complete, self-contained codec artifact.  Dataset-level shared
state (for example a zstd dictionary or a delta reference) is passed
separately, so an evaluator can account for it explicitly.
"""
from __future__ import annotations

import abc
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


Metadata = Mapping[str, Any]


class CodecError(RuntimeError):
    """Base class for all baseline codec failures."""


class MissingBinaryError(CodecError):
    """Raised when an optional command-line codec is not installed."""


class CodecTimeoutError(CodecError):
    """Raised when a codec command exceeds its configured time limit."""


class CodecExecutionError(CodecError):
    """Raised when a codec command exits unsuccessfully."""


class InvalidMetadataError(CodecError, ValueError):
    """Raised when the raw payload metadata is absent or inconsistent."""


@dataclass(frozen=True)
class EncodeResult:
    """Artifact and wall-clock time returned by :meth:`BaselineCodec.encode`."""

    artifact: bytes
    encode_s: float

    @property
    def artifact_bytes(self) -> bytes:
        """Compatibility alias with an explicit unit-bearing name."""
        return self.artifact

    @property
    def elapsed_s(self) -> float:
        return self.encode_s


@dataclass(frozen=True)
class DecodeResult:
    """Reconstructed payload and wall-clock decode time."""

    payload: bytes
    decode_s: float

    @property
    def payload_bytes(self) -> bytes:
        return self.payload

    @property
    def elapsed_s(self) -> float:
        return self.decode_s


@dataclass(frozen=True)
class CommandResult:
    stdout: bytes
    stderr: bytes
    elapsed_s: float


def _as_bytes(value: bytes | bytearray | memoryview, *, label: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{label} must be bytes-like, got {type(value).__name__}")
    return bytes(value)


def _metadata(metadata: Optional[Metadata]) -> Metadata:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError(
            f"metadata must be a mapping or None, got {type(metadata).__name__}"
        )
    return metadata


def discover_binary(
    candidates: Sequence[str], override: Optional[str | os.PathLike[str]] = None
) -> str:
    """Resolve an executable without invoking a shell.

    ``override`` may be either a command name or an explicit path.  Candidate
    order is significant and lets adapters support common alternate names.
    """
    requested = [os.fspath(override)] if override is not None else list(candidates)
    for value in requested:
        if not value:
            continue
        contains_sep = os.sep in value or (os.altsep is not None and os.altsep in value)
        if contains_sep:
            path = os.path.abspath(os.path.expanduser(value))
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        else:
            path = shutil.which(value)
            if path:
                return path
    names = ", ".join(repr(name) for name in requested or candidates)
    raise MissingBinaryError(
        f"Required codec executable was not found or is not executable: {names}. "
        "Install it and ensure it is on PATH, or pass binary='/path/to/tool'."
    )


def binary_available(*candidates: str) -> bool:
    """Return whether any candidate command is executable on PATH."""
    return any(shutil.which(candidate) for candidate in candidates)


def _display_command(argv: Sequence[str]) -> str:
    return shlex.join([str(arg) for arg in argv])


def run_command(
    argv: Sequence[str | os.PathLike[str]],
    *,
    timeout_s: float,
    input_data: Optional[bytes] = None,
    cwd: Optional[str | os.PathLike[str]] = None,
    operation: str = "codec command",
) -> CommandResult:
    """Run an argv-only command and return captured byte streams and timing."""
    if not argv:
        raise ValueError("argv must not be empty")
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s}")

    args = [os.fspath(arg) for arg in argv]
    start = time.perf_counter()
    try:
        process = subprocess.run(
            args,
            input=input_data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.fspath(cwd) if cwd is not None else None,
            timeout=timeout_s,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        raise CodecTimeoutError(
            f"{operation} timed out after {elapsed:.3f}s "
            f"(limit {timeout_s:g}s): {_display_command(args)}"
        ) from exc
    except FileNotFoundError as exc:
        raise MissingBinaryError(
            f"Executable disappeared before {operation}: {args[0]!r}"
        ) from exc
    except OSError as exc:
        raise CodecExecutionError(
            f"Could not start {operation} ({_display_command(args)}): {exc}"
        ) from exc

    elapsed = time.perf_counter() - start
    if process.returncode != 0:
        detail_bytes = process.stderr.strip() or process.stdout.strip()
        detail = detail_bytes.decode("utf-8", errors="replace")[-4000:]
        if not detail:
            detail = "no diagnostic output"
        raise CodecExecutionError(
            f"{operation} failed with exit code {process.returncode}: {detail}; "
            f"command={_display_command(args)}"
        )
    return CommandResult(process.stdout, process.stderr, elapsed)


class BaselineCodec(abc.ABC):
    """Interface implemented by all standalone whole-sample codecs."""

    name: str

    def __init__(self, *, timeout_s: float = 300.0) -> None:
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s}")
        self.timeout_s = float(timeout_s)

    @abc.abstractmethod
    def encode(
        self, payload: bytes, metadata: Optional[Metadata] = None
    ) -> EncodeResult:
        raise NotImplementedError

    @abc.abstractmethod
    def decode(
        self, artifact: bytes, metadata: Optional[Metadata] = None
    ) -> DecodeResult:
        raise NotImplementedError

    def compress(self, payload: bytes, metadata: Optional[Metadata] = None) -> bytes:
        """Small compatibility wrapper returning only the artifact bytes."""
        return self.encode(payload, metadata).artifact

    def decompress(
        self, artifact: bytes, metadata: Optional[Metadata] = None
    ) -> bytes:
        """Small compatibility wrapper returning only reconstructed bytes."""
        return self.decode(artifact, metadata).payload

    def version(self) -> str:
        """Return an implementation version suitable for result manifests."""
        return "unknown"

    def is_available(self) -> bool:
        return True


class ExternalCodec(BaselineCodec):
    """Base class for adapters backed by one command-line executable."""

    binary_candidates: tuple[str, ...] = ()
    version_args: tuple[str, ...] = ("--version",)

    def __init__(
        self,
        *,
        binary: Optional[str | os.PathLike[str]] = None,
        timeout_s: float = 300.0,
    ) -> None:
        super().__init__(timeout_s=timeout_s)
        self.binary = discover_binary(self.binary_candidates, binary)
        self._version_cache: Optional[str] = None

    def _run(
        self,
        args: Sequence[str | os.PathLike[str]],
        *,
        input_data: Optional[bytes] = None,
        cwd: Optional[str | os.PathLike[str]] = None,
        operation: str,
    ) -> CommandResult:
        return run_command(
            [self.binary, *args],
            timeout_s=self.timeout_s,
            input_data=input_data,
            cwd=cwd,
            operation=operation,
        )

    def version(self) -> str:
        if self._version_cache is None:
            result = self._run(self.version_args, operation=f"{self.name} version query")
            text = (result.stdout or result.stderr).decode(
                "utf-8", errors="replace"
            ).strip()
            self._version_cache = text.splitlines()[0] if text else "unknown"
        return self._version_cache

    def is_available(self) -> bool:
        return os.path.isfile(self.binary) and os.access(self.binary, os.X_OK)


class DeltaCodec(abc.ABC):
    """Interface for whole-target codecs conditioned on a whole reference."""

    name: str

    def __init__(self, *, timeout_s: float = 300.0) -> None:
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s}")
        self.timeout_s = float(timeout_s)

    @abc.abstractmethod
    def encode(
        self,
        target: bytes,
        reference: bytes,
        metadata: Optional[Metadata] = None,
    ) -> EncodeResult:
        raise NotImplementedError

    @abc.abstractmethod
    def decode(
        self,
        artifact: bytes,
        reference: bytes,
        metadata: Optional[Metadata] = None,
    ) -> DecodeResult:
        raise NotImplementedError

    def compress(
        self,
        target: bytes,
        reference: bytes,
        metadata: Optional[Metadata] = None,
    ) -> bytes:
        return self.encode(target, reference, metadata).artifact

    def decompress(
        self,
        artifact: bytes,
        reference: bytes,
        metadata: Optional[Metadata] = None,
    ) -> bytes:
        return self.decode(artifact, reference, metadata).payload

    def version(self) -> str:
        return "unknown"

    def is_available(self) -> bool:
        return True


class ExternalDeltaCodec(DeltaCodec):
    binary_candidates: tuple[str, ...] = ()
    version_args: tuple[str, ...] = ("--version",)

    def __init__(
        self,
        *,
        binary: Optional[str | os.PathLike[str]] = None,
        timeout_s: float = 300.0,
    ) -> None:
        super().__init__(timeout_s=timeout_s)
        self.binary = discover_binary(self.binary_candidates, binary)
        self._version_cache: Optional[str] = None

    def _run(
        self,
        args: Sequence[str | os.PathLike[str]],
        *,
        input_data: Optional[bytes] = None,
        cwd: Optional[str | os.PathLike[str]] = None,
        operation: str,
    ) -> CommandResult:
        return run_command(
            [self.binary, *args],
            timeout_s=self.timeout_s,
            input_data=input_data,
            cwd=cwd,
            operation=operation,
        )

    def version(self) -> str:
        if self._version_cache is None:
            result = self._run(self.version_args, operation=f"{self.name} version query")
            text = (result.stdout or result.stderr).decode(
                "utf-8", errors="replace"
            ).strip()
            self._version_cache = text.splitlines()[0] if text else "unknown"
        return self._version_cache

    def is_available(self) -> bool:
        return os.path.isfile(self.binary) and os.access(self.binary, os.X_OK)


def read_file(path: str | os.PathLike[str], *, operation: str) -> bytes:
    """Read a codec output with a consistent missing-output diagnostic."""
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise CodecExecutionError(f"Could not read {operation} output {path!s}: {exc}") from exc


__all__ = [
    "BaselineCodec",
    "CodecError",
    "CodecExecutionError",
    "CodecTimeoutError",
    "CommandResult",
    "DecodeResult",
    "DeltaCodec",
    "EncodeResult",
    "ExternalCodec",
    "ExternalDeltaCodec",
    "InvalidMetadataError",
    "Metadata",
    "MissingBinaryError",
    "_as_bytes",
    "_metadata",
    "binary_available",
    "discover_binary",
    "read_file",
    "run_command",
]
