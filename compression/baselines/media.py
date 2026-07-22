"""Lossless whole-image and whole-audio baseline adapters."""
from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from PIL import Image, __version__ as PILLOW_VERSION, features

from .base import (
    BaselineCodec,
    CodecError,
    DecodeResult,
    EncodeResult,
    InvalidMetadataError,
    Metadata,
    _as_bytes,
    _metadata,
    discover_binary,
    read_file,
    run_command,
)


_MODE_CHANNELS = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4}


def _image_spec(metadata: Optional[Metadata]) -> tuple[int, int, str]:
    values = _metadata(metadata)
    if "size" in values:
        try:
            size_width, size_height = values["size"]
        except (TypeError, ValueError) as exc:
            raise InvalidMetadataError(
                "image metadata['size'] must be a (width, height) pair"
            ) from exc
        width = values.get("width", size_width)
        height = values.get("height", size_height)
    else:
        width = values.get("width")
        height = values.get("height")
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise InvalidMetadataError(
            f"image metadata requires a positive integer width, got {width!r}"
        )
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise InvalidMetadataError(
            f"image metadata requires a positive integer height, got {height!r}"
        )
    mode = values.get("mode", "RGB")
    if mode not in _MODE_CHANNELS:
        raise InvalidMetadataError(
            f"unsupported raw image mode {mode!r}; expected one of "
            f"{sorted(_MODE_CHANNELS)}"
        )
    channels = values.get("channels")
    if channels is not None and channels != _MODE_CHANNELS[mode]:
        raise InvalidMetadataError(
            f"metadata channels={channels!r} is inconsistent with mode={mode!r}"
        )
    return width, height, mode


def _raw_image(payload: bytes, metadata: Optional[Metadata]) -> Image.Image:
    data = _as_bytes(payload, label="image payload")
    width, height, mode = _image_spec(metadata)
    expected = width * height * _MODE_CHANNELS[mode]
    if len(data) != expected:
        raise InvalidMetadataError(
            f"raw {mode} payload has {len(data)} bytes, expected {expected} "
            f"for {width}x{height}"
        )
    return Image.frombytes(mode, (width, height), data)


def _decoded_pixels(
    artifact: bytes, metadata: Optional[Metadata], *, format_name: str
) -> bytes:
    width, height, mode = _image_spec(metadata)
    try:
        with Image.open(io.BytesIO(artifact)) as source:
            source.load()
            if source.size != (width, height):
                raise InvalidMetadataError(
                    f"decoded {format_name} size is {source.size}, expected "
                    f"{(width, height)}"
                )
            return source.convert(mode).tobytes()
    except InvalidMetadataError:
        raise
    except Exception as exc:
        raise CodecError(f"Could not decode {format_name} artifact: {exc}") from exc


class PNGCodec(BaselineCodec):
    """Pillow PNG with maximum DEFLATE effort and optimization enabled."""

    name = "png"

    def encode(
        self, payload: bytes, metadata: Optional[Metadata] = None
    ) -> EncodeResult:
        start = time.perf_counter()
        image = _raw_image(payload, metadata)
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True, compress_level=9)
        return EncodeResult(output.getvalue(), time.perf_counter() - start)

    def decode(
        self, artifact: bytes, metadata: Optional[Metadata] = None
    ) -> DecodeResult:
        start = time.perf_counter()
        encoded = _as_bytes(artifact, label="PNG artifact")
        payload = _decoded_pixels(encoded, metadata, format_name="PNG")
        return DecodeResult(payload, time.perf_counter() - start)

    def version(self) -> str:
        return f"Pillow {PILLOW_VERSION} (PNG)"


class WebPCodec(BaselineCodec):
    """Pillow/libwebp in exact lossless mode at maximum method effort."""

    name = "webp"

    def __init__(self, *, timeout_s: float = 300.0) -> None:
        super().__init__(timeout_s=timeout_s)
        if not features.check("webp"):
            raise CodecError("This Pillow build does not include lossless WebP support")

    def encode(
        self, payload: bytes, metadata: Optional[Metadata] = None
    ) -> EncodeResult:
        start = time.perf_counter()
        image = _raw_image(payload, metadata)
        output = io.BytesIO()
        image.save(
            output,
            format="WEBP",
            lossless=True,
            quality=100,
            method=6,
            exact=True,
        )
        return EncodeResult(output.getvalue(), time.perf_counter() - start)

    def decode(
        self, artifact: bytes, metadata: Optional[Metadata] = None
    ) -> DecodeResult:
        start = time.perf_counter()
        encoded = _as_bytes(artifact, label="WebP artifact")
        payload = _decoded_pixels(encoded, metadata, format_name="WebP")
        return DecodeResult(payload, time.perf_counter() - start)

    def version(self) -> str:
        libwebp = features.version("webp") or "unknown"
        return f"Pillow {PILLOW_VERSION} (libwebp {libwebp})"


class JPEGXLCodec(BaselineCodec):
    """libjxl ``cjxl``/``djxl`` adapter in mathematically lossless mode."""

    name = "jpegxl"

    def __init__(
        self,
        *,
        encoder_binary: Optional[str | os.PathLike[str]] = None,
        decoder_binary: Optional[str | os.PathLike[str]] = None,
        binary: Optional[str | os.PathLike[str]] = None,
        timeout_s: float = 300.0,
        effort: int = 10,
        temp_dir: Optional[str | os.PathLike[str]] = None,
    ) -> None:
        super().__init__(timeout_s=timeout_s)
        if not 1 <= effort <= 10:
            raise ValueError(f"JPEG XL effort must be in [1, 10], got {effort}")
        if binary is not None and encoder_binary is not None:
            raise ValueError("Pass only one of binary and encoder_binary for JPEG XL")
        self.encoder_binary = discover_binary(
            ("cjxl",), encoder_binary if encoder_binary is not None else binary
        )
        self.decoder_binary = discover_binary(("djxl",), decoder_binary)
        self.effort = int(effort)
        self.temp_dir = os.fspath(temp_dir) if temp_dir is not None else None
        self._version_cache: Optional[str] = None

    def encode(
        self, payload: bytes, metadata: Optional[Metadata] = None
    ) -> EncodeResult:
        image = _raw_image(payload, metadata)
        with tempfile.TemporaryDirectory(
            prefix="raglm-jxl-", dir=self.temp_dir
        ) as work_dir:
            source = Path(work_dir, "source.png")
            artifact = Path(work_dir, "artifact.jxl")
            image.save(source, format="PNG", compress_level=0)
            result = run_command(
                (
                    self.encoder_binary,
                    source,
                    artifact,
                    "--distance=0",
                    f"--effort={self.effort}",
                    "--num_threads=1",
                ),
                timeout_s=self.timeout_s,
                operation="JPEG XL lossless encode",
            )
            output = read_file(artifact, operation="JPEG XL encode")
        return EncodeResult(output, result.elapsed_s)

    def decode(
        self, artifact: bytes, metadata: Optional[Metadata] = None
    ) -> DecodeResult:
        encoded = _as_bytes(artifact, label="JPEG XL artifact")
        with tempfile.TemporaryDirectory(
            prefix="raglm-jxl-", dir=self.temp_dir
        ) as work_dir:
            source = Path(work_dir, "artifact.jxl")
            output = Path(work_dir, "decoded.png")
            source.write_bytes(encoded)
            result = run_command(
                (
                    self.decoder_binary,
                    source,
                    output,
                    "--num_threads=1",
                ),
                timeout_s=self.timeout_s,
                operation="JPEG XL decode",
            )
            decoded_artifact = read_file(output, operation="JPEG XL decode")
        payload = _decoded_pixels(decoded_artifact, metadata, format_name="JPEG XL")
        return DecodeResult(payload, result.elapsed_s)

    def version(self) -> str:
        if self._version_cache is None:
            enc = run_command(
                (self.encoder_binary, "--version"),
                timeout_s=self.timeout_s,
                operation="cjxl version query",
            )
            dec = run_command(
                (self.decoder_binary, "--version"),
                timeout_s=self.timeout_s,
                operation="djxl version query",
            )
            enc_text = (enc.stdout or enc.stderr).decode("utf-8", errors="replace").strip()
            dec_text = (dec.stdout or dec.stderr).decode("utf-8", errors="replace").strip()
            self._version_cache = f"{enc_text}; {dec_text}"
        return self._version_cache

    def is_available(self) -> bool:
        return all(
            os.path.isfile(path) and os.access(path, os.X_OK)
            for path in (self.encoder_binary, self.decoder_binary)
        )


def _audio_spec(metadata: Optional[Metadata]) -> tuple[int, int]:
    values = _metadata(metadata)
    sample_rate = values.get("sample_rate")
    channels = values.get("channels", 1)
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise InvalidMetadataError(
            f"FLAC metadata requires a positive integer sample_rate, got {sample_rate!r}"
        )
    if not isinstance(channels, int) or isinstance(channels, bool) or channels <= 0:
        raise InvalidMetadataError(
            f"FLAC metadata requires positive integer channels, got {channels!r}"
        )
    sample_width = values.get("sample_width", 1)
    bits_per_sample = values.get("bits_per_sample", sample_width * 8)
    if sample_width != 1 or bits_per_sample != 8:
        raise InvalidMetadataError(
            "This baseline expects the repository's PCM_U8 payload "
            f"(sample_width=1, bits_per_sample=8), got {sample_width!r} and "
            f"{bits_per_sample!r}"
        )
    return sample_rate, channels


class FLACCodec(BaselineCodec):
    """FLAC ``-8 -e -p`` over header-free interleaved PCM_U8 samples."""

    name = "flac"

    def __init__(
        self,
        *,
        binary: Optional[str | os.PathLike[str]] = None,
        timeout_s: float = 300.0,
    ) -> None:
        super().__init__(timeout_s=timeout_s)
        self.binary = discover_binary(("flac",), binary)
        self._version_cache: Optional[str] = None

    def encode(
        self, payload: bytes, metadata: Optional[Metadata] = None
    ) -> EncodeResult:
        data = _as_bytes(payload, label="PCM_U8 payload")
        sample_rate, channels = _audio_spec(metadata)
        if len(data) % channels:
            raise InvalidMetadataError(
                f"PCM payload length {len(data)} is not divisible by {channels} channels"
            )
        result = run_command(
            (
                self.binary,
                "--silent",
                "--stdout",
                "--force-raw-format",
                "--endian=little",
                "--sign=unsigned",
                f"--channels={channels}",
                "--bps=8",
                f"--sample-rate={sample_rate}",
                "--no-padding",
                "--no-seektable",
                "-8",
                "-e",
                "-p",
                "-",
            ),
            timeout_s=self.timeout_s,
            input_data=data,
            operation="FLAC PCM_U8 encode",
        )
        return EncodeResult(result.stdout, result.elapsed_s)

    def decode(
        self, artifact: bytes, metadata: Optional[Metadata] = None
    ) -> DecodeResult:
        _audio_spec(metadata)
        encoded = _as_bytes(artifact, label="FLAC artifact")
        result = run_command(
            (
                self.binary,
                "--decode",
                "--silent",
                "--stdout",
                "--force-raw-format",
                "--endian=little",
                "--sign=unsigned",
                "-",
            ),
            timeout_s=self.timeout_s,
            input_data=encoded,
            operation="FLAC PCM_U8 decode",
        )
        return DecodeResult(result.stdout, result.elapsed_s)

    def version(self) -> str:
        if self._version_cache is None:
            result = run_command(
                (self.binary, "--version"),
                timeout_s=self.timeout_s,
                operation="FLAC version query",
            )
            text = (result.stdout or result.stderr).decode(
                "utf-8", errors="replace"
            ).strip()
            self._version_cache = text.splitlines()[0] if text else "unknown"
        return self._version_cache

    def is_available(self) -> bool:
        return os.path.isfile(self.binary) and os.access(self.binary, os.X_OK)


__all__ = ["FLACCodec", "JPEGXLCodec", "PNGCodec", "WebPCodec"]
