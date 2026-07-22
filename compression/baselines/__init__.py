"""Factories and public API for conventional compression baselines."""
from __future__ import annotations

import os
from typing import Any, Optional

from .base import (
    BaselineCodec,
    CodecError,
    CodecExecutionError,
    CodecTimeoutError,
    DecodeResult,
    DeltaCodec,
    EncodeResult,
    InvalidMetadataError,
    MissingBinaryError,
    binary_available,
    discover_binary,
)
from .delta import BSDiffCodec, OpenVCDiffCodec, ZstdPatchCodec
from .generic import (
    CmixCodec,
    DictionaryTrainResult,
    OpenZLCodec,
    ZstdCodec,
    ZstdDictionaryCodec,
    train_zstd_dictionary,
)
from .media import FLACCodec, JPEGXLCodec, PNGCodec, WebPCodec


CODEC_NAMES = (
    "zstd",
    "openzl",
    "cmix",
    "zstd-dict",
    "png",
    "webp",
    "jpegxl",
    "flac",
)

DELTA_CODEC_NAMES = ("zstd-patch", "bsdiff", "open-vcdiff")


def _normalise_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"codec name must be a non-empty string, got {name!r}")
    return name.strip().lower().replace("_", "-")


def create_codec(
    name: str,
    *,
    timeout_s: float = 300.0,
    tmp_dir: Optional[str | os.PathLike[str]] = None,
    temp_dir: Optional[str | os.PathLike[str]] = None,
    **kwargs: Any,
) -> BaselineCodec:
    """Create a standalone baseline by its stable command-line name."""
    key = _normalise_name(name)
    aliases = {
        "open-zl": "openzl",
        "zstd-dictionary": "zstd-dict",
        "jxl": "jpegxl",
        "jpeg-xl": "jpegxl",
    }
    key = aliases.get(key, key)
    if tmp_dir is not None and temp_dir is not None:
        raise ValueError("Pass only one of tmp_dir and temp_dir")
    work_dir = temp_dir if temp_dir is not None else tmp_dir

    if key == "zstd":
        codec: BaselineCodec = ZstdCodec(timeout_s=timeout_s, **kwargs)
    elif key == "openzl":
        codec = OpenZLCodec(timeout_s=timeout_s, temp_dir=work_dir, **kwargs)
    elif key == "cmix":
        codec = CmixCodec(timeout_s=timeout_s, temp_dir=work_dir, **kwargs)
    elif key == "zstd-dict":
        codec = ZstdDictionaryCodec(
            timeout_s=timeout_s, temp_dir=work_dir, **kwargs
        )
    elif key == "png":
        codec = PNGCodec(timeout_s=timeout_s, **kwargs)
    elif key == "webp":
        codec = WebPCodec(timeout_s=timeout_s, **kwargs)
    elif key == "jpegxl":
        codec = JPEGXLCodec(timeout_s=timeout_s, temp_dir=work_dir, **kwargs)
    elif key == "flac":
        codec = FLACCodec(timeout_s=timeout_s, **kwargs)
    else:
        raise ValueError(
            f"Unknown standalone codec {name!r}; choose one of {', '.join(CODEC_NAMES)}"
        )

    return codec


def create_delta_codec(
    name: str,
    *,
    timeout_s: float = 300.0,
    tmp_dir: Optional[str | os.PathLike[str]] = None,
    temp_dir: Optional[str | os.PathLike[str]] = None,
    **kwargs: Any,
) -> DeltaCodec:
    """Create a whole-target delta codec.

    Reference index bits are deliberately outside this API and must be charged
    by the evaluator.
    """
    key = _normalise_name(name)
    aliases = {
        "zstd-patch-from": "zstd-patch",
        "patch-from": "zstd-patch",
        "vcdiff": "open-vcdiff",
        "openvcdiff": "open-vcdiff",
    }
    key = aliases.get(key, key)
    if tmp_dir is not None and temp_dir is not None:
        raise ValueError("Pass only one of tmp_dir and temp_dir")
    work_dir = temp_dir if temp_dir is not None else tmp_dir

    if key == "zstd-patch":
        return ZstdPatchCodec(timeout_s=timeout_s, temp_dir=work_dir, **kwargs)
    if key == "bsdiff":
        return BSDiffCodec(timeout_s=timeout_s, temp_dir=work_dir, **kwargs)
    if key == "open-vcdiff":
        return OpenVCDiffCodec(timeout_s=timeout_s, temp_dir=work_dir, **kwargs)
    raise ValueError(
        f"Unknown delta codec {name!r}; choose one of {', '.join(DELTA_CODEC_NAMES)}"
    )


__all__ = [
    "BSDiffCodec",
    "BaselineCodec",
    "CODEC_NAMES",
    "CmixCodec",
    "CodecError",
    "CodecExecutionError",
    "CodecTimeoutError",
    "DELTA_CODEC_NAMES",
    "DecodeResult",
    "DeltaCodec",
    "DictionaryTrainResult",
    "EncodeResult",
    "FLACCodec",
    "InvalidMetadataError",
    "JPEGXLCodec",
    "MissingBinaryError",
    "OpenVCDiffCodec",
    "OpenZLCodec",
    "PNGCodec",
    "WebPCodec",
    "ZstdCodec",
    "ZstdDictionaryCodec",
    "ZstdPatchCodec",
    "binary_available",
    "create_codec",
    "create_delta_codec",
    "discover_binary",
    "train_zstd_dictionary",
]
