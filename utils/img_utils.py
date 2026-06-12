"""Image preprocessing for compression.

Dataset loaders are registered by name; ``load_image_files`` auto-detects
the dataset from the path and dispatches to the matching loader.

Adding a new dataset
--------------------
    from utils.img_utils import register_image_loader

    @register_image_loader("my_dataset")
    def _load_my_dataset(path: str, n: Optional[int] = None) -> List[str]:
        ...  # return list of image file paths
"""
from __future__ import annotations

import glob as _glob
import io
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------

ImageLoader = Callable[[str, Optional[int]], List[str]]

_IMAGE_LOADERS: Dict[str, ImageLoader] = {}


def register_image_loader(name: str):
    """Decorator to register an image dataset loader by name.

    Detection: *name* (hyphens normalised to underscores) must appear as a
    substring of the normalised dataset path.
    """
    def decorator(fn: ImageLoader) -> ImageLoader:
        _IMAGE_LOADERS[name] = fn
        return fn
    return decorator


def _find_image_loader(path: str) -> ImageLoader:
    key = os.path.basename(os.path.normpath(path)).lower().replace("-", "_")
    for name, loader in _IMAGE_LOADERS.items():
        if name.replace("-", "_") in key:
            return loader
    raise ValueError(
        f"No image loader registered for {path!r}.\n"
        f"Known datasets: {sorted(_IMAGE_LOADERS)}.\n"
        f"Register a new one with @register_image_loader('name')."
    )


def load_image_files(path: str, n: Optional[int] = None) -> List[str]:
    """Dispatch to the registered loader for the image dataset at *path*."""
    return _find_image_loader(path)(path, n)


# ---------------------------------------------------------------------------
# Shared low-level helpers used by built-in loaders
# ---------------------------------------------------------------------------

def _load_image_dir(
    path: str,
    n: Optional[int],
    extensions: Sequence[str] = (
        ".bmp", ".png", ".jpg", ".jpeg", ".webp", ".tiff"),
) -> List[str]:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Image dataset directory not found: {path}")
    files: List[str] = []
    for ext in extensions:
        files.extend(_glob.glob(os.path.join(path, f"*{ext}")))
        files.extend(_glob.glob(os.path.join(path, f"*{ext.upper()}")))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"No image files ({', '.join(extensions)}) found in {path}"
        )
    return files[:n] if n is not None else files


# ---------------------------------------------------------------------------
# Built-in dataset loaders
# ---------------------------------------------------------------------------

@register_image_loader("clic2024")
def _load_clic2024(path: str, n: Optional[int] = None) -> List[str]:
    return _load_image_dir(path, n, extensions=(".bmp",))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ImagePatchRecord:
    """A single image patch with provenance, produced by patchify_images_for_compression.

    Mirrors TextChunk in text_utils: patch is the payload, sample_idx identifies
    the source image so chunk-level results can be aggregated back per image.
    meta is shared across all patches of the same image and is stored here so
    the aggregation step needs only this flat list — no separate sample_info dict.
    """
    patch:      "ImagePatch"
    sample_idx: int             # index into the worker's local sample list
    meta:       "ImagePatchMeta"


@dataclass
class ImagePatch:
    index: int
    x: int
    y: int
    width: int
    height: int
    data: bytes          # BMP bytes for this patch


@dataclass
class ImagePatchMeta:
    original_width: int
    original_height: int
    padded_width: int
    padded_height: int
    patch_size: int
    mode: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _image_to_pil(data: Any) -> Image.Image:
    if isinstance(data, Image.Image):
        return data
    if isinstance(data, np.ndarray):
        return Image.fromarray(data)
    if isinstance(data, (bytes, bytearray)):
        return Image.open(io.BytesIO(data)).convert("RGB")
    if isinstance(data, dict):
        if data.get("bytes") is not None:
            return Image.open(io.BytesIO(data["bytes"])).convert("RGB")
        if data.get("path") is not None:
            return Image.open(data["path"]).convert("RGB")
        if data.get("array") is not None:
            return Image.fromarray(np.asarray(data["array"]))
    raise TypeError(f"Unsupported image data type: {type(data)!r}")


def _pil_to_bmp_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="BMP")
    return buf.getvalue()


def _bmp_bytes_to_pil(raw_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def patchify_image(
    data: Any,
    patch_size: int = 32,
    mode: str = "RGB",
) -> Tuple[List[ImagePatch], ImagePatchMeta]:
    """Convert an image row to fixed-size BMP patches.

    The image is padded on the right/bottom to a multiple of patch_size.
    Reassembly crops back to the original dimensions.
    """
    image = _image_to_pil(data).convert(mode)
    original_width, original_height = image.size

    padded_width = ((original_width + patch_size - 1) //
                    patch_size) * patch_size
    padded_height = ((original_height + patch_size - 1) //
                     patch_size) * patch_size

    padded = Image.new(mode, (padded_width, padded_height))
    padded.paste(image, (0, 0))

    patches: List[ImagePatch] = []
    idx = 0
    for y in range(0, padded_height, patch_size):
        for x in range(0, padded_width, patch_size):
            patch = padded.crop((x, y, x + patch_size, y + patch_size))
            patches.append(ImagePatch(
                index=idx, x=x, y=y,
                width=patch_size, height=patch_size,
                data=_pil_to_bmp_bytes(patch),
            ))
            idx += 1

    meta = ImagePatchMeta(
        original_width=original_width,
        original_height=original_height,
        padded_width=padded_width,
        padded_height=padded_height,
        patch_size=patch_size,
        mode=mode,
    )
    return patches, meta


def patchify_images_for_compression(
    bmp_files: List[str],
    indices: List[int],
    patch_size: int = 32,
) -> List[ImagePatchRecord]:
    """Patchify all images in a worker shard into a flat list of ImagePatchRecords.

    Pure preprocessing step — no compression logic involved.
    Mirrors chunk_documents_for_compression in text_utils.
    """
    all_records: List[ImagePatchRecord] = []
    for local_idx, i in enumerate(indices):
        with open(bmp_files[i], "rb") as f:
            patches, meta = patchify_image(f.read(), patch_size=patch_size)
        for p in patches:
            all_records.append(ImagePatchRecord(
                patch=p, sample_idx=local_idx, meta=meta))
    return all_records


def reassemble_image_patches(
    patches: Sequence[ImagePatch],
    meta: ImagePatchMeta,
) -> Image.Image:
    """Reassemble decoded BMP patch bytes back into a PIL image."""
    canvas = Image.new(meta.mode, (meta.padded_width, meta.padded_height))
    for patch in sorted(patches, key=lambda p: p.index):
        canvas.paste(_bmp_bytes_to_pil(patch.data).convert(
            meta.mode), (patch.x, patch.y))
    return canvas.crop((0, 0, meta.original_width, meta.original_height))
