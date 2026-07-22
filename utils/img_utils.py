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
import os
import pickle
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image


# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------

ImageLoader = Callable[[str, Optional[int]], List[str]]

_IMAGE_LOADERS: Dict[str, ImageLoader] = {}


def register_image_loader(name: str):
    """Decorator to register an image dataset loader by name.

    Detection: *name* (hyphens normalised to underscores) must appear as a
    substring of the whole normalised dataset path, so multi-segment names
    like ``"eurosat/Forest"`` match a path ``datasets/eurosat/Forest`` — not
    just the final basename. When several names match, the longest (most
    specific) one wins.
    """
    def decorator(fn: ImageLoader) -> ImageLoader:
        _IMAGE_LOADERS[name] = fn
        return fn
    return decorator


def _find_image_loader(path: str) -> ImageLoader:
    key = os.path.normpath(path).replace(os.sep, "/").lower().replace("-", "_")
    best_loader: Optional[ImageLoader] = None
    best_len = -1
    for name, loader in _IMAGE_LOADERS.items():
        needle = name.replace(os.sep, "/").lower().replace("-", "_")
        if needle in key and len(needle) > best_len:
            best_loader, best_len = loader, len(needle)
    if best_loader is not None:
        return best_loader
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


@register_image_loader("eval_samples.pkl")
def load_rac_eval_samples(path: str, n: Optional[int] = None) -> List[str]:
    """Load held-out image paths emitted by prepare_rac_data_bgpt."""
    pkl_path = path
    if os.path.isdir(path):
        pkl_path = os.path.join(path, "eval_samples.pkl")
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"RAC eval pickle not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)
    selected = samples[:n] if n is not None else samples
    for sample_idx, sample in enumerate(selected):
        if not isinstance(sample, str):
            raise TypeError(
                f"RAC image sample {sample_idx} must be a file path string, "
                f"got {type(sample)!r}; rebuild the database")
    return selected


# ---------------------------------------------------------------------------
# Built-in dataset loaders
# ---------------------------------------------------------------------------

@register_image_loader("eurosat/Forest")
def load_eurosat_forest(path: str, n: Optional[int] = None) -> List[str]:
    return _load_image_dir(path, n, extensions=(".bmp",))

@register_image_loader("eurosat/Industry")
def load_eurosat_industry(path: str, n: Optional[int] = None) -> List[str]:
    return _load_image_dir(path, n, extensions=(".bmp",))

@register_image_loader("medmnist/bloodmnist28/test")
def load_eurosat_medmnist_bloodmnist28(path: str, n: Optional[int] = None) -> List[str]:
    return _load_image_dir(path, n, extensions=(".bmp",))

@register_image_loader("medmnist/retinamnist128/test")
def load_eurosat_medmnist_retinamnist128(path: str, n: Optional[int] = None) -> List[str]:
    return _load_image_dir(path, n, extensions=(".bmp",))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ImagePatchRecord:
    """One image patch with provenance, produced by patchify_images_for_compression.

    The 2D analog of ``AudioChunkRecord`` — one flat struct, no nesting. ``data``
    is the header-free BMP pixel array: BGR channels, bottom-up rows, and
    four-byte row alignment. ``sample_idx`` identifies the source image and
    ``patch_idx`` is the raster position within it. ``orig_width``/``orig_height``
    and the nominal ``patch_width``/``patch_height`` are enough to place the
    patch and rebuild the image. Patch x/y and its clipped size are derived from
    ``patch_idx`` at reassembly, not stored — an edge patch holds fewer bytes
    than the nominal rectangle because it is clipped to the image rather than
    padded out to it.
    """
    data:        bytes
    sample_idx:  int   # index into the worker's local sample list
    patch_idx:   int   # 0-based raster position within this image's patches
    orig_width:  int   # source image width  (to place patches and rebuild)
    orig_height: int   # source image height
    patch_width: int   # nominal grid patch width in pixels
    patch_height: int  # nominal grid patch height in pixels


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bmp_row_stride(width: int) -> int:
    """Return the byte width of one 24-bit BMP row, including alignment."""
    return ((width * 3 + 3) // 4) * 4


def bmp_payload_nbytes(width: int, height: int) -> int:
    """Byte length of a header-free 24-bit BMP pixel array."""
    if width <= 0 or height <= 0:
        raise ValueError(
            f"BMP patch dimensions must be positive, got {width}x{height}")
    return _bmp_row_stride(width) * height


def validate_bmp_patch_shape(
    payload_bytes: int,
    width: int,
    height: int,
) -> Tuple[int, int]:
    """Validate an explicit rectangle against an exact BMP payload budget."""
    if payload_bytes <= 0:
        raise ValueError(f"payload_bytes must be positive, got {payload_bytes}")
    if width <= 0:
        raise ValueError(f"image patch width must be positive, got {width}")
    if height <= 0:
        raise ValueError(f"image patch height must be positive, got {height}")
    actual = bmp_payload_nbytes(width, height)
    if actual != payload_bytes:
        raise ValueError(
            f"image patch {width}x{height} has a {actual} B BMP payload, "
            f"but this window requires exactly {payload_bytes} B")
    return width, height


def _pil_to_bmp_payload(image: Image.Image) -> bytes:
    """Encode an RGB image as BMP's pixel array, without file/DIB headers."""
    if image.mode != "RGB":
        raise ValueError(f"bGPT image patches must be RGB, got mode={image.mode!r}")
    return image.tobytes(
        "raw", "BGR", _bmp_row_stride(image.width), -1)


def _bmp_payload_to_pil(data: bytes, width: int, height: int) -> Image.Image:
    """Decode a header-free 24-bit BMP pixel array into an RGB image."""
    stride = _bmp_row_stride(width)
    expected = stride * height
    if len(data) != expected:
        raise ValueError(
            f"Invalid BMP pixel payload length: got {len(data)}, expected {expected}"
        )
    return Image.frombytes("RGB", (width, height), data, "raw", "BGR", stride, -1)


def _patch_boxes(
    width: int,
    height: int,
    patch_width: int,
    patch_height: int,
) -> List[Tuple[int, int, int, int]]:
    """Raster-ordered ``(x, y, width, height)`` of the patches covering an image.

    Edge patches are *clipped* to the image rather than padded out to the
    nominal rectangle, so the patches partition exactly the source pixels. This
    is the single definition of the patch grid: splitting and reassembly both
    read it, so they cannot disagree.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive, got {width}x{height}")
    if patch_width <= 0 or patch_height <= 0:
        raise ValueError(
            "image patch dimensions must be positive, got "
            f"{patch_width}x{patch_height}")
    return [
        (x, y, min(patch_width, width - x), min(patch_height, height - y))
        for y in range(0, height, patch_height)
        for x in range(0, width, patch_width)
    ]


def patchify_pil_image(
    image: Image.Image,
    patch_width: int,
    patch_height: int,
) -> List[bytes]:
    """Split an RGB image into raster-ordered, header-free BMP pixel arrays.

    Because edge patches are clipped (see :func:`_patch_boxes`), the payloads
    concatenate to exactly the source pixels: no invented bytes are compressed
    and the total is the image's true size. Each patch keeps BMP's
    BGR/bottom-up/four-byte-aligned row layout but carries no BMP or DIB header.
    """
    width, height = image.size
    return [
        _pil_to_bmp_payload(image.crop((x, y, x + w, y + h)))
        for x, y, w, h in _patch_boxes(width, height, patch_width, patch_height)
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def patchify_images_for_compression(
    image_files: List[str],
    indices: List[int],
    patch_width: int,
    patch_height: int,
) -> List[ImagePatchRecord]:
    """Patchify all images in a worker shard into a flat list of ImagePatchRecords.

    Pure preprocessing step — no compression logic involved. The 2D counterpart
    of ``chunk_audio_for_compression``: yields BMP-compatible pixel payloads
    without repeating a BMP header for every patch, and — like the audio
    chunker's short final chunk — clips edge patches to the image so the
    payloads sum to exactly the image's own byte count. The dimensions are
    explicit because RAC payloads may be rectangular rather than square.
    """
    all_records: List[ImagePatchRecord] = []
    for local_idx, i in enumerate(indices):
        with Image.open(image_files[i]) as source:
            image = source.convert("RGB")
        orig_w, orig_h = image.size
        patch_datas = patchify_pil_image(image, patch_width, patch_height)
        for patch_idx, data in enumerate(patch_datas):
            all_records.append(ImagePatchRecord(
                data=data, sample_idx=local_idx, patch_idx=patch_idx,
                orig_width=orig_w, orig_height=orig_h,
                patch_width=patch_width, patch_height=patch_height))
    return all_records


def reassemble_image_patches(
    patch_datas: Sequence[bytes],
    orig_width: int,
    orig_height: int,
    patch_width: int,
    patch_height: int,
) -> Image.Image:
    """Rebuild a PIL image from raster-ordered BMP pixel payloads.

    The 2D analog of ``pcm_payload_to_wav`` — re-wraps header-free payload back
    into a viewable form. ``patch_datas`` must be the image's whole patch set in
    raster order; each patch's x/y and its clipped size are derived from its
    position, mirroring :func:`patchify_pil_image`.
    """
    boxes = _patch_boxes(orig_width, orig_height, patch_width, patch_height)
    if len(patch_datas) != len(boxes):
        raise ValueError(
            f"got {len(patch_datas)} patches, expected {len(boxes)} for "
            f"a {orig_width}x{orig_height} image with "
            f"{patch_width}x{patch_height} patches")

    canvas = Image.new("RGB", (orig_width, orig_height))
    for data, (x, y, patch_w, patch_h) in zip(patch_datas, boxes):
        canvas.paste(_bmp_payload_to_pil(data, patch_w, patch_h), (x, y))
    return canvas
