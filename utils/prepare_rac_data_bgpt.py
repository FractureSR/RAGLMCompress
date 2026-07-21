"""Build the RAC retrieval **database** for bGPT (audio / image bytes).

The byte-domain counterpart of ``prepare_rac_data_llm.py``: fix a slice of an
audio/image dataset as the base corpus, chunk it into fixed-size **byte** chunks
(one retrieval/condition unit — an image patch or a small audio chunk), and
index them by byte similarity. ``eval_rac_bgpt.py`` then chunks + retrieves the
held-out eval samples *live*. The condition unit is independent of the eval-side
payload window: evaluation reserves one or more database chunks as conditions,
then gives the payload the remainder of the fixed 8000-byte audio or 3072-byte
image body budget.

Base chunks are kept only at their full (modal) byte length so every retrieved
condition has the same raw-byte length. Partial trailing chunks are dropped from
the base.

Audio datasets must be directories of preprocessed mono, 8-bit PCM WAV files at
their native sample rate (no resampling). Dataset download scripts perform
decoding and format conversion once; this script validates the WAV files and
indexes header-free PCM chunk payloads, chunked by a fixed byte count since
clips no longer share a common sample rate.

Outputs under ``--out``:
  base_chunks.pkl  [{id, sample_idx, ext, data (bytes)}]  retrieval units /
                   conditions; the eval reads ``base_tokens`` from ``data``.
  eval_samples.pkl held-out WAV bytes for audio, or source file paths for image;
                   the eval chunks + retrieves them live. Image source files
                   must therefore remain available.
  retriever/       saved BM25 byte index.
  meta.json        {format_version, dataset, modality, seed, base_frac, n_samples,
                    base_sample_indices, unit (patch_width/patch_height or
                    audio_chunk_bytes), unit_bytes, chunk_size (raw condition
                    bytes), patch_size, ext, payload_format, signals, kgram}.

Example
-------
    python utils/prepare_rac_data_bgpt.py \\
        --dataset datasets/clic2024/bmp --modality image --n-samples 400 \\
        --base-frac 0.5 --image-patch-width 16 --image-patch-height 16 \\
        --retriever bm25 --out results/rac_img_db
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.bgpt_codec_utils import (
    RAC_BGPT_DB_FORMAT_VERSION,
    make_bgpt_retriever,
)
from bgpt.config import PATCH_SIZE


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build the bGPT RAC retrieval database")
    p.add_argument(
        "--dataset",
        required=True,
        help="Image dataset path, or directory of preprocessed WAV files",
    )
    p.add_argument("--modality", required=True, choices=["image", "audio"])
    p.add_argument("--n-samples", type=int, default=None, help="Samples to load from the dataset")
    p.add_argument("--base-frac", type=float, default=0.5,
                   help="Fraction of samples fixed as the base/database (rest are eval)")
    p.add_argument("--image-patch-width", type=int, default=None,
                   help="image: condition patch width (required)")
    p.add_argument("--image-patch-height", type=int, default=None,
                   help="image: condition patch height (required)")
    p.add_argument("--audio-chunk-bytes", type=int, default=512,
                   help="audio: chunk length in bytes = one retrieval/condition unit")
    p.add_argument("--retriever", default="bm25", choices=["bm25"],
                   help="retrieval signal over base chunk bytes (bm25 = syntactic byte k-grams)")
    p.add_argument("--kgram", type=int, default=4, help="byte k-gram size for BM25")
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="Output directory")
    return p


def _load_samples(modality: str, path: str, n):
    if modality == "image":
        from utils.img_utils import load_image_files
        return load_image_files(path, n)
    elif modality == "audio":
        from utils.audio_utils import load_audio_samples
        return load_audio_samples(path, n)


def _chunk_base(
    modality: str,
    samples,
    base_indices,
    unit: dict,
):
    """Chunk the base samples into (byte payload, source sample idx) records.

    Uses the same preprocessors as ``eval_bgpt.py``/``eval_rac_bgpt.py``.
    ``sample_idx`` is the *global* dataset index of the sample a chunk came
    from (for provenance).
    """
    if modality == "image":
        from utils.img_utils import patchify_images_for_compression
        recs = patchify_images_for_compression(
            samples,
            base_indices,
            patch_width=unit["patch_width"],
            patch_height=unit["patch_height"],
        )
        return [(r.data, base_indices[r.sample_idx]) for r in recs], "bmp"
    elif modality == "audio":
        from utils.audio_utils import chunk_audio_for_compression
        recs = chunk_audio_for_compression(
            samples, base_indices,
            audio_chunk_bytes=unit["audio_chunk_bytes"])
        return [(r.data, base_indices[r.sample_idx]) for r in recs], "wav"


def main() -> None:
    args = _build_parser().parse_args()

    if not 0 < args.base_frac < 1:
        raise ValueError(f"--base-frac must be between 0 and 1, got {args.base_frac}")
    if args.modality == "image":
        if args.image_patch_width is None or args.image_patch_height is None:
            raise ValueError(
                "image preparation requires --image-patch-width and "
                "--image-patch-height")
        if args.image_patch_width <= 0 or args.image_patch_height <= 0:
            raise ValueError(
                "image patch dimensions must be positive, got "
                f"{args.image_patch_width}x{args.image_patch_height}")
        unit = {
            "patch_width": args.image_patch_width,
            "patch_height": args.image_patch_height,
        }
    else:
        unit = {"audio_chunk_bytes": args.audio_chunk_bytes}

    samples = _load_samples(args.modality, args.dataset, args.n_samples)
    n = len(samples)
    idx = list(range(n))
    random.Random(args.seed).shuffle(idx)
    n_base = int(round(args.base_frac * n))
    base_sample_indices = sorted(idx[:n_base])
    eval_sample_indices = [i for i in range(n) if i not in set(base_sample_indices)]
    print(f"Samples: {n} | base: {len(base_sample_indices)} | "
          f"eval (held-out): {len(eval_sample_indices)}")

    # Keep only full-length chunks so every base condition has the same length.
    chunk_recs, ext = _chunk_base(
        args.modality,
        samples,
        base_sample_indices,
        unit,
    )
    if not chunk_recs:
        raise ValueError("No base chunks produced; increase --n-samples or --base-frac")

    unit_bytes = max(len(d) for d, _ in chunk_recs)
    full_chunks = [record for record in chunk_recs if len(record[0]) == unit_bytes]
    base_chunks = [
        {"id": i, "sample_idx": sample_idx, "ext": ext, "data": data}
        for i, (data, sample_idx) in enumerate(full_chunks)
    ]
    dropped = len(chunk_recs) - len(base_chunks)
    chunk_size = len(base_chunks[0]["data"])
    print(f"Base chunks (retrieval units): {len(base_chunks)} of {unit_bytes} B "
          f"({chunk_size} raw byte-tokens) | dropped {dropped} partial chunks")

    os.makedirs(args.out, exist_ok=True)

    # 1. Base chunks (the compressor's conditions) — explicit, like base_chunks.json.
    with open(os.path.join(args.out, "base_chunks.pkl"), "wb") as f:
        pickle.dump(base_chunks, f)

    # 2. Held-out WAV bytes or image paths, chunked + retrieved live at eval.
    eval_samples = [samples[i] for i in eval_sample_indices]
    with open(os.path.join(args.out, "eval_samples.pkl"), "wb") as f:
        pickle.dump(eval_samples, f)
    print(f"Held-out eval samples: {len(eval_samples)} -> eval_samples.pkl")

    # 3. Byte retriever over the base chunk bytes.
    print(f"Indexing base ({args.retriever}, kgram={args.kgram}) ...")
    retriever = make_bgpt_retriever(signals=args.retriever, kgram=args.kgram, rrf_k=args.rrf_k)
    retriever.build([b["data"] for b in base_chunks])
    retriever.save(os.path.join(args.out, "retriever"))

    meta = {"format_version": RAC_BGPT_DB_FORMAT_VERSION,
            "dataset": args.dataset, "modality": args.modality, "seed": args.seed,
            "base_frac": args.base_frac, "n_samples": args.n_samples,
            "base_sample_indices": base_sample_indices,
            "unit": unit,
            "unit_bytes": unit_bytes, "chunk_size": chunk_size,
            "patch_size": PATCH_SIZE, "ext": ext,
            "signals": args.retriever, "kgram": args.kgram}
    meta["payload_format"] = (
        "pcm_u8" if args.modality == "audio" else "bmp_rgb24_pixel_array")
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"Database -> {args.out}")


if __name__ == "__main__":
    main()
