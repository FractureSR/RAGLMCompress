"""Build the RAC retrieval **database** for bGPT (audio / image bytes).

The byte-domain counterpart of ``prepare_rac_data_llm.py``: fix a slice of an
audio/image dataset as the base corpus, chunk it into fixed-size **byte** chunks
(one bGPT compression/retrieval unit — an image patch or an audio chunk), and
index them by byte similarity. ``eval_rac_bgpt.py`` then chunks + retrieves the
held-out eval samples *live*, exactly the way ``eval_bgpt.py`` chunks its input.

Base chunks are kept only at their full (modal) byte length so every retrieved
condition shares one patch-aligned prefix length — the byte analog of the text
prep's ``align_last_window`` (partial trailing chunks are dropped from the base).

Outputs under ``--out`` (a self-contained database, mirroring the text prep):
  base_chunks.pkl  [{id, sample_idx, ext, data (bytes)}]  retrieval units /
                   conditions; the eval reads ``base_tokens`` from ``data``.
  eval_samples.pkl the held-out eval sample payloads (the byte analog of
                   ``eval_docs.jsonl``); the eval chunks + retrieves them live.
  retriever/       saved BM25 byte index.
  meta.json        {dataset, modality, seed, base_frac, n_samples,
                    base_sample_indices, unit (patch_px|chunk_ms), chunk_bytes,
                    chunk_size (token length, a multiple of patch_size),
                    patch_size, ext, signals, kgram}.

Example
-------
    python utils/prepare_rac_data_bgpt.py \\
        --dataset datasets/clic2024/bmp --modality image --n-samples 400 \\
        --base-frac 0.5 --patch-px 16 --patch-size 16 --retriever bm25 \\
        --out results/rac_img_db
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.bgpt_codec_utils import bytes_to_padded_tokens, make_bgpt_retriever


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build the bGPT RAC retrieval database")
    p.add_argument("--dataset", required=True, help="Audio/image dataset path (registered loader)")
    p.add_argument("--modality", required=True, choices=["image", "audio"])
    p.add_argument("--n-samples", type=int, default=None, help="Samples to load from the dataset")
    p.add_argument("--base-frac", type=float, default=0.5,
                   help="Fraction of samples fixed as the base/database (rest are eval)")
    p.add_argument("--patch-px", type=int, default=16,
                   help="image: pixel patch size = one retrieval/compression unit")
    p.add_argument("--chunk-ms", type=int, default=250,
                   help="audio: chunk duration ms = one retrieval/compression unit")
    p.add_argument("--patch-size", type=int, default=16,
                   help="bGPT byte-patch size (must match the model)")
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


def _chunk_base(modality: str, samples, base_indices, patch_px: int, chunk_ms: int):
    """Chunk the base samples into (byte payload, source sample idx) records.

    Reuses the exact preprocessors ``eval_bgpt.py``/``eval_rac_bgpt.py`` use, so
    base and eval units are produced identically. ``sample_idx`` is the *global*
    dataset index of the sample a chunk came from (for provenance).
    """
    if modality == "image":
        from utils.img_utils import patchify_images_for_compression
        recs = patchify_images_for_compression(samples, base_indices, patch_size=patch_px)
        return [(r.patch.data, base_indices[r.sample_idx]) for r in recs], "bmp"
    elif modality == "audio":
        from utils.audio_utils import chunk_audio_for_compression
        recs = chunk_audio_for_compression(samples, base_indices, chunk_ms=chunk_ms)
        return [(r.data, base_indices[r.sample_idx]) for r in recs], "wav"


def main() -> None:
    args = _build_parser().parse_args()

    samples = _load_samples(args.modality, args.dataset, args.n_samples)
    n = len(samples)
    idx = list(range(n))
    random.Random(args.seed).shuffle(idx)
    n_base = int(round(args.base_frac * n))
    base_sample_indices = sorted(idx[:n_base])
    eval_sample_indices = [i for i in range(n) if i not in set(base_sample_indices)]
    print(f"Samples: {n} | base: {len(base_sample_indices)} | "
          f"eval (held-out): {len(eval_sample_indices)}")

    # Chunk the base samples into byte units, then keep only full-length chunks so
    # every base condition shares one patch-aligned prefix length (drop partials).
    chunk_recs, ext = _chunk_base(args.modality, samples, base_sample_indices,
                                  args.patch_px, args.chunk_ms)
    if not chunk_recs:
        raise ValueError("No base chunks produced; increase --n-samples or --base-frac")

    chunk_bytes = max(len(d) for d, _ in chunk_recs)
    base_chunks = [
        {"id": i, "sample_idx": sidx, "ext": ext, "data": bytes(data)}
        for i, (data, sidx) in enumerate(d for d in chunk_recs if len(d[0]) == chunk_bytes)
    ]
    dropped = len(chunk_recs) - len(base_chunks)
    chunk_size = len(bytes_to_padded_tokens(base_chunks[0]["data"], args.patch_size))
    print(f"Base chunks (retrieval units): {len(base_chunks)} of {chunk_bytes} B "
          f"({chunk_size} byte-tokens) | dropped {dropped} partial chunks")

    os.makedirs(args.out, exist_ok=True)

    # 1. Base chunks (the compressor's conditions) — explicit, like base_chunks.json.
    with open(os.path.join(args.out, "base_chunks.pkl"), "wb") as f:
        pickle.dump(base_chunks, f)

    # 2. Held-out eval samples (chunked + retrieved live at eval) — like eval_docs.jsonl.
    eval_samples = [samples[i] for i in eval_sample_indices]
    with open(os.path.join(args.out, "eval_samples.pkl"), "wb") as f:
        pickle.dump(eval_samples, f)
    print(f"Held-out eval samples: {len(eval_samples)} -> eval_samples.pkl")

    # 3. Byte retriever over the base chunk bytes.
    print(f"Indexing base ({args.retriever}, kgram={args.kgram}) ...")
    retriever = make_bgpt_retriever(signals=args.retriever, kgram=args.kgram, rrf_k=args.rrf_k)
    retriever.build([b["data"] for b in base_chunks])
    retriever.save(os.path.join(args.out, "retriever"))

    meta = {"dataset": args.dataset, "modality": args.modality, "seed": args.seed,
            "base_frac": args.base_frac, "n_samples": args.n_samples,
            "base_sample_indices": base_sample_indices,
            "unit": {"patch_px": args.patch_px} if args.modality == "image"
                    else {"chunk_ms": args.chunk_ms},
            "chunk_bytes": chunk_bytes, "chunk_size": chunk_size,
            "patch_size": args.patch_size, "ext": ext,
            "signals": args.retriever, "kgram": args.kgram}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"Database -> {args.out}")


if __name__ == "__main__":
    main()
