"""bGPT compression evaluation — image and audio.

Usage examples
--------------
# Image (clic2024 BMP files)
python evaluation/eval_bgpt.py \\
    --modality image \\
    --dataset  datasets/clic2024/bmp \\
    --model    pretrained/bgpt/weights-image.pth \\
    --n-samples 100 \\
    --device cuda:0 \\
    --output results/bgpt_image.csv

# Audio (peoples_speech — pass the dataset directory; pd.read_parquet reads all parquets inside)
python evaluation/eval_bgpt.py \\
    --modality audio \\
    --dataset  datasets/peoples_speech \\
    --model    pretrained/bgpt/weights-audio.pth \\
    --n-samples 50 \\
    --device cuda:0,cuda:1 \\
    --save-compressed results/compressed \\
    --output results/bgpt_audio.csv

# Skip decompression (compression-only benchmark)
python evaluation/eval_bgpt.py --modality image ... --no-decompress

# New dataset: register a loader in utils/img_utils.py or utils/audio_utils.py
"""
from __future__ import annotations
from transformers import GPT2Config
import argparse
import os
import sys
import time
from collections import defaultdict
from typing import List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio_utils import (
    load_audio_samples,
    chunk_audio_for_compression,
)
from utils.img_utils import (
    load_image_files, reassemble_image_patches, ImagePatch,
    patchify_images_for_compression,
)
from utils.eval_utils import (
    EvalResult, EvalStats, MemoryTracker,
    auto_batch_size,
    parse_devices, run_multi_gpu,
    save_csv, save_compressed, save_decompressed,
)
from utils.bgpt_codec_utils import pad_input_for_bgpt, bytes_to_padded_tokens
from compression.bgpt_compressor import BGPTCompressor
from bgpt.utils import bGPTLMHeadModel
from bgpt.config import BYTE_NUM_LAYERS, HIDDEN_SIZE, PATCH_NUM_LAYERS, PATCH_SIZE
PATCH_LENGTH = 512


# ---------------------------------------------------------------------------
# Model loader (module-level → picklable for mp.spawn)
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str, device: torch.device) -> bGPTLMHeadModel:
    patch_cfg = GPT2Config(
        num_hidden_layers=PATCH_NUM_LAYERS, max_length=PATCH_LENGTH,
        max_position_embeddings=PATCH_LENGTH, hidden_size=HIDDEN_SIZE,
        n_head=HIDDEN_SIZE // 64, vocab_size=1,
    )
    byte_cfg = GPT2Config(
        num_hidden_layers=BYTE_NUM_LAYERS, max_length=PATCH_SIZE + 1,
        max_position_embeddings=PATCH_SIZE + 1, hidden_size=HIDDEN_SIZE,
        n_head=HIDDEN_SIZE // 64, vocab_size=257,
    )
    m = bGPTLMHeadModel(patch_cfg, byte_cfg)
    ckpt = torch.load(checkpoint_path, map_location=device)
    m.load_state_dict(ckpt["model"], strict=False)
    m.to(device).eval()
    return m


# ---------------------------------------------------------------------------
# Workers (module-level → picklable for mp.spawn)
# ---------------------------------------------------------------------------

def _bgpt_probe_factory(model, device):
    """Return a probe_fn compatible with auto_batch_size for bGPT forward passes."""
    def _probe(batch_sz: int, seq_len: int) -> None:
        payload = [0] * seq_len
        ext = [ord("b"), ord("m"), ord("p")]
        padded = pad_input_for_bgpt(
            [payload] * batch_sz, [ext] * batch_sz,
            device=device, patch_size=PATCH_SIZE,
        )
        with torch.inference_mode():
            model(patches=padded["patches"], masks=padded["masks"])
    return _probe


def _image_worker(
    device: torch.device,
    indices: List[int],
    model_path: str,
    bmp_files: List[str],
    no_decomp: bool,
    save_comp_dir: Optional[str],
    save_decomp_dir: Optional[str],
    image_patch_px: int,
    max_decode_tokens: Optional[int] = None,
) -> List[EvalResult]:
    import tqdm

    model = _load_model(model_path, device)
    comp = BGPTCompressor(model, patch_size=PATCH_SIZE, device=device)

    # ── 1. Preprocessing: patchify all images ─────────────────────────────────
    all_patches = patchify_images_for_compression(
        bmp_files, indices, image_patch_px)

    # ── 2. Auto-select batch size ─────────────────────────────────────────────
    # All patches are the same byte size; use the first as the representative length.
    # have to use bytes_to_padded_tokens to pad the sequence to a multiple of the PATCH_SIZE
    rep_lens = [len(bytes_to_padded_tokens(
        all_patches[0].patch.data, PATCH_SIZE))] if all_patches else []
    batch_size = auto_batch_size(
        _bgpt_probe_factory(model, device), device, rep_lens,
        max_batch=512, n_samples=len(all_patches), verbose=True,
    )
    print(f"  [{device}] batch_size={batch_size}  images={len(indices)}  patches={len(all_patches)}")

    # ── 3. Compress (and optionally decompress) all patches ───────────────────
    # flat_idx → (cd, roundtrip_ok, compress_s, decompress_s, mem, rec_bytes_or_None)
    patch_results = {}

    def _process_patch_batch(batch_positions: List[int]) -> None:
        batch = [all_patches[pos] for pos in batch_positions]
        patches_b = [r.patch for r in batch]
        sids_b = [r.sample_idx for r in batch]

        with MemoryTracker(device) as mem:
            t0 = time.time()
            cds = comp.compress_batch([(p.data, "bmp") for p in patches_b])
            compress_s = time.time() - t0
        per_c_s = compress_s / len(batch)

        recs = None
        per_d_s = -1.0
        if not no_decomp:
            t0 = time.time()
            recs = comp.decompress_batch(cds, show_progress=True,
                                         max_tokens=max_decode_tokens)
            per_d_s = (time.time() - t0) / len(batch)

        for local_i, (patch, sid, cd) in enumerate(zip(patches_b, sids_b, cds)):
            rt_ok = -1
            rec_data = None
            if recs is not None:
                rec_data = recs[local_i]
                rt_ok = -1 if max_decode_tokens else int(rec_data == patch.data)
            if save_comp_dir:
                save_compressed(cd.compressed_bytes, cd.metadata, cd.original_length,
                                save_comp_dir, f"image{indices[sid]:06d}_p{patch.index:04d}")
            patch_results[batch_positions[local_i]] = (
                cd, rt_ok, per_c_s, per_d_s, mem, rec_data)

    pbar = tqdm.tqdm(total=len(all_patches), desc=f"Image [{device}]",
                     position=device.index or 0)
    for cursor in range(0, len(all_patches), batch_size):
        batch_pos = list(
            range(cursor, min(cursor + batch_size, len(all_patches))))
        _process_patch_batch(batch_pos)
        pbar.update(len(batch_pos))
    pbar.close()

    # ── 4. Aggregate patch results per original image ─────────────────────────
    sample_patch_pos: defaultdict = defaultdict(list)
    for flat_idx, rec in enumerate(all_patches):
        sample_patch_pos[rec.sample_idx].append(flat_idx)

    results = []
    for local_idx, global_idx in enumerate(indices):
        sample_id = f"image{global_idx:06d}"
        pos_list = sample_patch_pos[local_idx]
        recs = [all_patches[pos] for pos in pos_list]
        data = [patch_results[pos] for pos in pos_list]

        orig_bytes = sum(len(r.patch.data) for r in recs)
        comp_bytes = sum(d[0].compressed_length for d in data)
        rt_ok = (1 if all(d[1] == 1 for d in data) else
                 -1 if all(d[1] == -1 for d in data) else 0)
        c_s = sum(d[2] for d in data)
        d_s_vals = [d[3] for d in data if d[3] >= 0]
        d_s = sum(d_s_vals) if d_s_vals else -1.0
        mem = data[0][4]

        if save_decomp_dir and all(d[5] is not None for d in data):
            rec_patches = [
                ImagePatch(index=r.patch.index, x=r.patch.x, y=r.patch.y,
                           width=r.patch.width, height=r.patch.height, data=d[5])
                for r, d in zip(recs, data)
            ]
            img = reassemble_image_patches(rec_patches, recs[0].meta)
            os.makedirs(save_decomp_dir, exist_ok=True)
            img.save(os.path.join(save_decomp_dir, f"{sample_id}.bmp"))

        results.append(EvalResult(
            sample_id=sample_id,
            original_bytes=orig_bytes, compressed_bytes=comp_bytes,
            bpb=comp_bytes * 8 / max(orig_bytes, 1),
            ratio=orig_bytes / max(comp_bytes, 1),
            compress_s=c_s, decompress_s=d_s,
            peak_gpu_mb=mem.peak_gpu_mb, peak_ram_mb=mem.peak_ram_mb,
            roundtrip_ok=rt_ok,
        ))
    return results


def _audio_worker(
    device: torch.device,
    indices: List[int],
    model_path: str,
    samples: list,
    no_decomp: bool,
    save_comp_dir: Optional[str],
    save_decomp_dir: Optional[str],
    chunk_ms: int,
    max_decode_tokens: Optional[int] = None,
) -> List[EvalResult]:
    import tqdm

    model = _load_model(model_path, device)
    comp = BGPTCompressor(model, patch_size=PATCH_SIZE, device=device)

    # ── 1. Preprocessing: chunk all audio clips ───────────────────────────────
    all_chunks = chunk_audio_for_compression(samples, indices, chunk_ms)

    # ── 2. Auto-select batch size ─────────────────────────────────────────────
    # All chunks are the same byte size; use the first as the representative length.
    rep_lens = [len(bytes_to_padded_tokens(
        all_chunks[0].data, PATCH_SIZE))] if all_chunks else []
    batch_size = auto_batch_size(
        _bgpt_probe_factory(model, device), device, rep_lens,
        max_batch=512, n_samples=len(all_chunks), verbose=True,
    )
    print(
        f"  [{device}] batch_size={batch_size}  clips={len(indices)}  chunks={len(all_chunks)}")

    # ── 3. Compress (and optionally decompress) all chunks ────────────────────
    # flat_idx → (cd, roundtrip_ok, compress_s, decompress_s, mem, rec_bytes_or_None)
    chunk_results = {}

    def _process_chunk_batch(batch_positions: List[int]) -> None:
        batch = [all_chunks[pos] for pos in batch_positions]
        chunks_b = [r.data for r in batch]
        sids_b = [r.sample_idx for r in batch]
        cidxs_b = [r.chunk_idx for r in batch]

        with MemoryTracker(device) as mem:
            t0 = time.time()
            cds = comp.compress_batch([(c, "wav") for c in chunks_b])
            compress_s = time.time() - t0
        per_c_s = compress_s / len(batch)

        recs = None
        per_d_s = -1.0
        if not no_decomp:
            t0 = time.time()
            recs = comp.decompress_batch(cds, show_progress=True,
                                         max_tokens=max_decode_tokens)
            per_d_s = (time.time() - t0) / len(batch)

        for local_i, (chunk, sid, cidx, cd) in enumerate(zip(chunks_b, sids_b, cidxs_b, cds)):
            rt_ok = -1
            rec_data = None
            if recs is not None:
                rec_data = recs[local_i]
                rt_ok = -1 if max_decode_tokens else int(rec_data == chunk)
            if save_comp_dir:
                save_compressed(cd.compressed_bytes, cd.metadata, cd.original_length,
                                save_comp_dir, f"audio{indices[sid]:06d}_chunk{cidx:04d}")
            chunk_results[batch_positions[local_i]] = (
                cd, rt_ok, per_c_s, per_d_s, mem, rec_data)

    pbar = tqdm.tqdm(total=len(all_chunks), desc=f"Audio [{device}]",
                     position=device.index or 0)
    for cursor in range(0, len(all_chunks), batch_size):
        batch_pos = list(
            range(cursor, min(cursor + batch_size, len(all_chunks))))
        _process_chunk_batch(batch_pos)
        pbar.update(len(batch_pos))
    pbar.close()

    # ── 4. Aggregate chunk results per original clip ───────────────────────────
    sample_chunk_pos: defaultdict = defaultdict(list)
    for flat_idx, rec in enumerate(all_chunks):
        sample_chunk_pos[rec.sample_idx].append(flat_idx)

    results = []
    for local_idx, global_idx in enumerate(indices):
        sample_id = f"audio{global_idx:06d}"
        data = [chunk_results[pos] for pos in sample_chunk_pos[local_idx]]

        orig_bytes = sum(len(all_chunks[pos].data)
                         for pos in sample_chunk_pos[local_idx])
        comp_bytes = sum(d[0].compressed_length for d in data)
        rt_ok = (1 if all(d[1] == 1 for d in data) else
                 -1 if all(d[1] == -1 for d in data) else 0)
        c_s = sum(d[2] for d in data)
        d_s_vals = [d[3] for d in data if d[3] >= 0]
        d_s = sum(d_s_vals) if d_s_vals else -1.0
        mem = data[0][4]

        if save_decomp_dir:
            for chunk_idx, d in enumerate(data):
                if d[5] is not None:
                    save_decompressed(d[5], save_decomp_dir,
                                      f"{sample_id}_chunk{chunk_idx:04d}", "wav")

        results.append(EvalResult(
            sample_id=sample_id,
            original_bytes=orig_bytes, compressed_bytes=comp_bytes,
            bpb=comp_bytes * 8 / max(orig_bytes, 1),
            ratio=orig_bytes / max(comp_bytes, 1),
            compress_s=c_s, decompress_s=d_s,
            peak_gpu_mb=mem.peak_gpu_mb, peak_ram_mb=mem.peak_ram_mb,
            roundtrip_ok=rt_ok,
        ))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="bGPT compression evaluation")
    p.add_argument("--modality",   required=True, choices=["image", "audio"])
    p.add_argument("--dataset",    required=True,
                   help="Dataset path (must match a registered image/audio loader name)")
    p.add_argument("--model",      required=True,
                   help="Path to bGPT checkpoint (.pth)")
    p.add_argument("--n-samples",  type=int, default=None,
                   help="Max samples (default: all)")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Comma-separated devices, e.g. cuda:0,cuda:1")
    p.add_argument("--save-compressed",   default=None, metavar="DIR")
    p.add_argument("--save-decompressed", default=None, metavar="DIR")
    p.add_argument("--output",     default=None, metavar="CSV",
                   help="Path to write results CSV")
    p.add_argument("--no-decompress", action="store_true",
                   help="Skip decompression")
    p.add_argument("--max-decode-tokens", type=int, default=None,
                   help="Decode only the first N tokens per sample (round-trip check skipped)")
    p.add_argument("--image-patch-px", type=int, default=32,
                   help="Pixel patch size (image, default: 32)")
    p.add_argument("--chunk-ms",       type=int, default=1000,
                   help="Audio chunk duration ms (default: 1000)")
    p.add_argument("--tmp-dir", default="tmp",
                   help="Directory for inter-process temp files (default: tmp)")
    return p


def main():
    args = _build_parser().parse_args()
    args.model = os.path.normpath(args.model)
    devices = parse_devices(args.device)

    if args.modality == "image":
        samples = load_image_files(args.dataset, args.n_samples)
        print(f"Loaded {len(samples)} image files | devices: {devices}")
        results = run_multi_gpu(
            _image_worker, list(range(len(samples))), devices,
            fn_kwargs=dict(
                model_path=args.model, bmp_files=samples,
                no_decomp=args.no_decompress,
                save_comp_dir=args.save_compressed,
                save_decomp_dir=args.save_decompressed,
                image_patch_px=args.image_patch_px,
                max_decode_tokens=args.max_decode_tokens,
            ),
            tmp_prefix=os.path.join(args.tmp_dir, '_eval_worker'),
        )
    else:  # audio
        samples = load_audio_samples(args.dataset, args.n_samples)
        print(f"Loaded {len(samples)} audio samples | devices: {devices}")
        results = run_multi_gpu(
            _audio_worker, list(range(len(samples))), devices,
            fn_kwargs=dict(
                model_path=args.model, samples=samples,
                no_decomp=args.no_decompress,
                save_comp_dir=args.save_compressed,
                save_decomp_dir=args.save_decompressed,
                chunk_ms=args.chunk_ms,
                max_decode_tokens=args.max_decode_tokens,
            ),
            tmp_prefix=os.path.join(args.tmp_dir, '_eval_worker'),
        )

    stats = EvalStats()
    for r in results:
        stats.update(r)
    stats.print_summary(label=f"bGPT {args.modality}")

    if args.output:
        save_csv(results, args.output)


if __name__ == "__main__":
    main()
