"""Oracle RAC compression evaluation — audio / image (bGPT).

The byte-domain counterpart of ``eval_rac_llm.py``: it is to ``eval_bgpt.py`` what
``eval_rac_llm.py`` is to ``eval_llm.py``. It loads a database built by
``prepare_rac_data_bgpt.py``, chunks the held-out eval samples into byte payload
units, retrieves top-m similar base chunks per unit, and runs
``RACBGPTCompressor``'s oracle (keep the best retrieved byte prefix only if it
beats its transmitted-id cost, else no condition). The chosen base ids travel as
side info; their bit cost is added for honest bpb/ratio, just like the text eval.

Payload vs condition size (the text analogy): as in ``eval_rac_llm.py`` — where
the data piece is ``ctx_len - max_ctx`` LM tokens while conditions keep the
database ``chunk_size`` — the audio payload window defaults to the byte context
left after the prefix budget (override with ``--audio-chunk-bytes``), while each
retrieved condition stays one database chunk. Retrieval queries with the whole
payload; the chosen conditions are prepended as byte prefixes and the combined
sequence is patchified/padded to bGPT's format by the compressor. Image payloads
remain single database-sized patches (a 2D unit has no window to grow).

Usage
-----
    python evaluation/eval_rac_bgpt.py --database results/rac_img_db \\
        --model pretrained/bgpt/weights-image.pth --m 16 --n-samples 50 \\
        --device cuda:0
"""
from __future__ import annotations

from transformers import GPT2Config
import argparse
import gc
import json
import os
import pickle
import sys
import tempfile
import time
from collections import defaultdict
from typing import Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.bgpt_compressor import BGPTCompressor
from compression.rac_bgpt_compressor import RACBGPTCompressor
from compression.rac_index import CalibratedIndexCoder, FixedIndexCoder, load_index_coder
from utils.bgpt_codec_utils import (
    bytes_to_padded_tokens, make_bgpt_retriever, pad_input_for_bgpt,
)
from utils.eval_utils import (
    EvalResult, EvalStats,
    auto_batch_size,
    parse_devices, run_multi_gpu,
    save_csv,
)
from bgpt.utils import bGPTLMHeadModel
from bgpt.config import BYTE_NUM_LAYERS, HIDDEN_SIZE, PATCH_NUM_LAYERS, PATCH_SIZE

PATCH_LENGTH = 512   # patch-decoder context (max patches per forward); see eval_bgpt


# ---------------------------------------------------------------------------
# Model / database / sample loading
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str, device: torch.device) -> bGPTLMHeadModel:
    """Build the bGPT model and load a checkpoint (mirrors eval_bgpt._load_model)."""
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
    return m.to(device).eval()


def _load_database(database_dir: str, signals: str, kgram: int, patch_size: int):
    """Load the base chunks (→ base_tokens) and the byte retriever."""
    with open(os.path.join(database_dir, "base_chunks.pkl"), "rb") as f:
        base = pickle.load(f)
    base_tokens = [bytes_to_padded_tokens(b["data"], patch_size) for b in base]
    if not base_tokens:
        raise ValueError(f"No base chunks found in {database_dir}/base_chunks.pkl")
    retriever = make_bgpt_retriever(signals=signals, kgram=kgram)
    retriever.load(os.path.join(database_dir, "retriever"))
    return base_tokens, retriever


def _load_samples(modality: str, path: str, n: Optional[int]):
    """Load eval samples through the modality's registered dataset loaders."""
    if modality == "image":
        from utils.img_utils import load_image_files
        return load_image_files(path, n)
    elif modality == "audio":
        from utils.audio_utils import load_audio_samples
        return load_audio_samples(path, n)
    raise ValueError(f"unsupported modality: {modality!r}")


def _chunk_eval_samples(modality: str, samples, indices: List[int], unit: dict):
    """Chunk a shard of eval samples into (data, ext, sample_local_idx) byte units.

    ``unit`` sizes the eval-side payload window: for image it is the database
    patch size, for audio it is the (decoupled, usually much larger) payload
    byte window computed in ``main``. sample_local_idx indexes into ``indices``
    (the shard), for per-sample rollup.
    """
    if modality == "image":
        from utils.img_utils import patchify_images_for_compression
        recs = patchify_images_for_compression(samples, indices, patch_size=unit["patch_px"])
        return [(r.patch.data, "bmp", r.sample_idx) for r in recs]
    elif modality == "audio":
        from utils.audio_utils import chunk_audio_for_compression
        recs = chunk_audio_for_compression(
            samples, indices, audio_chunk_bytes=unit["audio_chunk_bytes"])
        return [(r.data, "wav", r.sample_idx) for r in recs]


def _prefix_metrics(sample_data: List[tuple]) -> dict:
    hist: Dict[int, int] = defaultdict(int)
    gain_sums: Dict[int, float] = defaultdict(float)
    net_gain_sums: Dict[int, float] = defaultdict(float)
    id_bit_sums: Dict[int, float] = defaultdict(float)
    gain_counts: Dict[int, int] = defaultdict(int)

    for cd, *_ in sample_data:
        n_prefix = len(cd.metadata.get("ctx_ids", []))
        hist[n_prefix] += 1
        gains = cd.metadata.get("ctx_gain_bits", [])
        net_gains = cd.metadata.get("ctx_net_gain_bits", [])
        id_bits = cd.metadata.get("ctx_id_bits", [])
        for pos, gain in enumerate(gains, start=1):
            gain_sums[pos] += float(gain)
            net_gain_sums[pos] += float(net_gains[pos - 1])
            id_bit_sums[pos] += float(id_bits[pos - 1])
            gain_counts[pos] += 1

    return dict(
        prefix_hist=dict(hist),
        prefix_gain_sums=dict(gain_sums),
        prefix_net_gain_sums=dict(net_gain_sums),
        prefix_id_bit_sums=dict(id_bit_sums),
        prefix_gain_counts=dict(gain_counts),
    )


def _merge_numeric_dict(dst: Dict[int, float], src: Dict[int, float]) -> None:
    for k, v in src.items():
        dst[int(k)] += v


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _rac_worker(
    device: torch.device,
    indices: List[int],
    model_path: str,
    samples: list,
    database_dir: str,
    modality: str,
    unit: dict,
    signals: str,
    kgram: int,
    m: int,
    cfg: dict,
    index_path: Optional[str],
    no_decomp: bool,
) -> List[dict]:
    model = _load_model(model_path, device)
    bgpt = BGPTCompressor(model, patch_size=cfg["patch_size"], device=device)
    base_tokens, retriever = _load_database(database_dir, signals, kgram, cfg["patch_size"])

    index_coder = load_index_coder(index_path) if index_path else FixedIndexCoder(len(base_tokens))
    rac = RACBGPTCompressor(
        bgpt, base_tokens,
        index_coder=index_coder,
        max_ctx=cfg["max_ctx"],
        margin_bits=cfg["margin_bits"],
        batch_size=cfg["batch_size"] or 1,
        cascade=cfg["cascade"],
        cascade_max_cond=cfg["cascade_max_cond"],
        cascade_nll_thresh=cfg["cascade_nll_thresh"],
        cascade_min_frac=cfg["cascade_min_frac"],
        cascade_top_k=cfg["cascade_top_k"],
        retriever=retriever if cfg["cascade_retriever"] else None,
        chunk_size=cfg["chunk_size"],
        show_progress=True,
        device=device,
    )

    # ── 1. Preprocessing: split samples into compression units ────────────────
    chunks = _chunk_eval_samples(modality, samples, indices, unit)
    segments = [(data, ext) for data, ext, _ in chunks]

    # ── 2. Retrieve top-k candidates for every unit ──────────────────────────
    queries = [data for data, _, _ in chunks]
    cand_lists = (
        [[cid for cid, _ in hits]
         for hits in retriever.retrieve_many(queries, top_k=m)]
        if queries else []
    )

    # ── 3. Auto-select RAC scoring batch size ────────────────────────────────
    ext = segments[0][1] if segments else ("bmp" if modality == "image" else "wav")
    ext_ids = [ord(c) for c in ext][:cfg["patch_size"]] or [0]

    def _probe(batch_size: int, seq_len: int) -> None:
        padded = pad_input_for_bgpt(
            [[0] * seq_len] * batch_size, [ext_ids] * batch_size,
            device=device, patch_size=cfg["patch_size"],
        )
        with torch.inference_mode():
            model(patches=padded["patches"], masks=padded["masks"])

    score_lens = [cfg["payload_bytes"] + cfg["max_ctx"]]
    score_batch_size = cfg["batch_size"] or auto_batch_size(
        _probe, device, score_lens, max_batch=256,
        n_samples=max(1, len(segments) * m), verbose=True,
    )
    rac.batch_size = score_batch_size
    print(f"  [{device}] score_batch_size={score_batch_size}  samples={len(indices)}  "
          f"units={len(segments)}")

    # ── 4. Compress (and optionally decompress) all units ──────────────────────
    effective_score_bs = score_batch_size
    while True:
        try:
            rac.batch_size = effective_score_bs
            t0 = time.time()
            cds = rac.compress_batch(segments, cand_lists)
            compress_s = time.time() - t0
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if effective_score_bs == 1:
                raise
            effective_score_bs = max(1, effective_score_bs // 2)
            print(f"\n  OOM — retrying with score_batch_size={effective_score_bs}")

    per_c_s = compress_s / max(len(chunks), 1)

    recs = None
    per_d_s = -1.0
    if not no_decomp:
        t0 = time.time()
        recs = rac.decompress_batch(cds)
        per_d_s = (time.time() - t0) / max(len(chunks), 1)

    # unit position → (cd, roundtrip_ok, compress_s, decompress_s)
    unit_results: Dict[int, tuple] = {}
    for pos, ((data, _, _), cd) in enumerate(zip(chunks, cds)):
        rt_ok = -1
        if recs is not None:
            rt_ok = int(recs[pos] == data)
        unit_results[pos] = (cd, rt_ok, per_c_s, per_d_s)

    # ── 5. Aggregate unit results per original sample ──────────────────────────
    sample_unit_pos: Dict[int, List[int]] = defaultdict(list)
    for pos, (_, _, sidx) in enumerate(chunks):
        sample_unit_pos[sidx].append(pos)

    results = []
    for local_idx, global_idx in enumerate(indices):
        positions = sample_unit_pos.get(local_idx, [])
        sdata = [unit_results[p] for p in positions]
        sample_id = f"rac_bgpt:{modality}{global_idx:06d}"

        if not sdata:
            results.append({
                "rac": EvalResult(sample_id=sample_id, original_bytes=0,
                                  compressed_bytes=0, bpb=0.0, ratio=0.0,
                                  compress_s=0.0, decompress_s=-1.0,
                                  peak_gpu_mb=-1, peak_ram_mb=-1, roundtrip_ok=-1),
                "used": 0, "ncond": 0, "n_units": 0, **_prefix_metrics([]),
            })
            continue

        orig_b = sum(len(chunks[p][0]) for p in positions)
        comp_b = sum(cd.compressed_length + cd.metadata["index_bits"] / 8.0
                     for cd, *_ in sdata)
        rt_ok = (1 if all(d[1] == 1 for d in sdata) else
                 -1 if all(d[1] == -1 for d in sdata) else 0)
        c_s = sum(d[2] for d in sdata)
        d_s_vals = [d[3] for d in sdata if d[3] >= 0]
        d_s = sum(d_s_vals) if d_s_vals else -1.0
        ncond = sum(len(d[0].metadata["ctx_ids"]) for d in sdata)

        results.append({
            "rac": EvalResult(
                sample_id=sample_id,
                original_bytes=orig_b, compressed_bytes=comp_b,
                bpb=comp_b * 8 / max(orig_b, 1), ratio=orig_b / max(comp_b, 1),
                compress_s=c_s, decompress_s=d_s,
                peak_gpu_mb=-1, peak_ram_mb=-1, roundtrip_ok=rt_ok),
            "used": sum(1 for d in sdata if d[0].metadata["ctx_ids"]),
            "ncond": ncond, "n_units": len(sdata),
            **_prefix_metrics(sdata),
        })
    return results


# ---------------------------------------------------------------------------
# Calibration (static index coder) — mirrors eval_rac_llm._calibrate
# ---------------------------------------------------------------------------

def _calibrate(device, model_path, samples, calib_idx, database_dir, modality, unit,
               signals, kgram, m, cfg, alpha, save_path):
    model = bgpt = base_tokens = retriever = rac = None
    chunks = segments = queries = cand_lists = cds = None
    try:
        model = _load_model(model_path, device)
        bgpt = BGPTCompressor(model, patch_size=cfg["patch_size"], device=device)
        base_tokens, retriever = _load_database(database_dir, signals, kgram, cfg["patch_size"])
        rac = RACBGPTCompressor(
            bgpt, base_tokens,
            max_ctx=cfg["max_ctx"],
            margin_bits=cfg["margin_bits"],
            batch_size=cfg["batch_size"] or 1,
            cascade=cfg["cascade"],
            cascade_max_cond=cfg["cascade_max_cond"],
            cascade_nll_thresh=cfg["cascade_nll_thresh"],
            cascade_min_frac=cfg["cascade_min_frac"],
            cascade_top_k=cfg["cascade_top_k"],
            retriever=retriever if cfg["cascade_retriever"] else None,
            chunk_size=cfg["chunk_size"],
            show_progress=True,
            device=device,
        )

        chunks = _chunk_eval_samples(modality, samples, calib_idx, unit)
        segments = [(data, ext) for data, ext, _ in chunks]
        queries = [data for data, _, _ in chunks]
        cand_lists = (
            [[cid for cid, _ in hits]
             for hits in retriever.retrieve_many(queries, top_k=m)]
            if queries else []
        )

        ext = segments[0][1] if segments else ("bmp" if modality == "image" else "wav")
        ext_ids = [ord(c) for c in ext][:cfg["patch_size"]] or [0]

        def _probe(batch_size: int, seq_len: int) -> None:
            padded = pad_input_for_bgpt(
                [[0] * seq_len] * batch_size, [ext_ids] * batch_size,
                device=device, patch_size=cfg["patch_size"],
            )
            with torch.inference_mode():
                model(patches=padded["patches"], masks=padded["masks"])

        score_lens = [cfg["payload_bytes"] + cfg["max_ctx"]]
        rac.batch_size = cfg["batch_size"] or auto_batch_size(
            _probe, device, score_lens, max_batch=256,
            n_samples=max(1, len(segments) * m), verbose=False,
        )
        effective_score_bs = rac.batch_size
        while True:
            try:
                rac.batch_size = effective_score_bs
                cds = rac.compress_batch(segments, cand_lists)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if effective_score_bs == 1:
                    raise
                effective_score_bs = max(1, effective_score_bs // 2)
                print(f"\n  calibration OOM -- retrying with score_batch_size={effective_score_bs}")
        seqs = [list(cd.metadata["ctx_ids"]) for cd in cds]
        CalibratedIndexCoder.calibrate(seqs, len(base_tokens), alpha=alpha).save(save_path)
        print(f"  calibrated index on {len(seqs)} units "
              f"(n_base={len(base_tokens)}) -> {save_path}")
    finally:
        del cds, cand_lists, queries, segments, chunks
        del rac, retriever, base_tokens, bgpt, model
        gc.collect()
        if torch.cuda.is_available() and device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)


def _calibrate_mp_worker(rank, args):
    del rank
    _calibrate(*args)


def _calibrate_isolated(*args):
    device = args[0]
    if device.type != "cuda":
        _calibrate(*args)
        return
    import torch.multiprocessing as mp
    mp.spawn(_calibrate_mp_worker, args=(args,), nprocs=1, join=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Oracle RAC compression evaluation — bGPT")
    p.add_argument("--database", required=True,
                   help="prepare_rac_data_bgpt --out dir (base_chunks + eval_samples + retriever + meta)")
    p.add_argument("--model", required=True, help="bGPT checkpoint (.pth)")
    p.add_argument("--dataset", default=None,
                   help=("eval corpus override: image dataset path or preprocessed "
                         "WAV directory (default: database eval_samples.pkl)"))
    p.add_argument("--n-samples", type=int, default=None,
                   help="cap on eval samples (default: all held-out samples)")
    p.add_argument("--m", type=int, default=16, help="top-k candidates tried per unit")
    p.add_argument("--audio-chunk-bytes", type=int, default=None,
                   help=("audio payload bytes per compression unit (default: fill "
                         "the byte context left after the prefix budget, the "
                         "analog of eval_rac_llm's ctx_len - max_ctx)"))
    p.add_argument("--max-ctx", type=int, default=None,
                   help="total prefix byte-token budget (default: chunk_size * max conditions)")
    p.add_argument("--margin-bits", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=None,
                   help="candidates scored per bGPT forward (default: auto-probe)")
    p.add_argument("--cascade", action="store_true")
    p.add_argument("--cascade-max-cond", type=int, default=2)
    p.add_argument("--cascade-nll-thresh", type=float, default=4.0)
    p.add_argument("--cascade-min-frac", type=float, default=0.05)
    p.add_argument("--cascade-top-k", type=int, default=16)
    p.add_argument("--cascade-retriever", action="store_true",
                   help="re-retrieve the next condition from high-entropy residual")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--calib-samples", type=int, default=20)
    p.add_argument("--calib-alpha", type=float, default=0.5)
    p.add_argument("--save-index", default=None, metavar="JSON")
    p.add_argument("--no-decompress", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Comma-separated devices, e.g. cuda:0,cuda:1")
    p.add_argument("--tmp-dir", default="tmp")
    p.add_argument("--output", default=None, metavar="CSV")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    args.model = os.path.normpath(args.model)
    devices = parse_devices(args.device)

    with open(os.path.join(args.database, "meta.json")) as f:
        meta = json.load(f)

    modality = meta["modality"]
    unit = meta["unit"]
    patch_size = meta["patch_size"]
    chunk_size = meta["chunk_size"]            # condition token length (multiple of patch_size)
    max_cond = args.cascade_max_cond if args.cascade else 1
    max_ctx = args.max_ctx if args.max_ctx is not None else chunk_size * max_cond

    # Payload window, decoupled from the condition size (the text analogy:
    # eval_rac_llm's data piece is ctx_len - max_ctx while conditions keep the
    # database chunk_size). byte_ctx is the byte budget once the ext and
    # trailing pad patches are set aside; prefix + payload must fit in it.
    # (Both get patch-padded, but max_ctx and byte_ctx are patch multiples, so
    # the byte-level check is exact.)
    byte_ctx = (PATCH_LENGTH - 2) * patch_size
    if modality == "audio":
        payload_bytes = args.audio_chunk_bytes or byte_ctx - max_ctx
        unit = {"audio_chunk_bytes": payload_bytes}
    else:
        payload_bytes = chunk_size
    if payload_bytes <= 0 or payload_bytes + max_ctx > byte_ctx:
        raise ValueError(
            f"payload {payload_bytes} B + prefix budget {max_ctx} B exceeds the "
            f"bGPT byte context {byte_ctx} B ({PATCH_LENGTH} patches minus ext "
            f"and trailing pad). Lower --audio-chunk-bytes / --cascade-max-cond, "
            f"or rebuild the database with smaller units.")

    # Eval set = the held-out samples persisted by prepare_rac_data_bgpt
    # (self-contained, no re-load/complement), or a fresh --dataset override.
    n_calib = args.calib_samples if args.calibrate else 0
    n_load = args.n_samples + n_calib if args.n_samples else None
    eval_path = args.dataset or os.path.join(args.database, "eval_samples.pkl")
    samples = _load_samples(modality, eval_path, n_load)

    all_idx = list(range(len(samples)))
    if n_calib >= len(all_idx) and n_calib > 0:
        raise ValueError(
            f"--calib-samples={n_calib} leaves no eval samples; "
            f"loaded only {len(all_idx)} samples")
    if n_calib:
        # the latter part of the eval set is used for calibration, the former for testing
        calib_idx = all_idx[-n_calib:]
        test_idx = all_idx[:-n_calib]
    else:
        calib_idx = []
        test_idx = all_idx
    source = eval_path
    print(f"Loaded {len(samples)} held-out {modality} samples from {source} | "
          f"eval {len(test_idx)} | payload {payload_bytes} B | condition "
          f"{chunk_size} tok x {max_cond} (prefix budget {max_ctx}) | "
          f"m {args.m} | devices: {devices}")

    cfg = dict(
        patch_size=patch_size, chunk_size=chunk_size, max_ctx=max_ctx,
        payload_bytes=payload_bytes,
        margin_bits=args.margin_bits, batch_size=args.batch_size,
        cascade=args.cascade, cascade_max_cond=args.cascade_max_cond,
        cascade_nll_thresh=args.cascade_nll_thresh,
        cascade_min_frac=args.cascade_min_frac, cascade_top_k=args.cascade_top_k,
        cascade_retriever=args.cascade_retriever,
    )

    index_path = None
    if args.calibrate:
        index_path = args.save_index or os.path.join(tempfile.mkdtemp(), "index.json")
        print(f"Calibrating index coder on {len(calib_idx)} held-out samples ...")
        _calibrate_isolated(
            devices[0], args.model, samples, calib_idx, args.database, modality, unit,
            meta["signals"], meta["kgram"], args.m, cfg, args.calib_alpha, index_path,
        )

    results = run_multi_gpu(
        _rac_worker, test_idx, devices,
        fn_kwargs=dict(
            model_path=args.model, samples=samples, database_dir=args.database,
            modality=modality, unit=unit, signals=meta["signals"], kgram=meta["kgram"],
            m=args.m, cfg=cfg, index_path=index_path, no_decomp=args.no_decompress,
        ),
        tmp_prefix=os.path.join(args.tmp_dir, "_eval_rac_bgpt"),
    )

    stats = EvalStats()
    rows: List[EvalResult] = []
    used = ncond = n_units = 0
    prefix_hist: Dict[int, int] = defaultdict(int)
    prefix_gain_sums: Dict[int, float] = defaultdict(float)
    prefix_net_gain_sums: Dict[int, float] = defaultdict(float)
    prefix_id_bit_sums: Dict[int, float] = defaultdict(float)
    prefix_gain_counts: Dict[int, int] = defaultdict(int)
    for r in results:
        stats.update(r["rac"])
        rows.append(r["rac"])
        used += r["used"]
        ncond += r["ncond"]
        n_units += r["n_units"]
        _merge_numeric_dict(prefix_hist, r["prefix_hist"])
        _merge_numeric_dict(prefix_gain_sums, r["prefix_gain_sums"])
        _merge_numeric_dict(prefix_net_gain_sums, r["prefix_net_gain_sums"])
        _merge_numeric_dict(prefix_id_bit_sums, r["prefix_id_bit_sums"])
        _merge_numeric_dict(prefix_gain_counts, r["prefix_gain_counts"])

    stats.print_summary(label=f"rac-bgpt {modality} (oracle)")
    print(f"\n  condition used on {100 * used / max(n_units, 1):.1f}% "
          f"of {n_units} units | {ncond / max(n_units, 1):.3f} cond/unit")
    if n_units:
        print("\n  prefix count distribution:")
        for n_prefix in sorted(prefix_hist):
            count = prefix_hist[n_prefix]
            print(f"    {n_prefix} prefix: {count} units ({100 * count / n_units:.1f}%)")
    if prefix_gain_counts:
        print("\n  average gain per accepted prefix:")
        for pos in sorted(prefix_gain_counts):
            count = prefix_gain_counts[pos]
            gain = prefix_gain_sums[pos] / max(count, 1)
            net_gain = prefix_net_gain_sums[pos] / max(count, 1)
            id_bits = prefix_id_bit_sums[pos] / max(count, 1)
            print(f"    prefix {pos}: data_gain={gain:.2f} bits  "
                  f"net_gain={net_gain:.2f} bits  id_cost={id_bits:.2f} bits  n={count}")

    if args.output:
        save_csv(rows, args.output)
        print(f"CSV -> {args.output}")


if __name__ == "__main__":
    main()
