"""Shared evaluation utilities for compression benchmarks.

Provides:
  EvalResult   — per-sample metrics dataclass
  EvalStats    — running aggregate
  MemoryTracker — context manager for peak GPU + RAM
  save_csv     — write results to CSV
  save_compressed / save_decompressed — persist artifacts
  parse_devices — "cuda:0,cuda:1" → list[torch.device]
  run_multi_gpu — data-parallel dispatch across devices via mp.spawn
"""
from __future__ import annotations

import csv
import os
import pickle
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional

import torch

try:
    import psutil as _psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    sample_id:        str
    original_bytes:   int
    compressed_bytes: int
    bpb:              float   # bits per original byte
    ratio:            float   # original / compressed
    compress_s:       float   # wall time for compression
    decompress_s:     float   # wall time for decompression (-1 = skipped)
    peak_gpu_mb:      float   # peak GPU memory during sample (-1 = CPU)
    # peak RSS memory during sample (-1 = unavailable)
    peak_ram_mb:      float
    roundtrip_ok:     int     # 1=pass, 0=fail, -1=skipped


class EvalStats:
    def __init__(self):
        self.n = 0
        self.orig = 0
        self.comp = 0
        self.t_enc = 0.0
        self.t_dec = 0.0
        self.n_rt_ok = 0
        self.n_rt = 0

    def update(self, r: EvalResult) -> None:
        self.n += 1
        self.orig += r.original_bytes
        self.comp += r.compressed_bytes
        self.t_enc += r.compress_s
        if r.decompress_s >= 0:
            self.t_dec += r.decompress_s
        if r.roundtrip_ok >= 0:
            self.n_rt += 1
            if r.roundtrip_ok:
                self.n_rt_ok += 1

    @property
    def bpb(self) -> float: return self.comp * 8 / max(self.orig, 1)
    @property
    def ratio(self) -> float: return self.orig / max(self.comp, 1)

    def print_summary(self, label: str = "") -> None:
        hdr = f"── {label} " if label else "── "
        print(f"\n{hdr}{'─' * (48 - len(hdr))}")
        print(f"  Samples:        {self.n}")
        print(
            f"  Original:       {self.orig:>14,} B  ({self.orig/2**20:.1f} MB)")
        print(
            f"  Compressed:     {self.comp:>14,} B  ({self.comp/2**20:.1f} MB)")
        print(f"  BPB:            {self.bpb:.4f}")
        print(f"  Ratio:          {self.ratio:.4f}×")
        print(f"  Avg compress:   {self.t_enc / max(self.n, 1):.3f} s/sample")
        if self.t_dec > 0:
            print(
                f"  Avg decompress: {self.t_dec / max(self.n, 1):.3f} s/sample")
        if self.n_rt > 0:
            print(f"  Round-trip OK:  {self.n_rt_ok}/{self.n_rt}")


# ---------------------------------------------------------------------------
# Memory tracker
# ---------------------------------------------------------------------------

class MemoryTracker:
    """Context manager that records peak GPU and CPU-RSS memory during a block."""

    def __init__(self, device: torch.device):
        self.device = device
        self.peak_gpu_mb = -1.0
        self.peak_ram_mb = -1.0

    def __enter__(self) -> "MemoryTracker":
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            self._proc = _psutil.Process(os.getpid())
        return self

    def __exit__(self, *_) -> None:
        if self.device.type == "cuda":
            self.peak_gpu_mb = torch.cuda.max_memory_allocated(
                self.device) / 2**20
            self.peak_ram_mb = self._proc.memory_info().rss / 2**20


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_csv(results: List[EvalResult], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)
    print(f"Results saved → {path}")


def save_compressed(cd_bytes: bytes, metadata: dict, original_length: int,
                    out_dir: str, sample_id: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{sample_id}.bin")
    with open(path, "wb") as f:
        pickle.dump({"bytes": cd_bytes, "meta": metadata,
                     "original_length": original_length}, f)


def save_decompressed(data: bytes, out_dir: str, sample_id: str, ext: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{sample_id}.{ext}"), "wb") as f:
        f.write(data)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Batch size auto-selection
# ---------------------------------------------------------------------------

def length_stats(lengths: List[int]) -> Dict[str, Any]:
    """Compute percentile statistics of a token-length distribution."""
    import numpy as np
    arr = np.array(lengths, dtype=float)
    def pct(q): return int(np.percentile(arr, q))
    p50, p90, p95, p99 = pct(50), pct(90), pct(95), pct(99)
    return dict(
        n=len(lengths), mean=float(arr.mean()),
        p50=p50, p90=p90, p95=p95, p99=p99, max=int(arr.max()),
        tail_ratio=round(p99 / max(p50, 1), 2),
    )


def auto_batch_size(
    probe_fn: Callable[[int, int], None],
    device: torch.device,
    lengths: List[int],
    safety: float = 0.75,
    max_batch: int = 64,
    n_samples: Optional[int] = None,
    verbose: bool = True,
) -> int:
    """Auto-select batch size from GPU memory + sequence length distribution.

    Probes at the maximum sequence length so the returned batch size is safe
    for every sample in the dataset.

    ``n_samples`` caps the batch at the total dataset size.  When omitted it
    defaults to ``len(lengths)``, which is correct when *lengths* covers the
    full dataset.  Pass it explicitly when *lengths* is only a representative
    sample (e.g. bGPT where all sequences are the same length and a single
    value suffices to describe the distribution).

    Returns 1 for CPU or on any probe failure.
    """
    if device.type != "cuda" or not lengths:
        return 1

    stats = length_stats(lengths)
    n_cap = n_samples if n_samples is not None else stats["n"]

    total_mem = torch.cuda.get_device_properties(device).total_memory
    alloc_mem = torch.cuda.memory_allocated(device)
    free_mem = total_mem - alloc_mem

    rep_len = stats["max"]

    if verbose:
        print(f"\n  [auto_batch_size]  GPU: {total_mem/2**20:.0f} MB total"
              f", {alloc_mem/2**20:.0f} MB used by model"
              f", {free_mem/2**20:.0f} MB free")
        print(f"  Length distribution (n={stats['n']}): "
              f"p50={stats['p50']}  p90={stats['p90']}  "
              f"p95={stats['p95']}  p99={stats['p99']}  "
              f"max={stats['max']}")
        print(f"  Representative length: {rep_len} (max)")

    # Probe: measure incremental activation memory for one sample at rep_len
    try:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        mem_before = torch.cuda.memory_allocated(device)

        probe_fn(1, rep_len)

        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        mem_per_sample = max(peak - mem_before, 1)
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        if verbose:
            print(f"  OOM at rep_len={rep_len} — batch_size=1")
        torch.cuda.empty_cache()
        return 1
    except Exception as exc:
        if verbose:
            print(f"  Probe failed ({exc}) — batch_size=1")
        return 1

    batch_size = max(1, min(
        max_batch,
        n_cap,
        int(free_mem * safety / mem_per_sample),
    ))

    if verbose:
        print(f"  Probe: {mem_per_sample/2**20:.1f} MB/sample"
              f"  → batch_size={batch_size}  (safety={safety})")

    return batch_size


def parse_devices(device_str: str) -> List[torch.device]:
    """'cuda:0,cuda:1' → [device('cuda:0'), device('cuda:1')]"""
    return [torch.device(d.strip()) for d in device_str.split(",")]


# ---------------------------------------------------------------------------
# Multi-GPU dispatcher
# ---------------------------------------------------------------------------

def _mp_worker(rank: int, devices: List[torch.device], fn: Callable,
               shards: List[List[int]], fn_kwargs: dict, tmp_prefix: str) -> None:
    """Called by mp.spawn in each subprocess. Runs fn and pickles results."""
    results = fn(devices[rank], shards[rank], **fn_kwargs)
    tmp_path = f"{tmp_prefix}.rank{rank}"
    os.makedirs(os.path.dirname(os.path.abspath(tmp_path)), exist_ok=True)
    with open(tmp_path, "wb") as f:
        pickle.dump(results, f)


def run_multi_gpu(
    fn: Callable[[torch.device, List[int], Any], List[EvalResult]],
    all_indices: List[int],
    devices: List[torch.device],
    fn_kwargs: dict,
    tmp_prefix: str = "/tmp/_eval_worker",
) -> List[EvalResult]:
    """Data-parallel evaluation across devices.

    fn signature: fn(device, index_shard, **fn_kwargs) → List[EvalResult]
    Results are returned in the original all_indices order.
    """
    if len(devices) == 1:
        return fn(devices[0], all_indices, **fn_kwargs)

    import torch.multiprocessing as mp

    shards = [all_indices[r::len(devices)] for r in range(len(devices))]
    mp.spawn(_mp_worker,
             args=(devices, fn, shards, fn_kwargs, tmp_prefix),
             nprocs=len(devices),
             join=True)

    by_idx: Dict[int, EvalResult] = {}
    for rank, shard in enumerate(shards):
        with open(f"{tmp_prefix}.rank{rank}", "rb") as f:
            for idx, result in zip(shard, pickle.load(f)):
                by_idx[idx] = result
        os.remove(f"{tmp_prefix}.rank{rank}")

    return [by_idx[i] for i in all_indices]
