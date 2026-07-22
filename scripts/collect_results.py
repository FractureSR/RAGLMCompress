#!/usr/bin/env python3
"""Merge one main-experiment run into a single table with one shared rate.

The evaluators do not agree on a denominator.  ``eval_bgpt`` / ``eval_rac_bgpt``
divide by the padded per-patch byte count, which is larger than the image and
differs between the two of them whenever their patch rectangles tile the image
differently.  ``eval_baselines`` divides by tight RGB8.  Comparing the printed
``bpb``/``ratio`` columns across those tracks therefore compares different
fractions.

This script recomputes every method against the *true* canonical sample — the
exact bytes ``utils.baseline_data`` hands to the conventional codecs: UTF-8 for
text, ``w*h*3`` RGB8 for images, header-free PCM_U8 for audio.  Padding is
counted in a method's emitted bits (it really is transmitted) but never in the
denominator, so a method is never rewarded for padding its input.

Only samples that every listed method actually produced a row for are used, so
each dataset's table is a like-for-like micro average.

    python scripts/collect_results.py --run-dir results/main
    python scripts/collect_results.py --run-dir results/main --datasets arxiv_cl
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.baseline_data import load_eval_split


# Neural evaluators write utils.eval_utils.EvalResult; the conventional ones
# write their own dataclasses.  Method label -> (file, schema).
_NEURAL_FILES = (
    ("llm-none", "llm_baseline.csv"),
    ("bgpt-none", "bgpt_baseline.csv"),
    ("rac-fixed-index", "rac_fixed.csv"),
    ("rac-calibrated-index", "rac_calibrated.csv"),
)
_CODEC_FILES = ("codecs.csv", "codecs_dict.csv")
_DELTA_FILES = ("delta.csv",)


@dataclass
class Row:
    """One method's result for one held-out sample."""

    method: str
    position: int
    total_bits: float
    reported_original_bytes: float
    roundtrip_ok: int


@dataclass
class Summary:
    dataset: str
    modality: str
    method: str
    n: int
    true_bytes: int
    pixels: int
    total_bits: float
    reported_bytes: float
    roundtrip: str

    @property
    def bpb(self) -> float:
        return self.total_bits / max(self.true_bytes, 1)

    @property
    def bpp(self) -> float:
        return self.total_bits / self.pixels if self.pixels else math.nan

    @property
    def ratio(self) -> float:
        return self.true_bytes * 8 / self.total_bits if self.total_bits else math.inf

    @property
    def reported_bpb(self) -> float:
        """The evaluator's own bpb, kept so a padded denominator stays visible."""
        return self.total_bits / max(self.reported_bytes, 1)


def _position(sample_id: str) -> int:
    """Eval-split position from any evaluator's sample id.

    ``doc000012``, ``image000012``, ``rac:doc000012`` and
    ``rac_bgpt:image000012`` all denote position 12 of the persisted split.
    """
    tail = sample_id.rsplit(":", 1)[-1]
    digits = ""
    for char in reversed(tail):
        if not char.isdigit():
            break
        digits = char + digits
    if not digits:
        raise ValueError(f"cannot read an eval position from sample_id {sample_id!r}")
    return int(digits)


def _read_csv(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _neural_rows(path: str, method: str) -> List[Row]:
    rows = []
    for record in _read_csv(path):
        # compressed_bytes already carries the RAC index-bit cost.
        rows.append(Row(
            method=method,
            position=_position(record["sample_id"]),
            total_bits=float(record["compressed_bytes"]) * 8.0,
            reported_original_bytes=float(record["original_bytes"]),
            roundtrip_ok=int(record["roundtrip_ok"]),
        ))
    return rows


def _codec_rows(path: str) -> List[Row]:
    rows = []
    for record in _read_csv(path):
        rows.append(Row(
            method=record["codec"],
            position=_position(record["sample_id"]),
            total_bits=float(record["total_bits"]),
            reported_original_bytes=float(record["original_bytes"]),
            roundtrip_ok=int(record["roundtrip_ok"]),
        ))
    return rows


def _delta_rows(path: str) -> List[Row]:
    rows = []
    for record in _read_csv(path):
        if record.get("error"):
            continue
        rows.append(Row(
            method=f"{record['codec']}/{record['policy']}",
            position=_position(record["sample_id"]),
            total_bits=float(record["total_bits"]),
            reported_original_bytes=float(record["original_bytes"]),
            roundtrip_ok=int(record["roundtrip_ok"]),
        ))
    return rows


def collect_dataset(
    run_dir: str, dataset: str
) -> Tuple[List[Summary], List[str]]:
    """Summarise every method CSV found for one dataset."""
    notes: List[str] = []
    root = os.path.join(run_dir, dataset)
    csv_dir = os.path.join(root, "csv")
    database = os.path.join(root, "db")
    if not os.path.isdir(csv_dir):
        return [], [f"{dataset}: no csv/ directory"]
    if not os.path.isfile(os.path.join(database, "meta.json")):
        return [], [f"{dataset}: no prepared database at {database}"]

    split = load_eval_split(database)
    modality = split.modality
    truth: Dict[int, Tuple[int, int]] = {}
    for position, sample in enumerate(split.test):
        pixels = 0
        if modality == "image":
            pixels = int(sample.metadata["width"]) * int(sample.metadata["height"])
        truth[position] = (len(sample.data), pixels)

    by_method: Dict[str, List[Row]] = {}
    for method, filename in _NEURAL_FILES:
        path = os.path.join(csv_dir, filename)
        if os.path.isfile(path):
            by_method.setdefault(method, []).extend(_neural_rows(path, method))
    for filename in _CODEC_FILES:
        path = os.path.join(csv_dir, filename)
        if os.path.isfile(path):
            for row in _codec_rows(path):
                by_method.setdefault(row.method, []).append(row)
    for filename in _DELTA_FILES:
        path = os.path.join(csv_dir, filename)
        if os.path.isfile(path):
            for row in _delta_rows(path):
                by_method.setdefault(row.method, []).append(row)

    if not by_method:
        return [], [f"{dataset}: no method CSVs found in {csv_dir}"]

    # Like-for-like: keep only the samples every method scored.
    common = set.intersection(
        *({row.position for row in rows} for rows in by_method.values()))
    unknown = sorted(p for p in common if p not in truth)
    if unknown:
        notes.append(
            f"{dataset}: {len(unknown)} sample position(s) are outside the "
            f"persisted eval split (first: {unknown[0]}); dropped")
        common -= set(unknown)
    if not common:
        return [], [f"{dataset}: methods share no evaluated samples"]

    for method, rows in sorted(by_method.items()):
        extra = {row.position for row in rows} - common
        if extra:
            notes.append(
                f"{dataset}/{method}: {len(extra)} sample(s) dropped, not "
                "evaluated by every method")

    summaries: List[Summary] = []
    for method, rows in sorted(by_method.items()):
        seen: Dict[int, Row] = {}
        for row in rows:
            if row.position in common:
                seen[row.position] = row
        if not seen:
            continue
        true_bytes = sum(truth[p][0] for p in seen)
        pixels = sum(truth[p][1] for p in seen)
        roundtrip_values = {row.roundtrip_ok for row in seen.values()}
        if roundtrip_values == {-1}:
            roundtrip = "skipped"
        elif 0 in roundtrip_values:
            roundtrip = "FAILED"
        elif -1 in roundtrip_values:
            roundtrip = "partial"
        else:
            roundtrip = "ok"
        summaries.append(Summary(
            dataset=dataset,
            modality=modality,
            method=method,
            n=len(seen),
            true_bytes=true_bytes,
            pixels=pixels,
            total_bits=sum(row.total_bits for row in seen.values()),
            reported_bytes=sum(row.reported_original_bytes for row in seen.values()),
            roundtrip=roundtrip,
        ))
    summaries.sort(key=lambda s: s.bpb)
    return summaries, notes


def print_dataset(summaries: Sequence[Summary]) -> None:
    if not summaries:
        return
    head = summaries[0]
    is_image = head.modality == "image"
    print(f"\n== {head.dataset} ({head.modality}, n={head.n} samples, "
          f"{head.true_bytes:,} canonical bytes)")
    columns = f"{'method':<24} {'bpb':>9} {'ratio':>9}"
    if is_image:
        columns += f" {'bpp':>9}"
    columns += f" {'pad%':>7} {'roundtrip':>10}"
    print(columns)
    print("-" * len(columns))
    for summary in summaries:
        # How much of this method's own denominator was padding the evaluator
        # counted as payload.  0 for every conventional codec by construction.
        padding = 100.0 * (summary.reported_bytes - summary.true_bytes) / max(
            summary.true_bytes, 1)
        line = f"{summary.method:<24} {summary.bpb:>9.4f} {summary.ratio:>9.4f}"
        if is_image:
            line += f" {summary.bpp:>9.4f}"
        line += f" {padding:>6.1f}% {summary.roundtrip:>10}"
        print(line)


def save_csv(summaries: Sequence[Summary], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "dataset", "modality", "method", "n", "true_bytes", "pixels",
        "total_bits", "bpb", "bpp", "ratio", "reported_bytes", "reported_bpb",
        "roundtrip",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({
                "dataset": summary.dataset,
                "modality": summary.modality,
                "method": summary.method,
                "n": summary.n,
                "true_bytes": summary.true_bytes,
                "pixels": summary.pixels,
                "total_bits": f"{summary.total_bits:.3f}",
                "bpb": f"{summary.bpb:.6f}",
                "bpp": "" if math.isnan(summary.bpp) else f"{summary.bpp:.6f}",
                "ratio": f"{summary.ratio:.6f}",
                "reported_bytes": f"{summary.reported_bytes:.0f}",
                "reported_bpb": f"{summary.reported_bpb:.6f}",
                "roundtrip": summary.roundtrip,
            })
    print(f"\nCSV -> {path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge a main-experiment run into one comparable table")
    parser.add_argument("--run-dir", default="results/main",
                        help="OUT_DIR of scripts/run_main_experiment.sh")
    parser.add_argument("--datasets", default=None,
                        help="comma/space separated subset (default: all found)")
    parser.add_argument("--output", default=None, metavar="CSV",
                        help="default: <run-dir>/summary.csv")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    if args.datasets:
        datasets = [d for d in args.datasets.replace(",", " ").split() if d]
    else:
        datasets = sorted(
            name for name in os.listdir(run_dir)
            if os.path.isdir(os.path.join(run_dir, name, "csv")))
    if not datasets:
        raise SystemExit(f"no dataset directories with csv/ under {run_dir}")

    all_summaries: List[Summary] = []
    all_notes: List[str] = []
    for dataset in datasets:
        summaries, notes = collect_dataset(run_dir, dataset)
        all_notes.extend(notes)
        all_summaries.extend(summaries)
        print_dataset(summaries)

    if all_notes:
        print("\nNotes:")
        for note in all_notes:
            print(f"  {note}")
    if not all_summaries:
        raise SystemExit("no results to summarise")

    save_csv(all_summaries, args.output or os.path.join(run_dir, "summary.csv"))
    print("\nbpb/ratio/bpp are micro averages against the true canonical sample "
          "(padding excluded from the denominator, included in the bits).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
