#!/usr/bin/env python3
"""Evaluate conventional lossless codecs on complete prepared eval samples.

Examples
--------
    python evaluation/eval_baselines.py \
        --database results/rac_c_db --modality text \
        --codecs zstd,openzl,cmix --output results/text_baselines.csv

    python evaluation/eval_baselines.py \
        --database results/rac_img_db --modality image \
        --codecs zstd,png,jpegxl,webp --output results/image_baselines.csv

No matched-window track is implemented here.  Every codec sees one complete
logical sample: document UTF-8, image RGB8 pixels, or header-free PCM_U8 audio.
The encoded artifact is always decoded and compared byte-for-byte before a row
is accepted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.baseline_data import BaselineSample, load_base_samples, load_eval_split


DEFAULT_CODECS: Mapping[str, Tuple[str, ...]] = {
    "text": ("zstd", "openzl", "cmix"),
    "image": ("zstd", "openzl", "cmix", "png", "jpegxl", "webp"),
    "audio": ("zstd", "openzl", "cmix", "flac"),
}

ALLOWED_CODECS: Mapping[str, frozenset[str]] = {
    "text": frozenset(("zstd", "openzl", "cmix", "zstd-dict")),
    "image": frozenset(
        ("zstd", "openzl", "cmix", "zstd-dict", "png", "jpegxl", "webp")
    ),
    "audio": frozenset(("zstd", "openzl", "cmix", "zstd-dict", "flac")),
}

_ALIASES = {"jxl": "jpegxl", "jpeg-xl": "jpegxl", "zstd_dict": "zstd-dict"}
_ARTIFACT_SUFFIX = {
    "zstd": ".zst",
    "zstd-dict": ".zst",
    "openzl": ".openzl",
    "cmix": ".cmix",
    "png": ".png",
    "jpegxl": ".jxl",
    "webp": ".webp",
    "flac": ".flac",
}

CODEC_SETTINGS: Mapping[str, str] = {
    "zstd": "--ultra -22 -T1",
    "zstd-dict": "--ultra -22 -T1; shared dictionary",
    "openzl": "zli compress --profile serial",
    "cmix": "reference CLI default model",
    "png": "Pillow optimize=True compress_level=9",
    "jpegxl": "lossless distance=0 effort=10 threads=1",
    "webp": "lossless quality=100 method=6 exact=True",
    "flac": "raw PCM_U8 -8 -e -p; no padding/seektable",
}


@dataclass(frozen=True)
class BaselineResult:
    codec: str
    codec_version: str
    codec_settings: str
    modality: str
    sample_id: str
    eval_position: int
    source_index: int
    input_sha256: str
    original_bytes: int
    source_pixels: int
    compressed_bytes: int
    side_info_bits: int
    total_bits: int
    bpb: float
    bpp: float
    ratio: float
    compress_s: float
    decompress_s: float
    roundtrip_ok: int


@dataclass(frozen=True)
class BaselineSummary:
    codec: str
    codec_version: str
    codec_settings: str
    modality: str
    n_samples: int
    total_original_bytes: int
    total_source_pixels: int
    total_compressed_bytes: int
    shared_state_bytes: int
    self_contained_bytes: int
    bpb: float
    bpp: float
    self_contained_bpb: float
    self_contained_bpp: float
    ratio: float
    self_contained_ratio: float
    total_compress_s: float
    total_decompress_s: float
    roundtrip_ok: int


def normalize_codec_names(
    modality: str, codecs: Optional[str | Sequence[str]]
) -> List[str]:
    """Parse comma/repeated codec names, canonicalize aliases, and validate."""
    if modality not in DEFAULT_CODECS:
        raise ValueError(f"Unknown modality {modality!r}")
    if codecs is None:
        values: Iterable[str] = DEFAULT_CODECS[modality]
    elif isinstance(codecs, str):
        values = codecs.split(",")
    else:
        expanded: List[str] = []
        for item in codecs:
            expanded.extend(item.split(","))
        values = expanded

    result: List[str] = []
    for raw in values:
        name = _ALIASES.get(raw.strip().lower(), raw.strip().lower())
        if not name:
            continue
        if name not in ALLOWED_CODECS[modality]:
            raise ValueError(
                f"Codec {raw!r} is not a whole-{modality} baseline; allowed: "
                f"{', '.join(sorted(ALLOWED_CODECS[modality]))}"
            )
        if name not in result:
            result.append(name)
    if not result:
        raise ValueError("No codecs selected")
    return result


def evaluate_codec(
    codec: Any,
    samples: Sequence[BaselineSample],
    *,
    codec_name: Optional[str] = None,
    codec_version: Optional[str] = None,
    save_artifacts: Optional[os.PathLike[str] | str] = None,
) -> List[BaselineResult]:
    """Encode, genuinely decode, and byte-verify every whole sample."""
    if not samples:
        return []
    name = codec_name or str(getattr(codec, "name", type(codec).__name__))
    version = codec_version if codec_version is not None else _safe_version(codec)
    artifact_dir = Path(save_artifacts) if save_artifacts is not None else None
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)

    rows: List[BaselineResult] = []
    for position, sample in enumerate(samples, start=1):
        print(
            f"  [{name}] {position}/{len(samples)} {sample.sample_id} "
            f"({len(sample.data):,} B)",
            flush=True,
        )
        metadata = dict(sample.metadata)

        start = time.perf_counter()
        encoded_result = codec.encode(sample.data, metadata)
        encode_wall_s = time.perf_counter() - start
        artifact = _result_bytes(
            encoded_result,
            attributes=("artifact", "artifact_bytes", "data", "compressed_bytes"),
            operation=f"{name} encode",
        )
        # Use end-to-end wall time for every codec.  Adapter subprocess timing
        # alone would omit mandatory bridges such as RGB->PNG before cjxl and
        # PNG->RGB after djxl, as well as temporary-file I/O.
        encode_s = encode_wall_s

        start = time.perf_counter()
        decoded_result = codec.decode(artifact, metadata)
        decode_wall_s = time.perf_counter() - start
        decoded = _result_bytes(
            decoded_result,
            attributes=("payload", "payload_bytes", "data", "decoded_bytes"),
            operation=f"{name} decode",
        )
        decode_s = decode_wall_s
        roundtrip_ok = int(decoded == sample.data)

        if artifact_dir is not None:
            suffix = _ARTIFACT_SUFFIX.get(name, ".bin")
            path = artifact_dir / name / f"{sample.sample_id}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(artifact)

        original_bytes = len(sample.data)
        source_pixels = 0
        if sample.modality == "image":
            source_pixels = int(metadata["width"]) * int(metadata["height"])
        compressed_bytes = len(artifact)
        total_bits = compressed_bytes * 8
        eval_position = (
            int(sample.eval_position)
            if sample.eval_position is not None
            else position - 1
        )
        rows.append(
            BaselineResult(
                codec=name,
                codec_version=version,
                codec_settings=CODEC_SETTINGS.get(name, "unknown"),
                modality=sample.modality,
                sample_id=sample.sample_id,
                eval_position=eval_position,
                source_index=sample.source_index,
                input_sha256=hashlib.sha256(sample.data).hexdigest(),
                original_bytes=original_bytes,
                source_pixels=source_pixels,
                compressed_bytes=compressed_bytes,
                side_info_bits=0,
                total_bits=total_bits,
                bpb=total_bits / max(original_bytes, 1),
                bpp=(total_bits / source_pixels) if source_pixels else math.nan,
                ratio=original_bytes / max(compressed_bytes, 1),
                compress_s=encode_s,
                decompress_s=decode_s,
                roundtrip_ok=roundtrip_ok,
            )
        )
        if not roundtrip_ok:
            raise RuntimeError(
                f"{name} failed exact round-trip verification for "
                f"{sample.sample_id}: decoded {len(decoded)} bytes, expected "
                f"{len(sample.data)}. No result should be reported for this codec."
            )
    return rows


def summarize_results(
    results: Sequence[BaselineResult],
    shared_state_bytes: Optional[Mapping[str, int]] = None,
) -> List[BaselineSummary]:
    """Compute codec-level micro averages (sum bytes, then divide)."""
    grouped: Dict[Tuple[str, str, str, str], List[BaselineResult]] = {}
    for row in results:
        grouped.setdefault(
            (row.codec, row.codec_version, row.codec_settings, row.modality), []
        ).append(row)
    state = shared_state_bytes or {}
    summaries: List[BaselineSummary] = []
    for (codec, version, settings, modality), rows in grouped.items():
        original = sum(row.original_bytes for row in rows)
        source_pixels = sum(row.source_pixels for row in rows)
        compressed = sum(row.compressed_bytes for row in rows)
        shared = int(state.get(codec, 0))
        self_contained = compressed + shared
        summaries.append(
            BaselineSummary(
                codec=codec,
                codec_version=version,
                codec_settings=settings,
                modality=modality,
                n_samples=len(rows),
                total_original_bytes=original,
                total_source_pixels=source_pixels,
                total_compressed_bytes=compressed,
                shared_state_bytes=shared,
                self_contained_bytes=self_contained,
                bpb=compressed * 8 / max(original, 1),
                bpp=(compressed * 8 / source_pixels) if source_pixels else math.nan,
                self_contained_bpb=self_contained * 8 / max(original, 1),
                self_contained_bpp=(
                    self_contained * 8 / source_pixels
                    if source_pixels
                    else math.nan
                ),
                ratio=original / max(compressed, 1),
                self_contained_ratio=original / max(self_contained, 1),
                total_compress_s=sum(row.compress_s for row in rows),
                total_decompress_s=sum(row.decompress_s for row in rows),
                roundtrip_ok=int(all(row.roundtrip_ok == 1 for row in rows)),
            )
        )
    return summaries


def save_results_csv(results: Sequence[BaselineResult], path: os.PathLike[str] | str) -> None:
    _save_dataclass_csv(results, path, BaselineResult)


def save_summary_csv(
    summaries: Sequence[BaselineSummary], path: os.PathLike[str] | str
) -> None:
    _save_dataclass_csv(summaries, path, BaselineSummary)


def print_summaries(summaries: Sequence[BaselineSummary]) -> None:
    if not summaries:
        print("No codec results (all requested optional codecs were unavailable).")
        return
    print("\nWhole-sample baseline micro summary")
    print(
        f"{'codec':<14} {'samples':>8} {'original B':>14} {'artifact B':>14} "
        f"{'BPB':>10} {'ratio':>10} {'roundtrip':>11}"
    )
    print("-" * 86)
    for summary in summaries:
        verified = "yes" if summary.roundtrip_ok else "NO"
        print(
            f"{summary.codec:<14} {summary.n_samples:>8d} "
            f"{summary.total_original_bytes:>14,d} "
            f"{summary.total_compressed_bytes:>14,d} "
            f"{summary.bpb:>10.4f} {summary.ratio:>10.4f} {verified:>11}"
        )
        if summary.total_source_pixels:
            print(f"  semantic image rate: {summary.bpp:.4f} bits/pixel")
        if summary.shared_state_bytes:
            print(
                f"  shared state: {summary.shared_state_bytes:,} B; "
                f"self-contained BPB={summary.self_contained_bpb:.4f}, "
                f"ratio={summary.self_contained_ratio:.4f}"
            )
            if summary.total_source_pixels:
                print(
                    "  self-contained semantic image rate: "
                    f"{summary.self_contained_bpp:.4f} bits/pixel"
                )


def _result_bytes(result: Any, *, attributes: Sequence[str], operation: str) -> bytes:
    if isinstance(result, (bytes, bytearray, memoryview)):
        return bytes(result)
    if isinstance(result, tuple) and result:
        candidate = result[0]
        if isinstance(candidate, (bytes, bytearray, memoryview)):
            return bytes(candidate)
    if isinstance(result, Mapping):
        for attribute in attributes:
            candidate = result.get(attribute)
            if isinstance(candidate, (bytes, bytearray, memoryview)):
                return bytes(candidate)
    for attribute in attributes:
        candidate = getattr(result, attribute, None)
        if isinstance(candidate, (bytes, bytearray, memoryview)):
            return bytes(candidate)
    raise TypeError(
        f"{operation} returned {type(result).__name__}, but no byte artifact/payload "
        f"was found in {tuple(attributes)}"
    )


def _safe_version(codec: Any) -> str:
    version = getattr(codec, "version", None)
    if not callable(version):
        return "unknown"
    try:
        value = version()
    except Exception as exc:
        return f"unknown ({type(exc).__name__})"
    return str(value).replace("\n", " ").strip() or "unknown"


def _save_dataclass_csv(
    rows: Sequence[Any],
    path: os.PathLike[str] | str,
    row_type: Any,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row_type.__dataclass_fields__)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    print(f"CSV -> {destination}")


def _create_codec_and_state(
    name: str,
    *,
    timeout_s: float,
    tmp_dir: Optional[str],
    database: str,
    modality: str,
    dictionary_path: Optional[str],
    dictionary_size: int,
    save_dictionary: Optional[str],
) -> Tuple[Any, int]:
    # Imported only when evaluation starts: --help and data utilities remain
    # usable without any optional codec package/binary.
    from compression.baselines import create_codec

    if name != "zstd-dict":
        return create_codec(name, timeout_s=timeout_s, temp_dir=tmp_dir), 0

    if dictionary_path:
        dictionary = Path(dictionary_path).read_bytes()
        if not dictionary:
            raise ValueError(f"zstd dictionary is empty: {dictionary_path}")
    else:
        from compression.baselines import train_zstd_dictionary

        base = load_base_samples(database, modality=modality)
        print(
            f"Training zstd dictionary from {len(base)} complete base samples "
            f"({sum(len(sample.data) for sample in base):,} B) ...",
            flush=True,
        )
        trained = train_zstd_dictionary(
            (sample.data for sample in base),
            dictionary_size=dictionary_size,
            timeout_s=timeout_s,
            temp_dir=tmp_dir,
        )
        dictionary = trained.dictionary
        if save_dictionary:
            destination = Path(save_dictionary)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(dictionary)
            print(f"zstd dictionary -> {destination}")
    return create_codec(
        name,
        timeout_s=timeout_s,
        temp_dir=tmp_dir,
        dictionary=dictionary,
    ), len(dictionary)


def _is_missing_codec(exc: BaseException) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True
    if type(exc).__name__ == "MissingBinaryError":
        return True
    message = str(exc).lower()
    return "does not include lossless webp support" in message


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Lossless conventional baselines over whole samples from a prepared "
            "RAC eval split"
        )
    )
    parser.add_argument(
        "--database",
        required=True,
        help="Directory produced by prepare_rac_data_llm.py or prepare_rac_data_bgpt.py",
    )
    parser.add_argument(
        "--modality", choices=("text", "image", "audio"), default=None,
        help="Normally inferred from database metadata/files",
    )
    parser.add_argument(
        "--codecs",
        default=None,
        help="Comma-separated codecs (default: all applicable requested baselines)",
    )
    parser.add_argument(
        "--n-samples", "--n-docs", dest="n_samples", type=int, default=None,
        help="Number of test samples from the start of the persisted eval split",
    )
    parser.add_argument(
        "--calib-samples", "--calib-docs", dest="calib_samples", type=int, default=0,
        help="Exclude this many samples from the selected eval tail",
    )
    parser.add_argument(
        "--timeout-s", type=float, default=3600.0,
        help="Per codec subprocess timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help="temporary-file parent for external codecs (default: system temp)",
    )
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="Skip unavailable optional codec binaries instead of failing",
    )
    parser.add_argument("--output", default=None, metavar="CSV")
    parser.add_argument("--summary-output", default=None, metavar="CSV")
    parser.add_argument("--save-artifacts", default=None, metavar="DIR")
    parser.add_argument(
        "--zstd-dictionary", default=None, metavar="FILE",
        help="Existing shared dictionary for codec zstd-dict",
    )
    parser.add_argument(
        "--dictionary-size", type=int, default=112_640,
        help="Dictionary size when zstd-dict is trained from the base split",
    )
    parser.add_argument("--save-zstd-dictionary", default=None, metavar="FILE")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.timeout_s <= 0:
        raise ValueError(f"--timeout-s must be positive, got {args.timeout_s}")
    if args.dictionary_size <= 0:
        raise ValueError(
            f"--dictionary-size must be positive, got {args.dictionary_size}"
        )

    split = load_eval_split(
        args.database,
        modality=args.modality,
        n_samples=args.n_samples,
        calib_samples=args.calib_samples,
    )
    samples = list(split.test)
    codec_names = normalize_codec_names(split.modality, args.codecs)
    print(
        f"Loaded persisted {split.modality} eval split from {args.database}: "
        f"test={len(samples)}, calibration excluded={len(split.calibration)}; "
        "track=whole-sample"
    )

    all_rows: List[BaselineResult] = []
    shared_state: Dict[str, int] = {}
    skipped: List[Tuple[str, str]] = []
    for name in codec_names:
        try:
            codec, state_bytes = _create_codec_and_state(
                name,
                timeout_s=args.timeout_s,
                tmp_dir=args.tmp_dir,
                database=args.database,
                modality=split.modality,
                dictionary_path=args.zstd_dictionary,
                dictionary_size=args.dictionary_size,
                save_dictionary=args.save_zstd_dictionary,
            )
        except Exception as exc:
            if args.allow_missing and _is_missing_codec(exc):
                skipped.append((name, str(exc)))
                print(f"Skipping unavailable codec {name}: {exc}")
                continue
            raise

        shared_state[name] = state_bytes
        version = _safe_version(codec)
        try:
            rows = evaluate_codec(
                codec,
                samples,
                codec_name=name,
                codec_version=version,
                save_artifacts=args.save_artifacts,
            )
        except Exception as exc:
            if args.allow_missing and _is_missing_codec(exc):
                skipped.append((name, str(exc)))
                print(f"Skipping unavailable codec {name}: {exc}")
                continue
            raise RuntimeError(f"{name} whole-sample evaluation failed: {exc}") from exc
        all_rows.extend(rows)

    summaries = summarize_results(all_rows, shared_state)
    print_summaries(summaries)
    if skipped:
        print("\nSkipped optional codecs:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")
    if args.output:
        save_results_csv(all_rows, args.output)
    if args.summary_output:
        save_summary_csv(summaries, args.summary_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
