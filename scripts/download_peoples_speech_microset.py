#!/usr/bin/env python3
"""Download People's Speech microset and export bGPT-ready WAV files."""
from __future__ import annotations

import argparse
import io
import json
import os
import wave
import warnings
from pathlib import Path
from typing import Optional

# MUST be set before any Hugging Face import so the mirror is used.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# PCM WAV export uses Python's wave module internally and does not need ffmpeg.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Couldn't find ffmpeg or avconv.*",
        category=RuntimeWarning,
    )
    from pydub import AudioSegment


REPO_ID = "MLCommons/peoples_speech"
PARQUET_FILE = "microset/train-00000-of-00001.parquet"
CHANNELS = 1
SAMPLE_WIDTH = 1


def _download_parquet(output_dir: Path) -> Path:
    download_dir = output_dir / "_downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(
        repo_id=REPO_ID,
        filename=PARQUET_FILE,
        repo_type="dataset",
        local_dir=str(download_dir),
    ))


def _decode_audio(encoded: bytes) -> AudioSegment:
    audio_f, sample_rate = sf.read(io.BytesIO(encoded))
    audio_f = np.asarray(audio_f)
    if audio_f.ndim not in (1, 2):
        raise ValueError(f"Expected mono/stereo PCM, got shape {audio_f.shape}")

    channels = 1 if audio_f.ndim == 1 else audio_f.shape[1]
    audio_i16 = (audio_f * 32768).clip(-32768, 32767).astype(np.int16)
    return AudioSegment(
        data=audio_i16.tobytes(),
        sample_width=audio_i16.dtype.itemsize,
        frame_rate=int(sample_rate),
        channels=channels,
    )


def _convert_to_bgpt_wav(encoded: bytes, output_path: Path) -> tuple[int, int]:
    """Convert to mono 8-bit PCM WAV, keeping the clip's native sample rate.

    Returns (frame_count, sample_rate).
    """
    segment = _decode_audio(encoded)
    segment = segment.set_channels(CHANNELS)
    segment = segment.set_sample_width(SAMPLE_WIDTH)
    segment.export(output_path, format="wav")

    with wave.open(str(output_path), "rb") as wav_file:
        actual = (
            wav_file.getnchannels(),
            wav_file.getsampwidth(),
            wav_file.getcomptype(),
        )
        expected = (CHANNELS, SAMPLE_WIDTH, "NONE")
        if actual != expected:
            raise ValueError(
                f"Unexpected output WAV format for {output_path}: "
                f"expected {expected}, got {actual}"
            )
        return wav_file.getnframes(), wav_file.getframerate()


def export_microset(
    parquet_path: Path,
    output_dir: Path,
    limit: Optional[int] = None,
    batch_size: int = 32,
) -> int:
    parquet_file = pq.ParquetFile(parquet_path)
    total = parquet_file.metadata.num_rows
    if limit is not None:
        total = min(total, limit)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    exported = 0

    columns = ["id", "audio", "duration_ms", "text"]
    with manifest_path.open("w", encoding="utf-8") as manifest:
        progress = tqdm(total=total, desc="Exporting WAV", unit="clip")
        try:
            for batch in parquet_file.iter_batches(
                batch_size=batch_size,
                columns=columns,
            ):
                for row in batch.to_pylist():
                    if limit is not None and exported >= limit:
                        break

                    audio = row["audio"]
                    if not isinstance(audio, dict) or not isinstance(
                        audio.get("bytes"), bytes
                    ):
                        raise TypeError(
                            f"Row {exported} must contain audio={{'bytes': bytes, ...}}"
                        )

                    filename = f"{exported:06d}.wav"
                    output_path = output_dir / filename
                    frames, sample_rate = _convert_to_bgpt_wav(audio["bytes"], output_path)
                    manifest.write(json.dumps({
                        "index": exported,
                        "file": filename,
                        "id": row["id"],
                        "source_path": audio.get("path"),
                        "source_duration_ms": row["duration_ms"],
                        "duration_ms": frames * 1000 // sample_rate,
                        "sample_rate": sample_rate,
                        "text": row["text"],
                    }, ensure_ascii=False) + "\n")
                    exported += 1
                    progress.update(1)

                if limit is not None and exported >= limit:
                    break
        finally:
            progress.close()

    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary:
        json.dump({
            "source_repo": REPO_ID,
            "source_file": PARQUET_FILE,
            "source_parquet": str(parquet_path),
            "num_samples": exported,
            "audio_format": {
                "container": "wav",
                "sample_rate": "native (unchanged per clip; see manifest.jsonl)",
                "channels": CHANNELS,
                "sample_width_bytes": SAMPLE_WIDTH,
                "codec": "PCM_U8",
            },
        }, summary, indent=2)
        summary.write("\n")

    return exported


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download People's Speech microset and convert it to "
            "mono 8-bit PCM WAV files at each clip's native sample rate."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/peoples_speech_microset_wav"),
        help="Output directory for WAV files and metadata.",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        help="Use an existing microset parquet instead of downloading it.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert only the first N clips (useful for a smoke test).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    parquet_path = args.parquet or _download_parquet(args.output)
    if not parquet_path.is_file():
        parser.error(f"Parquet file not found: {parquet_path}")

    count = export_microset(
        parquet_path=parquet_path,
        output_dir=args.output,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    print(f"Exported {count} WAV files to {args.output}")


if __name__ == "__main__":
    main()
