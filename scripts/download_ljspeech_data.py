#!/usr/bin/env python3
"""Download LJSpeech and export bGPT-ready WAV files.

Clips are kept at their **native** sample rate (LJSpeech is 22050 Hz mono) —
only channels and bit depth are converted, never the sample rate. Streams the
source tarball member-by-member so ``--limit-mb`` stops the download once
enough WAV data has been exported (no need to fetch the full ~2.7 GB archive
for a small experiment).
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
import wave
import warnings
from pathlib import Path
from typing import Optional, Tuple

import requests
from tqdm import tqdm

# PCM WAV export uses Python's wave module internally and does not need ffmpeg.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Couldn't find ffmpeg or avconv.*",
        category=RuntimeWarning,
    )
    from pydub import AudioSegment


SOURCE_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
CHANNELS = 1
SAMPLE_WIDTH = 1


def _convert_to_bgpt_wav(wav_bytes: bytes, output_path: Path) -> Tuple[int, int]:
    """Convert to mono 8-bit PCM WAV, keeping the clip's native sample rate.

    Returns (frame_count, sample_rate).
    """
    segment = AudioSegment.from_wav(io.BytesIO(wav_bytes))
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


def export_ljspeech(output_dir: Path, limit_mb: Optional[float], url: str) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    limit_bytes = int(limit_mb * 1024 * 1024) if limit_mb is not None else None
    exported = 0
    exported_bytes = 0

    print(f"Streaming {url} ...")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        resp.raw.decode_content = True
        with tarfile.open(fileobj=resp.raw, mode="r|bz2") as tar, \
                manifest_path.open("w", encoding="utf-8") as manifest:
            progress = tqdm(total=limit_bytes, desc="Exporting WAV",
                            unit="B", unit_scale=True, unit_divisor=1024)
            try:
                for member in tar:
                    if limit_bytes is not None and exported_bytes >= limit_bytes:
                        break
                    if not member.isfile() or not member.name.endswith(".wav"):
                        continue
                    fobj = tar.extractfile(member)
                    if fobj is None:
                        continue
                    wav_bytes = fobj.read()

                    clip_id = Path(member.name).stem
                    filename = f"{exported:06d}.wav"
                    output_path = output_dir / filename
                    frames, sample_rate = _convert_to_bgpt_wav(wav_bytes, output_path)
                    clip_bytes = output_path.stat().st_size
                    manifest.write(json.dumps({
                        "index": exported,
                        "file": filename,
                        "id": clip_id,
                        "duration_ms": frames * 1000 // sample_rate,
                        "sample_rate": sample_rate,
                        "bytes": clip_bytes,
                    }, ensure_ascii=False) + "\n")
                    exported += 1
                    exported_bytes += clip_bytes
                    progress.update(clip_bytes)
            finally:
                progress.close()
    # Streaming mode reads sequentially; the break above closes the tar (and
    # the HTTP connection) before the rest of the archive is downloaded.

    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary:
        json.dump({
            "source_url": url,
            "num_samples": exported,
            "total_bytes": exported_bytes,
            "audio_format": {
                "container": "wav",
                "sample_rate": "native (unchanged per clip; see manifest.jsonl)",
                "channels": CHANNELS,
                "sample_width_bytes": SAMPLE_WIDTH,
                "codec": "PCM_U8",
            },
        }, summary, indent=2)
        summary.write("\n")

    print(f"Exported {exported_bytes / (1024 * 1024):.1f} MiB "
          f"across {exported} clips")
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download LJSpeech and convert it to mono 8-bit PCM WAV files "
            "at each clip's native sample rate (no resampling)."
        )
    )
    parser.add_argument("--output", type=Path, default=Path("datasets/ljspeech_wav"))
    parser.add_argument("--url", default=SOURCE_URL)
    parser.add_argument(
        "--limit-mb", type=float, default=None,
        help=("Stop once this many MiB of WAV data have been exported "
              "(the clip crossing the limit is kept; default: whole dataset)."),
    )
    args = parser.parse_args()

    if args.limit_mb is not None and args.limit_mb <= 0:
        parser.error("--limit-mb must be positive")

    count = export_ljspeech(output_dir=args.output, limit_mb=args.limit_mb, url=args.url)
    print(f"Exported {count} WAV files to {args.output}")


if __name__ == "__main__":
    main()
