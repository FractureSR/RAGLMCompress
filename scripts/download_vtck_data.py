#!/usr/bin/env python3
"""Download the CSTR VCTK Corpus (0.92) and export bGPT-ready WAV files.

VCTK is 110 English speakers reading ~400 sentences each, recorded at 48 kHz
(FLAC, two microphones per utterance). Clips keep their **native** sample
rate — like the other speech download scripts, only channels and bit depth
are converted to mono 8-bit PCM, never the sample rate.

The source archive is a ~10.9 GB ZIP on Edinburgh DataShare. The server
supports HTTP range requests, so the script opens the remote ZIP directly
and fetches only the members it converts — ``--limit-mb``/``--speaker`` runs
never download the whole archive. For a full export, download the archive
once (``wget -c`` on the URL below) and pass it via ``--zip``.

Members are grouped by speaker inside the archive, so a bare ``--limit-mb``
run yields the first few speakers only; use ``--speaker p225 p226 ...`` to
pick specific ones. Each utterance exists as ``*_mic1.flac`` and
``*_mic2.flac``; ``--mic`` selects one (default mic1) so utterances are not
exported twice. FLAC decodes through soundfile/libsndfile — no ffmpeg needed.

Handle page: https://datashare.ed.ac.uk/handle/10283/3443
"""
from __future__ import annotations

import argparse
import io
import json
import time
import wave
import warnings
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import requests
import soundfile as sf
from tqdm import tqdm

# PCM WAV export uses Python's wave module internally and does not need ffmpeg.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Couldn't find ffmpeg or avconv.*",
        category=RuntimeWarning,
    )
    from pydub import AudioSegment


# The DataShare "bitstream/handle" URL pseudo-redirects (HTTP 200 + text body)
# to this API endpoint, which is what actually serves the ZIP with range
# support — so the script must point here directly.
SOURCE_URL = ("https://datashare.ed.ac.uk/server/api/core/bitstreams/"
              "535f4286-e54c-4038-838c-a02285e32cb2/content")
CHANNELS = 1
SAMPLE_WIDTH = 1


# ---------------------------------------------------------------------------
# Remote ZIP access (HTTP range requests)
# ---------------------------------------------------------------------------

class _HttpRangeFile(io.RawIOBase):
    """Read-only seekable file over HTTP range requests, for zipfile.

    zipfile reads the central directory from the end of the archive, then
    seeks to each member it extracts — with range requests only those byte
    spans are downloaded. Wrap in io.BufferedReader so the many small header
    reads coalesce into fewer HTTP requests. Large member reads are split
    into ≤ MAX_RANGE_BYTES requests (a short read is legal for readinto;
    callers loop) and each request is retried with backoff — range GETs are
    idempotent, and long single requests are exactly the ones flaky paths
    kill mid-body.
    """

    MAX_RANGE_BYTES = 4 << 20
    RETRIES = 5

    def __init__(self, url: str, session: Optional[requests.Session] = None):
        self.url = url
        self.session = session or requests.Session()
        head = self.session.head(url, timeout=30)
        head.raise_for_status()
        if head.headers.get("Accept-Ranges", "none").lower() != "bytes":
            raise IOError(
                f"{url} does not support HTTP range requests; download the "
                f"archive manually and pass it via --zip")
        self._size = int(head.headers["Content-Length"])
        self._pos = 0

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        return self._pos

    def tell(self) -> int:
        return self._pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        if self._pos >= self._size:
            return 0
        end = min(self._pos + len(b), self._pos + self.MAX_RANGE_BYTES,
                  self._size) - 1
        last_exc: Optional[Exception] = None
        for attempt in range(self.RETRIES):
            try:
                resp = self.session.get(
                    self.url, timeout=120,
                    headers={"Range": f"bytes={self._pos}-{end}"},
                )
                resp.raise_for_status()
                if resp.status_code != 206:
                    raise IOError(
                        f"server ignored the Range header (HTTP "
                        f"{resp.status_code}); download the archive manually "
                        f"and pass it via --zip")
                data = resp.content[: end - self._pos + 1]
                b[: len(data)] = data
                self._pos += len(data)
                return len(data)
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                wait = 2 ** attempt
                print(f"\n  range request bytes={self._pos}-{end} failed "
                      f"(attempt {attempt + 1}/{self.RETRIES}, retrying in "
                      f"{wait}s): {exc}")
                time.sleep(wait)
        raise IOError(
            f"range request bytes={self._pos}-{end} failed after "
            f"{self.RETRIES} attempts: {last_exc}") from last_exc


def _open_zip(zip_path: Optional[Path], url: str) -> zipfile.ZipFile:
    if zip_path is not None:
        if not zip_path.is_file():
            raise FileNotFoundError(f"ZIP archive not found: {zip_path}")
        return zipfile.ZipFile(zip_path)
    print(f"Opening remote ZIP via range requests: {url}")
    raw = _HttpRangeFile(url)
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 20))


# ---------------------------------------------------------------------------
# FLAC → mono 8-bit native-rate WAV
# ---------------------------------------------------------------------------

def _convert_to_bgpt_wav(flac_bytes: bytes, output_path: Path) -> Tuple[int, int]:
    """Convert to mono 8-bit PCM WAV, keeping the clip's native sample rate.

    Returns (frame_count, sample_rate).
    """
    audio_f, sample_rate = sf.read(io.BytesIO(flac_bytes))
    audio_f = np.asarray(audio_f)
    if audio_f.ndim not in (1, 2):
        raise ValueError(f"Expected mono/stereo PCM, got shape {audio_f.shape}")
    channels = 1 if audio_f.ndim == 1 else audio_f.shape[1]
    audio_i16 = (audio_f * 32768).clip(-32768, 32767).astype(np.int16)
    segment = AudioSegment(
        data=audio_i16.tobytes(),
        sample_width=audio_i16.dtype.itemsize,
        frame_rate=int(sample_rate),
        channels=channels,
    )
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


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_vctk(
    output_dir: Path,
    limit_mb: Optional[float],
    url: str,
    zip_path: Optional[Path],
    mic: str,
    speakers: Optional[list],
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    limit_bytes = int(limit_mb * 1024 * 1024) if limit_mb is not None else None
    wanted_speakers = set(speakers) if speakers else None
    exported = 0
    exported_bytes = 0
    source_bytes = 0
    skipped = 0

    with _open_zip(zip_path, url) as archive, \
            manifest_path.open("w", encoding="utf-8") as manifest:
        members = [
            m for m in archive.infolist()
            if not m.is_dir() and m.filename.lower().endswith(f"_{mic}.flac")
        ]
        if not members:
            raise ValueError(f"No *_{mic}.flac members found in the archive")
        progress = tqdm(total=limit_bytes, desc="Exporting WAV",
                        unit="B", unit_scale=True, unit_divisor=1024)
        try:
            for member in members:
                if limit_bytes is not None and exported_bytes >= limit_bytes:
                    break

                # wav48_silence_trimmed/p225/p225_001_mic1.flac
                utterance = Path(member.filename).stem      # p225_001_mic1
                speaker = utterance.split("_")[0]           # p225
                if wanted_speakers is not None and speaker not in wanted_speakers:
                    continue  # filtered out before any bytes are fetched

                flac_bytes = archive.read(member)
                source_bytes += member.compress_size
                progress.set_postfix_str(
                    f"src {source_bytes / (1 << 20):.0f}MiB", refresh=False)
                filename = f"{exported:06d}.wav"
                output_path = output_dir / filename
                try:
                    frames, out_rate = _convert_to_bgpt_wav(flac_bytes, output_path)
                except Exception as exc:
                    output_path.unlink(missing_ok=True)
                    print(f"\n  skipping unreadable clip {member.filename}: {exc}")
                    skipped += 1
                    continue

                clip_bytes = output_path.stat().st_size
                manifest.write(json.dumps({
                    "index": exported,
                    "file": filename,
                    "id": utterance,
                    "speaker": speaker,
                    "source_path": member.filename,
                    "duration_ms": frames * 1000 // out_rate,
                    "sample_rate": out_rate,
                    "bytes": clip_bytes,
                }, ensure_ascii=False) + "\n")
                exported += 1
                exported_bytes += clip_bytes
                progress.update(clip_bytes)
        finally:
            progress.close()

    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary:
        json.dump({
            "source_url": url,
            "source_zip": str(zip_path) if zip_path else None,
            "mic": mic,
            "speakers": sorted(wanted_speakers) if wanted_speakers else None,
            "num_samples": exported,
            "skipped_unreadable": skipped,
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
          f"across {exported} clips ({skipped} unreadable skipped; "
          f"{source_bytes / (1024 * 1024):.1f} MiB of source FLAC read)")
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download the VCTK corpus and convert it to mono 8-bit PCM WAV "
            "files at each clip's native sample rate (no resampling)."
        )
    )
    parser.add_argument("--output", type=Path, default=Path("datasets/vctk_wav"))
    parser.add_argument("--url", default=SOURCE_URL)
    parser.add_argument(
        "--zip", type=Path, default=None,
        help="Use a locally downloaded VCTK-Corpus-0.92.zip instead of ranged HTTP reads.",
    )
    parser.add_argument(
        "--mic", choices=["mic1", "mic2"], default="mic1",
        help="Which microphone recording to export per utterance (default: mic1).",
    )
    parser.add_argument(
        "--speaker", nargs="+", default=None, metavar="SPEAKER",
        help="Export only these speakers, e.g. p225 p226 (default: all).",
    )
    parser.add_argument(
        "--limit-mb", type=float, default=None,
        help=("Stop once this many MiB of WAV data have been exported "
              "(the clip crossing the limit is kept; default: whole dataset)."),
    )
    args = parser.parse_args()

    if args.limit_mb is not None and args.limit_mb <= 0:
        parser.error("--limit-mb must be positive")

    count = export_vctk(output_dir=args.output, limit_mb=args.limit_mb,
                        url=args.url, zip_path=args.zip, mic=args.mic,
                        speakers=args.speaker)
    print(f"Exported {count} WAV files to {args.output}")


if __name__ == "__main__":
    main()
