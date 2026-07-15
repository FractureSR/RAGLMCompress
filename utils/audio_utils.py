"""Load and chunk preprocessed bGPT audio.

All input clips must be uncompressed 8 kHz, mono, 8-bit PCM WAV files. Dataset
download scripts own decoding, resampling, channel conversion, and WAV export.
"""
from __future__ import annotations

import glob as _glob
import io
import os
import pickle
import wave
from dataclasses import dataclass
from typing import List, Optional, Sequence


SAMPLE_RATE = 8000
CHANNELS = 1
SAMPLE_WIDTH = 1


def load_audio_samples(path: str, n: Optional[int] = None) -> List[bytes]:
    """Read preprocessed WAV files from a directory in filename order."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Audio dataset directory not found: {path}")

    files = sorted(set(
        _glob.glob(os.path.join(path, "*.wav"))
        + _glob.glob(os.path.join(path, "*.WAV"))
    ))
    if not files:
        raise FileNotFoundError(f"No WAV files found in {path}")

    selected = files[:n] if n is not None else files
    samples: List[bytes] = []
    for file_path in selected:
        with open(file_path, "rb") as f:
            data = f.read()
        _validate_wav(data, file_path)
        samples.append(data)
    return samples


def load_rac_eval_samples(path: str, n: Optional[int] = None) -> List[bytes]:
    """Load preprocessed WAV bytes persisted by prepare_rac_data_bgpt."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"RAC eval pickle not found: {path}")
    with open(path, "rb") as f:
        samples = pickle.load(f)
    selected = samples[:n] if n is not None else samples
    for sample_idx, sample in enumerate(selected):
        if not isinstance(sample, bytes):
            raise TypeError(
                f"RAC eval sample {sample_idx} must be bytes, "
                f"got {type(sample)!r}; rebuild the database"
            )
        _validate_wav(sample, f"{path}[{sample_idx}]")
    return selected


def _validate_wav(data: bytes, source: str) -> None:
    if not isinstance(data, bytes):
        raise TypeError(f"Audio sample must be bytes, got {type(data)!r}")
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            actual = (
                wav_file.getframerate(),
                wav_file.getnchannels(),
                wav_file.getsampwidth(),
                wav_file.getcomptype(),
            )
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"Invalid WAV file: {source}") from exc

    expected = (SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH, "NONE")
    if actual != expected:
        raise ValueError(
            f"Expected 8 kHz mono 8-bit PCM WAV for {source}, "
            f"got rate/channels/sample-width/compression={actual}"
        )


# ---------------------------------------------------------------------------
# bGPT preprocessing helpers
# ---------------------------------------------------------------------------

@dataclass
class AudioChunkRecord:
    """A single audio chunk with provenance, produced by chunk_audio_for_compression.

    ``data`` contains raw unsigned 8-bit PCM frames without a WAV header.
    ``sample_idx`` identifies the source clip and ``chunk_idx`` is its position.
    """
    data:       bytes
    sample_idx: int   # index into the worker's local sample list
    chunk_idx:  int   # 0-based position within this clip's chunks


def chunk_audio_for_compression(
    samples: Sequence[bytes],
    indices: List[int],
    chunk_ms: int = 1000,
) -> List[AudioChunkRecord]:
    """Split all audio clips in a worker shard into a flat list of AudioChunkRecords.

    Each input is already an 8 kHz mono 8-bit PCM WAV. This function strips the
    container and returns chunks containing only raw PCM frames.
    Mirrors chunk_documents_for_compression in text_utils.
    """
    if chunk_ms <= 0:
        raise ValueError(f"chunk_ms must be positive, got {chunk_ms}")
    frames_per_chunk = SAMPLE_RATE * chunk_ms // 1000
    if frames_per_chunk <= 0:
        raise ValueError(f"chunk_ms={chunk_ms} produces an empty chunk")

    all_records: List[AudioChunkRecord] = []
    for local_idx, i in enumerate(indices):
        data = samples[i]
        _validate_wav(data, f"sample {i}")
        sample_chunks = 0
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            while True:
                frames = wav_file.readframes(frames_per_chunk)
                if not frames:
                    break
                all_records.append(AudioChunkRecord(
                    data=frames,
                    sample_idx=local_idx,
                    chunk_idx=sample_chunks,
                ))
                sample_chunks += 1
        if sample_chunks == 0:
            raise ValueError(f"Audio sample {i} produced no chunks")
    return all_records


# ---------------------------------------------------------------------------
# bGPT audio format helpers (8 kHz / mono / 8-bit unsigned PCM)
# ---------------------------------------------------------------------------

def pcm_payload_to_wav(payload: bytes) -> bytes:
    """Wrap raw 8 kHz mono PCM_U8 frames in a WAV container."""
    out = io.BytesIO()
    with wave.open(out, "wb") as wav_file:
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.writeframes(payload)
    return out.getvalue()
