"""Audio preprocessing for compression.

Dataset loaders are registered by name; ``load_audio_samples`` auto-detects
the dataset from the path and dispatches to the matching loader.

Loaders return a flat list of audio payloads — anything accepted by
``audio_to_pydub_seg``: a file path, encoded bytes (FLAC/WAV/OGG/...),
an HF-style audio dict, or a pydub AudioSegment.

Adding a new dataset
--------------------
    from utils.audio_utils import register_audio_loader

    @register_audio_loader("my_dataset")
    def _load_my_dataset(path: str, n: Optional[int] = None) -> List[Any]:
        ...  # return list of audio payloads (paths, bytes, ...)
"""
from __future__ import annotations

import glob as _glob
import io
import os
import wave
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------

AudioLoader = Callable[[str, Optional[int]], List[Any]]

_AUDIO_LOADERS: Dict[str, AudioLoader] = {}


def register_audio_loader(name: str):
    """Decorator to register an audio dataset loader by name.

    Detection: *name* (hyphens normalised to underscores) must appear as a
    substring of the normalised dataset path.
    """
    def decorator(fn: AudioLoader) -> AudioLoader:
        _AUDIO_LOADERS[name] = fn
        return fn
    return decorator


def _find_audio_loader(path: str) -> AudioLoader:
    key = os.path.basename(os.path.normpath(path)).lower().replace("-", "_")
    for name, loader in _AUDIO_LOADERS.items():
        if name.replace("-", "_") in key:
            return loader
    raise ValueError(
        f"No audio loader registered for {path!r}.\n"
        f"Known datasets: {sorted(_AUDIO_LOADERS)}.\n"
        f"Register a new one with @register_audio_loader('name')."
    )


def load_audio_samples(path: str, n: Optional[int] = None) -> List[Any]:
    """Dispatch to the registered loader for the audio dataset at *path*."""
    return _find_audio_loader(path)(path, n)


# Backward-compat alias
load_audio_rows = load_audio_samples


# ---------------------------------------------------------------------------
# Shared low-level helpers used by built-in loaders
# ---------------------------------------------------------------------------

def _load_audio_dir(
    path: str,
    n: Optional[int],
    extensions: Sequence[str] = (".wav", ".flac", ".ogg", ".mp3"),
) -> List[str]:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Audio dataset directory not found: {path}")
    files: List[str] = []
    for ext in extensions:
        files.extend(_glob.glob(os.path.join(path, f"*{ext}")))
        files.extend(_glob.glob(os.path.join(path, f"*{ext.upper()}")))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"No audio files ({', '.join(extensions)}) found in {path}"
        )
    return files[:n] if n is not None else files


# ---------------------------------------------------------------------------
# Built-in dataset loaders
# ---------------------------------------------------------------------------

@register_audio_loader("peoples_speech")
def _load_peoples_speech(path: str, n: Optional[int] = None) -> List[Any]:
    import pandas as pd
    # Accept either a directory of parquets or a single parquet file.
    df = pd.read_parquet(path)
    if n is not None:
        df = df.iloc[:n]
    # The parquet schema nests the payload under "audio"; unwrap it here so
    # downstream code sees plain audio payloads (dicts with "bytes"/"path").
    return df["audio"].tolist()


@register_audio_loader("hf_wav_u8_8k_trimmed")
def _load_hf_wav_u8_8k_trimmed(path: str, n: Optional[int] = None) -> List[Any]:
    return _load_audio_dir(path + "/audio", n, extensions=(".wav",))


# ---------------------------------------------------------------------------
# bGPT preprocessing helpers
# ---------------------------------------------------------------------------

@dataclass
class AudioChunkRecord:
    """A single audio chunk with provenance, produced by chunk_audio_for_compression.

    Mirrors TextChunk in text_utils: data is the payload, sample_idx identifies
    the source clip, chunk_idx is the position within the clip.
    """
    data:       bytes
    sample_idx: int   # index into the worker's local sample list
    chunk_idx:  int   # 0-based position within this clip's chunks


def chunk_audio_for_compression(
    samples: List[Any],
    indices: List[int],
    chunk_ms: int = 1000,
) -> List[AudioChunkRecord]:
    """Split all audio clips in a worker shard into a flat list of AudioChunkRecords.

    Each sample is any payload accepted by audio_to_pydub_seg (file path,
    encoded bytes, HF audio dict, AudioSegment).
    Pure preprocessing step — no compression logic involved.
    Mirrors chunk_documents_for_compression in text_utils.
    """
    all_records: List[AudioChunkRecord] = []
    for local_idx, i in enumerate(indices):
        seg = audio_to_pydub_seg(samples[i])
        chunks = chunk_pydub_audio(seg, chunk_ms=chunk_ms)
        if not chunks:
            raise ValueError(f"Audio sample {i} produced no chunks")
        for chunk_idx, c in enumerate(chunks):
            all_records.append(AudioChunkRecord(
                data=c, sample_idx=local_idx, chunk_idx=chunk_idx))
    return all_records


# ---------------------------------------------------------------------------
# bGPT audio format helpers (8 kHz / mono / 8-bit unsigned PCM)
# ---------------------------------------------------------------------------

def audio_to_pydub_seg(data: Any):
    """Decode an audio payload → pydub AudioSegment at its native sample rate.

    Accepts (mirrors _image_to_pil in img_utils):
      - a pydub AudioSegment (returned as-is)
      - encoded bytes in any format soundfile reads (FLAC, WAV, OGG, ...)
      - a file path
      - a dict with "bytes", "path", or "array" + "sampling_rate" keys
        (the HF datasets Audio feature shape)

    Uses soundfile for decoding (no ffmpeg required) so it works in any env.
    Returns a pydub AudioSegment ready for resampling / format conversion.
    """
    try:
        import soundfile as sf
        from pydub import AudioSegment
    except ImportError as e:
        raise ImportError(
            "audio_to_pydub_seg requires 'soundfile' and 'pydub'") from e

    if isinstance(data, AudioSegment):
        return data
    if isinstance(data, (bytes, bytearray)):
        audio_f, sr = sf.read(io.BytesIO(bytes(data)))
        return _float_pcm_to_pydub_seg(audio_f, sr)
    if isinstance(data, (str, os.PathLike)):
        audio_f, sr = sf.read(data)
        return _float_pcm_to_pydub_seg(audio_f, sr)
    if isinstance(data, dict):
        if data.get("bytes") is not None:
            return audio_to_pydub_seg(bytes(data["bytes"]))
        if data.get("path") is not None:
            return audio_to_pydub_seg(data["path"])
        if data.get("array") is not None and data.get("sampling_rate"):
            return _float_pcm_to_pydub_seg(data["array"], data["sampling_rate"])
    raise TypeError(f"Unsupported audio data type: {type(data)!r}")


# Backward-compat alias (accepts any encoded bytes, not just FLAC)
flac_to_pydub_seg = audio_to_pydub_seg


def _float_pcm_to_pydub_seg(audio_f, sr):
    """Float PCM samples + sample rate → pydub AudioSegment (via 16-bit WAV)."""
    from pydub import AudioSegment

    audio_i16 = (np.asarray(audio_f) * 32768).clip(-32768,
                                                   32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(audio_i16.tobytes())
    buf.seek(0)
    return AudioSegment.from_wav(buf)


def chunk_pydub_audio(seg, chunk_ms: int = 1000) -> List[bytes]:
    """Resample a pydub AudioSegment to 8 kHz / mono / 8-bit unsigned PCM and
    split into fixed-duration chunks.

    Each returned chunk is a complete WAV file in bytes (44-byte header +
    8000 * (chunk_ms/1000) sample bytes), matching the format bGPT's audio
    model was trained on.
    """
    from pydub import AudioSegment  # local import so the rest of the module stays lightweight

    seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(1)
    chunks: List[bytes] = []
    for start in range(0, len(seg), chunk_ms):
        out = io.BytesIO()
        seg[start:start + chunk_ms].export(out, format='wav')
        chunks.append(out.getvalue())
    return chunks
