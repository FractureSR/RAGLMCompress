"""Load and chunk preprocessed bGPT audio.

Dataset loaders are registered by name. Every loader must return complete,
uncompressed mono, 8-bit PCM WAV files as ``bytes``, at each clip's **native**
sample rate (we no longer resample — dataset download scripts only convert
channels and bit depth, not the sample rate).

Because sample rate now varies clip-to-clip, compression/retrieval units are
chunked by a fixed **byte** count (``audio_chunk_bytes``), not a fixed
duration — the same duration would span a different number of bytes per clip
and could overflow the model's context window.

Adding a new dataset
--------------------
    from utils.audio_utils import register_audio_loader

    @register_audio_loader("my_dataset")
    def _load_my_dataset(path: str, n: Optional[int] = None) -> List[bytes]:
        ...  # return complete preprocessed WAV files as bytes
"""
from __future__ import annotations

import glob as _glob
import io
import os
import pickle
import wave
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence


CHANNELS = 1
SAMPLE_WIDTH = 1


AudioLoader = Callable[[str, Optional[int]], List[bytes]]

_AUDIO_LOADERS: Dict[str, AudioLoader] = {}


def register_audio_loader(name: str):
    """Register a loader selected by matching *name* against the dataset path.

    *name* (hyphens normalised to underscores) must appear as a substring of
    the whole normalised dataset path, so multi-segment names like
    ``"vctk"`` match a path ``datasets/vctk_wav`` — not just the final
    basename. When several names match, the longest (most specific) one wins.
    """
    def decorator(fn: AudioLoader) -> AudioLoader:
        _AUDIO_LOADERS[name] = fn
        return fn
    return decorator


def _find_audio_loader(path: str) -> AudioLoader:
    key = os.path.normpath(path).replace(os.sep, "/").lower().replace("-", "_")
    best_loader: Optional[AudioLoader] = None
    best_len = -1
    for name, loader in _AUDIO_LOADERS.items():
        needle = name.replace(os.sep, "/").lower().replace("-", "_")
        if needle in key and len(needle) > best_len:
            best_loader, best_len = loader, len(needle)
    if best_loader is not None:
        return best_loader
    raise ValueError(
        f"No audio loader registered for {path!r}.\n"
        f"Known datasets: {sorted(_AUDIO_LOADERS)}.\n"
        f"Register a new one with @register_audio_loader('name')."
    )


def load_audio_samples(path: str, n: Optional[int] = None) -> List[bytes]:
    """Dispatch to the registered loader for the audio dataset at *path*."""
    return _find_audio_loader(path)(path, n)


def _load_wav_dir(path: str, n: Optional[int]) -> List[bytes]:
    """Read and validate preprocessed WAV files in filename order."""
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


@register_audio_loader("eval_samples.pkl")
def load_rac_eval_samples(path: str, n: Optional[int] = None) -> List[bytes]:
    """Load preprocessed WAV bytes persisted by prepare_rac_data_bgpt."""
    pkl_path = os.path.join(path, "eval_samples.pkl") if os.path.isdir(path) else path
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"RAC eval pickle not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)
    selected = samples[:n] if n is not None else samples
    for sample_idx, sample in enumerate(selected):
        if not isinstance(sample, bytes):
            raise TypeError(
                f"RAC eval sample {sample_idx} must be bytes, "
                f"got {type(sample)!r}; rebuild the database"
            )
        _validate_wav(sample, f"{pkl_path}[{sample_idx}]")
    return selected


@register_audio_loader("ljspeech_wav")
def _load_ljspeech(path: str, n: Optional[int] = None) -> List[bytes]:
    return _load_wav_dir(path, n)


@register_audio_loader("vctk/p225")
def _load_vctk(path: str, n: Optional[int] = None) -> List[bytes]:
    return _load_wav_dir(path, n)


def _validate_wav(data: bytes, source: str) -> None:
    if not isinstance(data, bytes):
        raise TypeError(f"Audio sample must be bytes, got {type(data)!r}")
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            actual = (
                wav_file.getnchannels(),
                wav_file.getsampwidth(),
                wav_file.getcomptype(),
            )
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"Invalid WAV file: {source}") from exc

    # Sample rate is intentionally unchecked: clips keep their native rate.
    expected = (CHANNELS, SAMPLE_WIDTH, "NONE")
    if actual != expected:
        raise ValueError(
            f"Expected mono 8-bit PCM WAV for {source}, "
            f"got channels/sample-width/compression={actual}"
        )


# ---------------------------------------------------------------------------
# bGPT preprocessing helpers
# ---------------------------------------------------------------------------

@dataclass
class AudioChunkRecord:
    """A single audio chunk with provenance, produced by chunk_audio_for_compression.

    ``data`` contains raw unsigned 8-bit PCM frames without a WAV header.
    ``sample_idx`` identifies the source clip and ``chunk_idx`` is its position.
    ``sample_rate`` is the source clip's native rate (frames == bytes here since
    channels/sample-width are fixed at 1, so it isn't needed to size the chunk —
    only to rebuild a playable WAV header, e.g. for --save-decompressed).
    """
    data:        bytes
    sample_idx:  int   # index into the worker's local sample list
    chunk_idx:   int   # 0-based position within this clip's chunks
    sample_rate: int   # native sample rate of the source clip


def chunk_audio_for_compression(
    samples: Sequence[bytes],
    indices: List[int],
    audio_chunk_bytes: int,
) -> List[AudioChunkRecord]:
    """Split all audio clips in a worker shard into a flat list of AudioChunkRecords.

    Each input is a mono 8-bit PCM WAV at its own native sample rate. This
    function strips the container and returns chunks containing only raw PCM
    frames, exactly ``audio_chunk_bytes`` bytes each (one byte == one frame
    for mono 8-bit PCM). Chunking is byte-count based — not duration based —
    because clips no longer share a common sample rate: a fixed duration
    would span a different number of bytes per clip and could overflow the
    model's context window. Mirrors chunk_documents_for_compression in
    text_utils.
    """
    if audio_chunk_bytes <= 0:
        raise ValueError(
            f"audio_chunk_bytes must be positive, got {audio_chunk_bytes}")

    all_records: List[AudioChunkRecord] = []
    for local_idx, i in enumerate(indices):
        data = samples[i]
        _validate_wav(data, f"sample {i}")
        sample_chunks = 0
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            rate = wav_file.getframerate()
            while True:
                frames = wav_file.readframes(audio_chunk_bytes)
                if not frames:
                    break
                all_records.append(AudioChunkRecord(
                    data=frames,
                    sample_idx=local_idx,
                    chunk_idx=sample_chunks,
                    sample_rate=rate,
                ))
                sample_chunks += 1
        if sample_chunks == 0:
            raise ValueError(f"Audio sample {i} produced no chunks")
    return all_records


# ---------------------------------------------------------------------------
# bGPT audio format helpers (native sample rate / mono / 8-bit unsigned PCM)
# ---------------------------------------------------------------------------

def pcm_payload_to_wav(payload: bytes, sample_rate: int) -> bytes:
    """Wrap raw mono PCM_U8 frames in a WAV container at the given sample rate."""
    out = io.BytesIO()
    with wave.open(out, "wb") as wav_file:
        wav_file.setframerate(sample_rate)
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.writeframes(payload)
    return out.getvalue()
