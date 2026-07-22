"""Canonical whole-sample data used by the non-neural baselines.

The RAC preparation scripts define the benchmark split.  This module always
reads the persisted eval split from a prepared database; it never makes a new
split.  Containers used only to persist a sample are removed before coding:
text is UTF-8, images are row-major RGB8, and audio is mono PCM_U8 at its
native sample rate.

Full base samples are not stored by the preparation scripts.  ``load_base_samples``
therefore reloads the original dataset recorded in ``meta.json``, selects the
recorded base indices, and checks the complementary records against the
persisted eval split.  This deliberately fails rather than silently training a
dictionary or constructing a delta reference from a different corpus.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import wave
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODALITIES = {"text", "image", "audio"}


@dataclass(frozen=True)
class BaselineSample:
    """One canonical, complete sample with split provenance."""

    sample_id: str
    modality: str
    data: bytes
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_index: int = -1
    split: str = "eval"
    source: Optional[str] = None
    # Position inside the persisted held-out split. Existing neural evaluators
    # use this position in sample_id; source_index preserves the original
    # pre-split dataset identity. Base references have no eval position.
    eval_position: Optional[int] = None

    @property
    def payload(self) -> bytes:
        """Alias used by codec and delta runners."""
        return self.data


@dataclass(frozen=True)
class BaselineSplit:
    """Test/calibration partition of the persisted held-out split."""

    modality: str
    meta: Mapping[str, Any]
    test: Tuple[BaselineSample, ...]
    calibration: Tuple[BaselineSample, ...] = ()

    @property
    def eval_samples(self) -> Tuple[BaselineSample, ...]:
        return self.test

    @property
    def calib_samples(self) -> Tuple[BaselineSample, ...]:
        return self.calibration


def load_database_meta(database: os.PathLike[str] | str) -> Dict[str, Any]:
    """Load and minimally validate ``prepare_*``'s ``meta.json``."""
    db = _database_dir(database)
    path = db / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared database metadata not found: {path}. "
            "Pass the --out directory produced by prepare_rac_data_*.py."
        )
    try:
        with path.open(encoding="utf-8") as handle:
            meta = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in prepared database metadata: {path}") from exc
    if not isinstance(meta, dict):
        raise TypeError(f"Prepared database metadata must be an object: {path}")
    if not isinstance(meta.get("dataset"), str) or not meta["dataset"]:
        raise ValueError(f"Prepared database metadata has no valid 'dataset': {path}")
    return meta


def infer_modality(
    database: os.PathLike[str] | str,
    modality: Optional[str] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> str:
    """Resolve a requested modality and reject metadata mismatches."""
    db = _database_dir(database)
    recorded = (meta or {}).get("modality")
    inferred: Optional[str]
    if recorded in _MODALITIES:
        inferred = str(recorded)
    elif (db / "eval_docs.jsonl").is_file():
        inferred = "text"
    else:
        inferred = None

    if modality is not None and modality not in _MODALITIES:
        raise ValueError(
            f"Unknown modality {modality!r}; expected one of {sorted(_MODALITIES)}"
        )
    if modality is not None and inferred is not None and modality != inferred:
        raise ValueError(
            f"Requested modality {modality!r}, but {db / 'meta.json'} records "
            f"{inferred!r}"
        )
    resolved = modality or inferred
    if resolved is None:
        raise ValueError(
            f"Cannot infer modality for {db}; pass --modality and ensure the "
            "database was produced by a current prepare script"
        )
    return resolved


def load_eval_split(
    database: os.PathLike[str] | str,
    modality: Optional[str] = None,
    n_samples: Optional[int] = None,
    calib_samples: int = 0,
) -> BaselineSplit:
    """Load test samples from a prepared database's persisted eval split.

    ``n_samples`` counts test samples.  When a calibration tail is requested,
    the first ``n_samples + calib_samples`` persisted records are loaded and the
    last ``calib_samples`` of those are excluded from test, matching the RAC
    evaluators.  With no ``n_samples`` limit, calibration is taken from the end
    of the complete persisted eval split.
    """
    if n_samples is not None and n_samples <= 0:
        raise ValueError(f"n_samples must be positive or None, got {n_samples}")
    if calib_samples < 0:
        raise ValueError(f"calib_samples must be non-negative, got {calib_samples}")

    db = _database_dir(database)
    meta = load_database_meta(db)
    resolved = infer_modality(db, modality, meta)
    raw = _load_persisted_eval(db, resolved)
    global_indices = _eval_global_indices(meta, len(raw), resolved)

    requested = None if n_samples is None else n_samples + calib_samples
    selected_count = len(raw) if requested is None else min(requested, len(raw))
    selected_positions = list(range(selected_count))
    if calib_samples and calib_samples >= len(selected_positions):
        raise ValueError(
            f"calib_samples={calib_samples} leaves no test samples; the selected "
            f"prefix contains only {len(selected_positions)} persisted eval samples"
        )
    split_at = len(selected_positions) - calib_samples if calib_samples else len(selected_positions)

    samples: List[BaselineSample] = []
    for local_idx in selected_positions:
        samples.append(
            _canonical_sample(
                resolved,
                raw[local_idx],
                global_indices[local_idx],
                split="eval",
                database=db,
                source_position=local_idx,
            )
        )
    test = tuple(replace(sample, split="test") for sample in samples[:split_at])
    calibration = tuple(
        replace(sample, split="calibration") for sample in samples[split_at:]
    )
    if not test:
        raise ValueError("The prepared eval split contains no test samples")
    return BaselineSplit(
        modality=resolved,
        meta=meta,
        test=test,
        calibration=calibration,
    )


def load_eval_samples(
    database: os.PathLike[str] | str,
    modality: Optional[str] = None,
    n_samples: Optional[int] = None,
    calib_samples: int = 0,
) -> List[BaselineSample]:
    """Convenience wrapper returning only the test portion."""
    return list(
        load_eval_split(
            database,
            modality=modality,
            n_samples=n_samples,
            calib_samples=calib_samples,
        ).test
    )


def load_base_samples(
    database: os.PathLike[str] | str,
    modality: Optional[str] = None,
) -> List[BaselineSample]:
    """Reload canonical full base samples selected during preparation.

    The persisted database contains base *chunks*, not full source samples.
    Consequently the original path in ``meta['dataset']`` must still be
    available.  The complementary source records are checked against the
    persisted eval data before any base sample is returned.
    """
    db = _database_dir(database)
    meta = load_database_meta(db)
    resolved = infer_modality(db, modality, meta)
    raw_eval = _load_persisted_eval(db, resolved)
    base_indices = _base_indices(meta, resolved)
    expected_total = len(base_indices) + len(raw_eval)
    dataset = _resolve_dataset_path(meta["dataset"], db)

    try:
        source_samples = _load_source_dataset(
            resolved, dataset, expected_total
        )
    except Exception as exc:
        message = (
            f"Cannot reconstruct full {resolved} base samples for prepared "
            f"database {db}. The prepare output stores base chunks only; restore "
            f"the original dataset recorded in meta.json ({meta['dataset']!r}) "
            "with the same ordering, or rebuild the database."
        )
        if isinstance(exc, FileNotFoundError):
            raise FileNotFoundError(message) from exc
        raise RuntimeError(message) from exc

    if len(source_samples) != expected_total:
        raise ValueError(
            f"Original dataset mismatch for {db}: expected exactly {expected_total} "
            f"prepared samples (base={len(base_indices)}, eval={len(raw_eval)}), "
            f"but reloading {meta['dataset']!r} yielded {len(source_samples)}. "
            "Restore the dataset version/order used by prepare."
        )

    base_set = set(base_indices)
    complement = [
        source_samples[index]
        for index in range(expected_total)
        if index not in base_set
    ]
    _validate_eval_complement(resolved, complement, raw_eval, db)

    result: List[BaselineSample] = []
    for source_index in base_indices:
        result.append(
            _canonical_sample(
                resolved,
                source_samples[source_index],
                source_index,
                split="base",
                database=db,
                source_position=source_index,
            )
        )
    return result


def canonical_image(path: os.PathLike[str] | str) -> Tuple[bytes, Dict[str, Any]]:
    """Decode an image into exact row-major RGB8 bytes and dimensions."""
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Image baselines require Pillow") from exc
    source = os.fspath(path)
    if not os.path.isfile(source):
        raise FileNotFoundError(f"Image source referenced by eval split is missing: {source}")
    try:
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            image.load()
            width, height = image.size
            payload = image.tobytes("raw", "RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image sample as RGB8: {source}") from exc
    expected = width * height * 3
    if len(payload) != expected:
        raise ValueError(
            f"Unexpected RGB8 payload size for {source}: {len(payload)} != {expected}"
        )
    return payload, {
        "width": width,
        "height": height,
        "channels": 3,
        "mode": "RGB",
        "pixel_format": "rgb8",
    }


def canonical_audio(wav_bytes: bytes, source: str = "<memory>") -> Tuple[bytes, Dict[str, Any]]:
    """Extract complete mono PCM_U8 frames and native sample rate from WAV."""
    if not isinstance(wav_bytes, bytes):
        raise TypeError(f"WAV sample must be bytes, got {type(wav_bytes)!r}")
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            compression = wav_file.getcomptype()
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            payload = wav_file.readframes(n_frames)
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"Invalid WAV sample: {source}") from exc
    if (channels, sample_width, compression) != (1, 1, "NONE"):
        raise ValueError(
            f"Expected mono PCM_U8 WAV for {source}; got channels={channels}, "
            f"sample_width={sample_width}, compression={compression!r}"
        )
    if sample_rate <= 0:
        raise ValueError(f"Invalid WAV sample rate {sample_rate} for {source}")
    if len(payload) != n_frames:
        raise ValueError(
            f"Truncated PCM payload for {source}: read {len(payload)} of {n_frames} frames"
        )
    return payload, {
        "sample_rate": sample_rate,
        "channels": 1,
        "sample_width": 1,
        "sample_format": "pcm_u8",
        "n_frames": n_frames,
    }


def _database_dir(database: os.PathLike[str] | str) -> Path:
    db = Path(database).expanduser()
    if not db.is_dir():
        raise FileNotFoundError(f"Prepared database directory not found: {db}")
    return db.resolve()


def _load_persisted_eval(db: Path, modality: str) -> List[Any]:
    if modality == "text":
        path = db / "eval_docs.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Prepared text eval split not found: {path}")
        records: List[str] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {path}:{line_number}; refusing to change "
                        "the prepared eval split"
                    ) from exc
                if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                    raise TypeError(
                        f"Expected {{'text': str}} at {path}:{line_number}"
                    )
                records.append(record["text"])
        if not records:
            raise ValueError(f"Prepared text eval split is empty: {path}")
        return records

    path = db / "eval_samples.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Prepared {modality} eval split not found: {path}")
    # This is a trusted, local artifact produced by prepare_rac_data_bgpt.py.
    with path.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError(f"Prepared {modality} eval split must be a non-empty list: {path}")
    if modality == "image":
        for position, value in enumerate(records):
            if not isinstance(value, (str, os.PathLike)):
                raise TypeError(
                    f"Image eval sample {position} in {path} must be a path, "
                    f"got {type(value)!r}"
                )
    else:
        for position, value in enumerate(records):
            if not isinstance(value, bytes):
                raise TypeError(
                    f"Audio eval sample {position} in {path} must be WAV bytes, "
                    f"got {type(value)!r}"
                )
    return list(records)


def _base_indices(meta: Mapping[str, Any], modality: str) -> List[int]:
    key = "base_doc_indices" if modality == "text" else "base_sample_indices"
    values = meta.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"Prepared metadata has no non-empty {key!r}")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError(f"Prepared metadata {key!r} must contain integer indices")
    if any(value < 0 for value in values) or len(set(values)) != len(values):
        raise ValueError(f"Prepared metadata {key!r} contains invalid/duplicate indices")
    if values != sorted(values):
        raise ValueError(f"Prepared metadata {key!r} must be sorted")
    return list(values)


def _eval_global_indices(
    meta: Mapping[str, Any], n_eval: int, modality: str
) -> List[int]:
    base = _base_indices(meta, modality)
    total = len(base) + n_eval
    if base[-1] >= total:
        raise ValueError(
            f"Prepared split metadata is inconsistent: base index {base[-1]} "
            f"is outside a {total}-sample base+eval universe"
        )
    base_set = set(base)
    indices = [index for index in range(total) if index not in base_set]
    if len(indices) != n_eval:
        raise ValueError("Prepared base/eval split metadata is inconsistent")
    return indices


def _sample_id(modality: str, source_index: int) -> str:
    prefix = {"text": "doc", "image": "image", "audio": "audio"}[modality]
    return f"{prefix}{source_index:06d}"


def _canonical_sample(
    modality: str,
    raw: Any,
    source_index: int,
    split: str,
    database: Path,
    source_position: int,
) -> BaselineSample:
    eval_position = source_position if split != "base" else None
    identity_index = source_index if eval_position is None else eval_position
    sample_id = _sample_id(modality, identity_index)
    if modality == "text":
        if not isinstance(raw, str):
            raise TypeError(f"Text sample {source_position} must be str")
        data = raw.encode("utf-8")
        metadata: Dict[str, Any] = {"encoding": "utf-8"}
        source = str(database / "eval_docs.jsonl") if split != "base" else None
    elif modality == "image":
        path = _resolve_sample_path(os.fspath(raw), database)
        data, metadata = canonical_image(path)
        source = str(path)
    else:
        source = str(database / "eval_samples.pkl") if split != "base" else None
        data, metadata = canonical_audio(raw, f"{source}[{source_position}]")

    metadata = dict(metadata)
    metadata.update(
        {
            "modality": modality,
            "sample_id": sample_id,
            "source_index": source_index,
            "eval_position": eval_position,
        }
    )
    return BaselineSample(
        sample_id=sample_id,
        modality=modality,
        data=data,
        metadata=metadata,
        source_index=source_index,
        split=split,
        source=source,
        eval_position=eval_position,
    )


def _candidate_paths(path: str, database: Path) -> List[Path]:
    original = Path(path).expanduser()
    if original.is_absolute():
        return [original]
    candidates = [original, _REPO_ROOT / original]
    for ancestor in database.parents:
        candidates.append(ancestor / original)
    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        marker = os.path.abspath(os.fspath(candidate))
        if marker not in seen:
            seen.add(marker)
            unique.append(candidate)
    return unique


def _resolve_sample_path(path: str, database: Path) -> Path:
    for candidate in _candidate_paths(path, database):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Image path persisted in {database / 'eval_samples.pkl'} is missing: "
        f"{path!r}. Restore the source images; the prepare output stores paths, "
        "not image bytes."
    )


def _resolve_dataset_path(path: str, database: Path) -> str:
    for candidate in _candidate_paths(path, database):
        if candidate.exists():
            return os.fspath(candidate)
    # A non-local Hugging Face identifier can still be handled by the registered
    # loader.  Local-looking missing paths will receive the clearer wrapper in
    # load_base_samples.
    return path


def _load_source_dataset(modality: str, path: str, n: int) -> List[Any]:
    if modality == "text":
        return _load_text_source(path, n)
    if modality == "image":
        from utils.img_utils import load_image_files
        return list(load_image_files(path, n))
    from utils.audio_utils import load_audio_samples
    return list(load_audio_samples(path, n))


def _load_text_source(path: str, n: int) -> List[str]:
    """Lightweight local JSONL loader, with registered-loader fallback."""
    normal = os.path.normpath(path).replace(os.sep, "/").lower()
    if os.path.isfile(path) and path.lower().endswith(".jsonl"):
        if "codeparrot_github_code" in normal:
            key = "code"
        elif "arxiv_tex" in normal:
            key = "text"
        else:
            key = "text"
        result: List[str] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if len(result) >= n:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # Match utils.text_utils._load_jsonl, which skips malformed rows.
                    continue
                if not isinstance(record, dict) or record.get(key) is None:
                    continue
                text = str(record[key]).strip()
                if text:
                    result.append(text)
        return result

    try:
        # Lazy import keeps eval-only text/image/audio baselines usable in a
        # minimal environment without torch/datasets.
        from utils.text_utils import load_text_documents
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Reloading the text base dataset {path!r} needs the dependencies "
            f"of utils.text_utils (missing {exc.name!r})"
        ) from exc
    return list(load_text_documents(path, num_documents=n))


def _validate_eval_complement(
    modality: str,
    source_eval: Sequence[Any],
    persisted_eval: Sequence[Any],
    database: Path,
) -> None:
    if len(source_eval) != len(persisted_eval):
        raise ValueError("Prepared eval complement has an unexpected length")
    mismatch: Optional[int] = None
    if modality in {"text", "audio"}:
        for index, (source, persisted) in enumerate(zip(source_eval, persisted_eval)):
            if source != persisted:
                mismatch = index
                break
    else:
        for index, (source, persisted) in enumerate(zip(source_eval, persisted_eval)):
            source_path = _resolve_sample_path(os.fspath(source), database)
            persisted_path = _resolve_sample_path(os.fspath(persisted), database)
            if source_path != persisted_path:
                mismatch = index
                break
    if mismatch is not None:
        raise ValueError(
            f"Original dataset/order no longer matches the persisted {modality} "
            f"eval split at eval position {mismatch}; refusing to construct base "
            "samples from a different corpus"
        )
