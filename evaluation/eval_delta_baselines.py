"""Whole-sample delta-codec baselines over a prepared RAC split.

The target set is *only* the held-out split persisted by the corresponding
``prepare_rac_data_*`` command.  Full base samples are reconstructed from the
dataset path and base indices recorded in ``meta.json``.  Retrieval still uses
the prepared chunk index, but a returned chunk is mapped through its provenance
to the full source document/image/clip before a patch is made.

For every target and delta codec this evaluator writes two rows:

``patch-only``
    The best retrieved reference is mandatory.  Its cost is the patch artifact
    plus a fixed-width full-base-sample id and the one-bit stop symbol.

``adaptive``
    The above choice competes with standalone zstd --ultra -22.  Falling back
    to zstd transmits only the one-bit stop symbol.

No model/window chunking is performed: target and reference payloads are the
canonical whole-sample bytes supplied by :mod:`utils.baseline_data`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# Keep ``python evaluation/eval_delta_baselines.py --help`` usable in a light
# environment.  Dataset, retriever, and codec imports intentionally live below
# ``parse_args`` / inside the functions that need them.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEFAULT_CODECS = ("zstd-patch", "bsdiff", "open-vcdiff")

CODEC_SETTINGS = {
    "zstd-patch": "--ultra -22 -T1 --patch-from",
    "bsdiff": "reference CLI defaults",
    "open-vcdiff": "standard VCDIFF with -target_matches",
    "zstd": "--ultra -22 -T1",
}


@dataclass
class DeltaResult:
    """One CSV row (one target, delta codec, and selection policy)."""

    sample_id: str
    eval_position: int
    source_index: int
    modality: str
    codec: str
    artifact_codec: str
    artifact_codec_version: str
    artifact_codec_settings: str
    policy: str
    input_sha256: str
    original_bytes: int
    source_pixels: int
    artifact_bytes: int
    artifact_bits: int
    side_info_bits: float
    total_bits: float
    bpb: float
    bpp: float
    ratio: float
    reference_id: Optional[int]
    reference_source_index: Optional[int]
    reference_sha256: str
    candidate_rank: Optional[int]
    n_candidates: int
    n_base_references: int
    used_reference: int
    retrieval_s: float
    selection_s: float
    decode_s: float
    roundtrip_ok: int
    error: str = ""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Whole-sample delta baselines on a prepared RAC held-out split"
        )
    )
    parser.add_argument(
        "--database",
        required=True,
        help="prepare_rac_data_llm/bgpt --out directory",
    )
    parser.add_argument(
        "--modality",
        choices=("text", "image", "audio"),
        default=None,
        help="normally inferred from the prepared database",
    )
    parser.add_argument(
        "--codecs",
        default=",".join(DEFAULT_CODECS),
        help="comma-separated delta codecs: zstd-patch,bsdiff,open-vcdiff",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=16,
        help="number of unique full-base references retained after chunk retrieval",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="cap on the persisted held-out eval split (default: all)",
    )
    parser.add_argument(
        "--calib-samples",
        "--calib-docs",
        dest="calib_samples",
        type=int,
        default=0,
        help=(
            "exclude this many samples from the selected eval tail, matching "
            "the RAC calibration/test partition"
        ),
    )
    parser.add_argument(
        "--candidate-policy",
        choices=("retrieve", "all"),
        default="retrieve",
        help=(
            "retrieve: map prepared chunk hits to top-m full samples (main protocol); "
            "all: exact oracle over every full base sample (diagnostic, ignores --m)"
        ),
    )
    parser.add_argument(
        "--embed-model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="text dense/hybrid retriever model (ignored for BM25)",
    )
    parser.add_argument(
        "--embed-device",
        default=None,
        help="device for a text dense/hybrid retriever",
    )
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument(
        "--retrieval-pool",
        type=int,
        default=None,
        help=(
            "initial number of chunk hits to inspect; it grows automatically "
            "until --m unique source samples are found"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="per external codec command timeout in seconds",
    )
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help="temporary-file parent for external codecs (default: system temp)",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="record codec failures in CSV and continue (default: fail loudly)",
    )
    parser.add_argument("--output", default=None, metavar="CSV")
    return parser


def _parse_codec_names(value: str) -> List[str]:
    aliases = {"vcdiff": "open-vcdiff", "zstd_patch": "zstd-patch"}
    names: List[str] = []
    for raw in value.split(","):
        name = aliases.get(raw.strip().lower(), raw.strip().lower())
        if name and name not in names:
            names.append(name)
    if not names:
        raise ValueError("--codecs must contain at least one codec")
    unsupported = [name for name in names if name not in DEFAULT_CODECS]
    if unsupported:
        raise ValueError(
            f"unsupported delta codec(s): {unsupported}; "
            f"choose from {list(DEFAULT_CODECS)}"
        )
    return names


def _load_meta(database: str) -> Tuple[str, dict]:
    meta_path = os.path.join(database, "meta.json")
    try:
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"prepared database metadata not found: {meta_path}"
        ) from exc

    if os.path.isfile(os.path.join(database, "base_chunks.json")):
        inferred = "text"
    elif os.path.isfile(os.path.join(database, "base_chunks.pkl")):
        inferred = meta.get("modality")
    else:
        raise FileNotFoundError(
            f"{database!r} has neither base_chunks.json nor base_chunks.pkl"
        )
    if inferred not in {"text", "image", "audio"}:
        raise ValueError(f"cannot infer a supported modality from {database!r}")
    return inferred, meta


def _load_chunk_provenance(database: str, modality: str) -> List[int]:
    """Return global source-sample index for each retriever chunk id."""

    if modality == "text":
        path = os.path.join(database, "base_chunks.json")
        with open(path, encoding="utf-8") as handle:
            records = json.load(handle)
        provenance_key = "doc_idx"
    else:
        path = os.path.join(database, "base_chunks.pkl")
        with open(path, "rb") as handle:
            records = pickle.load(handle)
        provenance_key = "sample_idx"

    if not isinstance(records, list) or not records:
        raise ValueError(f"no base chunks in {path}")
    provenance: List[int] = []
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(
                f"base chunk {position} in {path} is not a mapping: "
                f"{type(record)!r}"
            )
        declared_id = int(record.get("id", position))
        if declared_id != position:
            raise ValueError(
                f"base chunk id {declared_id} is not its retriever position "
                f"{position}; rebuild the database"
            )
        try:
            provenance.append(int(record[provenance_key]))
        except KeyError as exc:
            raise ValueError(
                f"base chunk {position} lacks provenance field "
                f"{provenance_key!r}; rebuild the database"
            ) from exc
    return provenance


def _load_retriever(
    database: str,
    modality: str,
    meta: Mapping[str, Any],
    embed_model: str,
    embed_device: Optional[str],
    embed_batch_size: int,
):
    """Load the exact chunk retriever persisted by ``prepare_*`` lazily."""

    try:
        if modality == "text":
            from utils.text_utils import make_text_retriever

            retriever = make_text_retriever(
                embed_model=embed_model,
                device=embed_device,
                rrf_k=int(meta.get("rrf_k", 60)),
                batch_size=embed_batch_size,
                query_batch_size=1,
                signals=str(meta.get("signals", "bm25")),
            )
        else:
            from utils.bgpt_codec_utils import make_bgpt_retriever

            retriever = make_bgpt_retriever(
                signals=str(meta.get("signals", "bm25")),
                kgram=int(meta.get("kgram", 4)),
                rrf_k=int(meta.get("rrf_k", 60)),
            )
        retriever.load(os.path.join(database, "retriever"))
        return retriever
    except (ModuleNotFoundError, ImportError) as exc:
        raise RuntimeError(
            "retrieval dependencies are unavailable. Install the repository's "
            "retrieval environment (torch/transformers/bm25s/faiss as required), "
            "or use --candidate-policy all for the dependency-light diagnostic. "
            f"Original import error: {exc}"
        ) from exc


def _query_for_retriever(
    sample: Any, modality: str, meta: Optional[Mapping[str, Any]] = None
) -> Any:
    data = bytes(sample.data)
    if modality == "text":
        # The canonical text payload is exact UTF-8.  Strict decoding catches a
        # data-contract violation instead of changing retrieval text silently.
        return data.decode("utf-8", errors="strict")
    if modality == "image":
        # The prepared byte retriever was built over BGR, bottom-up, row-aligned
        # BMP patch payloads.  The standalone/delta codec payload is the logical
        # RGB image, so convert only the *retrieval query* back to the exact
        # representation used by prepare_rac_data_bgpt.  Compression itself
        # remains whole-image RGB8.
        unit = (meta or {}).get("unit")
        if not isinstance(unit, Mapping):
            raise ValueError("prepared image metadata has no patch unit")
        try:
            patch_width = int(unit["patch_width"])
            patch_height = int(unit["patch_height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "prepared image metadata has invalid patch dimensions"
            ) from exc
        try:
            width = int(sample.metadata["width"])
            height = int(sample.metadata["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("canonical image sample has invalid dimensions") from exc
        from PIL import Image
        from utils.img_utils import _pil_to_bmp_payload

        try:
            image = Image.frombytes("RGB", (width, height), data)
        except ValueError as exc:
            raise ValueError("canonical image payload does not match its dimensions") from exc
        padded_width = ((width + patch_width - 1) // patch_width) * patch_width
        padded_height = ((height + patch_height - 1) // patch_height) * patch_height
        padded = Image.new("RGB", (padded_width, padded_height))
        padded.paste(image, (0, 0))
        query_parts = []
        for y in range(0, padded_height, patch_height):
            for x in range(0, padded_width, patch_width):
                patch = padded.crop((x, y, x + patch_width, y + patch_height))
                query_parts.append(_pil_to_bmp_payload(patch))
        if not query_parts:
            raise ValueError("image retrieval query produced no patches")
        return b"".join(query_parts)
    return data


def _map_chunk_hits_to_references(
    hits: Iterable[Tuple[int, float]],
    chunk_provenance: Sequence[int],
    source_to_reference: Mapping[int, int],
    m: int,
) -> List[int]:
    """Map ranked chunk hits to ranked, de-duplicated full-reference ids."""

    references: List[int] = []
    seen = set()
    for chunk_id, _score in hits:
        chunk_id = int(chunk_id)
        if chunk_id < 0 or chunk_id >= len(chunk_provenance):
            raise IndexError(
                f"retriever returned chunk id {chunk_id}, but database has "
                f"{len(chunk_provenance)} chunks"
            )
        source_index = int(chunk_provenance[chunk_id])
        if source_index not in source_to_reference:
            raise ValueError(
                f"retrieved chunk {chunk_id} points to source sample "
                f"{source_index}, which is absent from the prepared base split"
            )
        reference_id = int(source_to_reference[source_index])
        if reference_id in seen:
            continue
        seen.add(reference_id)
        references.append(reference_id)
        if len(references) >= m:
            break
    return references


def _retrieve_reference_ids(
    retriever: Any,
    query: Any,
    chunk_provenance: Sequence[int],
    source_to_reference: Mapping[int, int],
    m: int,
    initial_pool: Optional[int],
) -> List[int]:
    """Grow the ranked chunk prefix until it contains ``m`` unique sources."""

    n_chunks = len(chunk_provenance)
    # Some prepared base samples can have no retained chunk (empty text, or a
    # media sample whose chunks were all shorter than the fixed condition
    # length). They remain addressable full references for ``all`` diagnostics,
    # but a chunk retriever cannot propose them.
    represented_sources = {
        int(source_index)
        for source_index in chunk_provenance
        if int(source_index) in source_to_reference
    }
    wanted = min(m, len(represented_sources))
    if wanted <= 0:
        return []
    top_k = initial_pool or max(50, 8 * wanted)
    top_k = min(max(top_k, wanted), n_chunks)

    while True:
        # Explicit ``pool`` makes the scorer contribution deterministic for this
        # ranked prefix.  It mirrors Retriever's normal 5x over-retrieval.
        scorer_pool = min(n_chunks, max(5 * top_k, 50))
        hits = retriever.retrieve(query, top_k=top_k, pool=scorer_pool)
        references = _map_chunk_hits_to_references(
            hits, chunk_provenance, source_to_reference, wanted
        )
        if len(references) >= wanted or top_k >= n_chunks:
            break
        top_k = min(n_chunks, max(top_k + 1, top_k * 2))

    if not references:
        raise RuntimeError("retrieval produced no usable full-sample references")
    if len(references) < wanted:
        raise RuntimeError(
            f"retrieval found only {len(references)} unique base samples, "
            f"but {wanted} were requested"
        )
    return references


def _artifact_bytes(encoded: Any) -> bytes:
    artifact = getattr(encoded, "artifact", encoded)
    if isinstance(artifact, bytearray):
        artifact = bytes(artifact)
    if not isinstance(artifact, bytes):
        raise TypeError(
            "codec encode result must expose a bytes .artifact; got "
            f"{type(artifact)!r}"
        )
    return artifact


def _decoded_bytes(decoded: Any) -> bytes:
    payload = getattr(decoded, "payload", decoded)
    if isinstance(payload, bytearray):
        payload = bytes(payload)
    if not isinstance(payload, bytes):
        raise TypeError(
            "codec decode result must expose a bytes .payload; got "
            f"{type(payload)!r}"
        )
    return payload


def _encode(codec: Any, target: bytes, reference: Optional[bytes] = None) -> Any:
    metadata = {"modality": "bytes"}
    if reference is None:
        return codec.encode(target, metadata=metadata)
    return codec.encode(target, reference, metadata=metadata)


def _decode(
    codec: Any,
    artifact: bytes,
    reference: Optional[bytes] = None,
    *,
    target_size: Optional[int] = None,
) -> Any:
    metadata = {"modality": "bytes"}
    if target_size is not None:
        metadata["target_size"] = int(target_size)
    if reference is None:
        return codec.decode(artifact, metadata=metadata)
    return codec.decode(artifact, reference, metadata=metadata)


def _make_row(
    *,
    sample: Any,
    modality: str,
    codec_name: str,
    artifact_codec: str,
    artifact_codec_version: str,
    policy: str,
    artifact: bytes,
    side_info_bits: float,
    reference_id: Optional[int],
    reference_source_index: Optional[int],
    reference_sha256: str,
    candidate_rank: Optional[int],
    n_candidates: int,
    n_base_references: int,
    retrieval_s: float,
    selection_s: float,
    decode_s: float,
    roundtrip_ok: bool,
) -> DeltaResult:
    original_bytes = len(sample.data)
    source_pixels = 0
    if modality == "image":
        source_pixels = int(sample.metadata["width"]) * int(
            sample.metadata["height"]
        )
    artifact_bits = len(artifact) * 8
    total_bits = artifact_bits + float(side_info_bits)
    eval_position = (
        int(sample.eval_position) if sample.eval_position is not None else -1
    )
    return DeltaResult(
        sample_id=str(sample.sample_id),
        eval_position=eval_position,
        source_index=int(sample.source_index),
        modality=modality,
        codec=codec_name,
        artifact_codec=artifact_codec,
        artifact_codec_version=artifact_codec_version,
        artifact_codec_settings=CODEC_SETTINGS.get(artifact_codec, "unknown"),
        policy=policy,
        input_sha256=hashlib.sha256(bytes(sample.data)).hexdigest(),
        original_bytes=original_bytes,
        source_pixels=source_pixels,
        artifact_bytes=len(artifact),
        artifact_bits=artifact_bits,
        side_info_bits=float(side_info_bits),
        total_bits=total_bits,
        bpb=total_bits / max(original_bytes, 1),
        bpp=(total_bits / source_pixels) if source_pixels else math.nan,
        ratio=(original_bytes * 8.0 / total_bits) if total_bits else math.inf,
        reference_id=reference_id,
        reference_source_index=reference_source_index,
        reference_sha256=reference_sha256,
        candidate_rank=candidate_rank,
        n_candidates=n_candidates,
        n_base_references=n_base_references,
        used_reference=int(reference_id is not None),
        retrieval_s=retrieval_s,
        selection_s=selection_s,
        decode_s=decode_s,
        roundtrip_ok=int(roundtrip_ok),
    )


def _error_rows(
    sample: Any,
    modality: str,
    codec_name: str,
    n_candidates: int,
    n_base_references: int,
    retrieval_s: float,
    error: BaseException,
) -> List[DeltaResult]:
    return [
        _error_row(
            sample,
            modality,
            codec_name,
            policy,
            n_candidates,
            n_base_references,
            retrieval_s,
            error,
        )
        for policy in ("patch-only", "adaptive")
    ]


def _error_row(
    sample: Any,
    modality: str,
    codec_name: str,
    policy: str,
    n_candidates: int,
    n_base_references: int,
    retrieval_s: float,
    error: BaseException,
) -> DeltaResult:
    message = f"{type(error).__name__}: {error}"
    return DeltaResult(
        sample_id=str(sample.sample_id),
        eval_position=(
            int(sample.eval_position) if sample.eval_position is not None else -1
        ),
        source_index=int(sample.source_index),
        modality=modality,
        codec=codec_name,
        artifact_codec="",
        artifact_codec_version="",
        artifact_codec_settings="",
        policy=policy,
        input_sha256=hashlib.sha256(bytes(sample.data)).hexdigest(),
        original_bytes=len(sample.data),
        source_pixels=(
            int(sample.metadata["width"]) * int(sample.metadata["height"])
            if modality == "image"
            else 0
        ),
        artifact_bytes=0,
        artifact_bits=0,
        side_info_bits=0.0,
        total_bits=0.0,
        bpb=math.nan,
        bpp=math.nan,
        ratio=math.nan,
        reference_id=None,
        reference_source_index=None,
        reference_sha256="",
        candidate_rank=None,
        n_candidates=n_candidates,
        n_base_references=n_base_references,
        used_reference=0,
        retrieval_s=retrieval_s,
        selection_s=0.0,
        decode_s=0.0,
        roundtrip_ok=0,
        error=message,
    )


def _evaluate_codec(
    *,
    sample: Any,
    modality: str,
    codec_name: str,
    delta_codec: Any,
    delta_codec_version: str,
    zstd_codec: Optional[Any],
    zstd_codec_version: str,
    references: Sequence[Any],
    reference_ids: Sequence[int],
    index_coder: Any,
    retrieval_s: float,
    keep_going: bool,
) -> List[DeltaResult]:
    """Evaluate mandatory-reference and adaptive policies for one target."""

    target = bytes(sample.data)
    if not reference_ids:
        raise ValueError("delta evaluation requires at least one base reference")

    # Encode every retrieved full reference.  Candidate choice includes both
    # the artifact and the reference-id + stop-symbol cost.
    t0 = time.perf_counter()
    encoded_candidates = []
    for rank, reference_id in enumerate(reference_ids):
        reference = references[reference_id]
        encoded = _encode(delta_codec, target, bytes(reference.data))
        artifact = _artifact_bytes(encoded)
        side_bits = (
            index_coder.cost_bits(int(reference_id))
            + index_coder.cost_bits(None)
        )
        encoded_candidates.append(
            (len(artifact) * 8 + side_bits, rank, reference_id, artifact)
        )
    encoded_candidates.sort(key=lambda item: (item[0], item[1]))
    _patch_total, best_rank, best_reference_id, best_artifact = encoded_candidates[0]
    patch_selection_s = time.perf_counter() - t0

    best_reference = references[best_reference_id]
    t0 = time.perf_counter()
    reconstructed = _decoded_bytes(
        _decode(
            delta_codec,
            best_artifact,
            bytes(best_reference.data),
            target_size=len(target),
        )
    )
    patch_decode_s = time.perf_counter() - t0
    patch_ok = reconstructed == target
    if not patch_ok:
        raise RuntimeError(
            f"{codec_name} failed byte-exact round trip for {sample.sample_id}"
        )

    patch_side_bits = (
        index_coder.cost_bits(int(best_reference_id))
        + index_coder.cost_bits(None)
    )
    patch_row = _make_row(
        sample=sample,
        modality=modality,
        codec_name=codec_name,
        artifact_codec=codec_name,
        artifact_codec_version=delta_codec_version,
        policy="patch-only",
        artifact=best_artifact,
        side_info_bits=patch_side_bits,
        reference_id=int(best_reference_id),
        reference_source_index=int(best_reference.source_index),
        reference_sha256=hashlib.sha256(bytes(best_reference.data)).hexdigest(),
        candidate_rank=int(best_rank),
        n_candidates=len(reference_ids),
        n_base_references=len(references),
        retrieval_s=retrieval_s,
        selection_s=patch_selection_s,
        decode_s=patch_decode_s,
        roundtrip_ok=True,
    )

    try:
        # Encode a real standalone zstd-22 artifact. It competes with the
        # selected patch after both choices' side information is included.
        if zstd_codec is None:
            raise RuntimeError("standalone zstd fallback is unavailable")
        t0 = time.perf_counter()
        standalone_encoded = _encode(zstd_codec, target)
        standalone_artifact = _artifact_bytes(standalone_encoded)
        standalone_encode_s = time.perf_counter() - t0
        stop_bits = index_coder.cost_bits(None)
        standalone_total = len(standalone_artifact) * 8 + stop_bits

        if standalone_total <= patch_row.total_bits:
            t0 = time.perf_counter()
            adaptive_reconstructed = _decoded_bytes(
                _decode(zstd_codec, standalone_artifact, target_size=len(target))
            )
            adaptive_decode_s = time.perf_counter() - t0
            if adaptive_reconstructed != target:
                raise RuntimeError(
                    "zstd fallback failed byte-exact round trip for "
                    f"{sample.sample_id}"
                )
            adaptive_row = _make_row(
                sample=sample,
                modality=modality,
                codec_name=codec_name,
                artifact_codec="zstd",
                artifact_codec_version=zstd_codec_version,
                policy="adaptive",
                artifact=standalone_artifact,
                side_info_bits=stop_bits,
                reference_id=None,
                reference_source_index=None,
                reference_sha256="",
                candidate_rank=None,
                n_candidates=len(reference_ids),
                n_base_references=len(references),
                retrieval_s=retrieval_s,
                selection_s=patch_selection_s + standalone_encode_s,
                decode_s=adaptive_decode_s,
                roundtrip_ok=True,
            )
        else:
            # Decode again because this is a separate reported policy/operation.
            t0 = time.perf_counter()
            adaptive_reconstructed = _decoded_bytes(
                _decode(
                    delta_codec,
                    best_artifact,
                    bytes(best_reference.data),
                    target_size=len(target),
                )
            )
            adaptive_decode_s = time.perf_counter() - t0
            if adaptive_reconstructed != target:
                raise RuntimeError(
                    f"{codec_name} adaptive choice failed byte-exact round trip "
                    f"for {sample.sample_id}"
                )
            adaptive_row = _make_row(
                sample=sample,
                modality=modality,
                codec_name=codec_name,
                artifact_codec=codec_name,
                artifact_codec_version=delta_codec_version,
                policy="adaptive",
                artifact=best_artifact,
                side_info_bits=patch_side_bits,
                reference_id=int(best_reference_id),
                reference_source_index=int(best_reference.source_index),
                reference_sha256=hashlib.sha256(
                    bytes(best_reference.data)
                ).hexdigest(),
                candidate_rank=int(best_rank),
                n_candidates=len(reference_ids),
                n_base_references=len(references),
                retrieval_s=retrieval_s,
                selection_s=patch_selection_s + standalone_encode_s,
                decode_s=adaptive_decode_s,
                roundtrip_ok=True,
            )
    except Exception as exc:
        if not keep_going:
            raise
        print(f"  {codec_name} adaptive: ERROR: {type(exc).__name__}: {exc}")
        adaptive_row = _error_row(
            sample,
            modality,
            codec_name,
            "adaptive",
            len(reference_ids),
            len(references),
            retrieval_s,
            exc,
        )
    return [patch_row, adaptive_row]


def _save_csv(rows: Sequence[DeltaResult], path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fieldnames = list(DeltaResult.__dataclass_fields__)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _print_summary(rows: Sequence[DeltaResult]) -> None:
    groups: Dict[Tuple[str, str], List[DeltaResult]] = {}
    for row in rows:
        groups.setdefault((row.codec, row.policy), []).append(row)
    print("\nWhole-sample delta baseline summary (micro average)")
    for (codec, policy), all_rows in sorted(groups.items()):
        failed = [row for row in all_rows if row.error]
        group = [row for row in all_rows if not row.error]
        if failed:
            print(
                f"  {codec:13s} {policy:10s}  INVALID  "
                f"success={len(group):5d}/{len(all_rows):5d}  "
                f"failed={len(failed):5d}; no aggregate rate reported"
            )
            continue
        original = sum(row.original_bytes for row in group)
        total_bits = sum(row.total_bits for row in group)
        used = sum(row.used_reference for row in group)
        source_pixels = sum(row.source_pixels for row in group)
        bpb = total_bits / max(original, 1)
        ratio = original * 8 / total_bits if total_bits else math.inf
        image_rate = (
            f"  bpp={total_bits / source_pixels:.5f}"
            if source_pixels
            else ""
        )
        print(
            f"  {codec:13s} {policy:10s}  n={len(group):5d}  "
            f"bpb={bpb:.5f}  ratio={ratio:.5f}  "
            f"reference={100 * used / max(len(group), 1):.1f}%{image_rate}"
        )
    errors = sum(bool(row.error) for row in rows)
    if errors:
        print(f"  failed policy rows: {errors}")


def _construct_codec(factory: Any, name: str, timeout: float, tmp_dir: Optional[str]):
    """Call the adapter factory while tolerating its optional location keyword."""

    kwargs: Dict[str, Any] = {"timeout_s": timeout}
    if tmp_dir is not None:
        kwargs["temp_dir"] = tmp_dir
    try:
        return factory(name, **kwargs)
    except TypeError as exc:
        # Keep compatibility with a minimal factory accepting only the name;
        # do not mask arbitrary TypeErrors raised after construction begins.
        message = str(exc)
        if "unexpected keyword" not in message:
            raise
        return factory(name)


def _safe_codec_version(codec: Any) -> str:
    version = getattr(codec, "version", None)
    if not callable(version):
        return "unknown"
    try:
        value = version()
    except Exception as exc:
        return f"unknown ({type(exc).__name__})"
    return str(value).replace("\n", " ").strip() or "unknown"


def main(argv: Optional[Sequence[str]] = None) -> List[DeltaResult]:
    args = _build_parser().parse_args(argv)
    if args.m <= 0:
        raise ValueError(f"--m must be positive, got {args.m}")
    if args.n_samples is not None and args.n_samples <= 0:
        raise ValueError("--n-samples must be positive")
    if args.calib_samples < 0:
        raise ValueError("--calib-samples must be non-negative")
    if args.retrieval_pool is not None and args.retrieval_pool <= 0:
        raise ValueError("--retrieval-pool must be positive")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    codec_names = _parse_codec_names(args.codecs)

    inferred_modality, meta = _load_meta(args.database)
    modality = args.modality or inferred_modality
    if modality != inferred_modality:
        raise ValueError(
            f"--modality={modality!r} conflicts with prepared database modality "
            f"{inferred_modality!r}"
        )

    try:
        from utils.baseline_data import load_base_samples, load_eval_split
    except (ModuleNotFoundError, ImportError) as exc:
        raise RuntimeError(
            "utils.baseline_data is required to reconstruct canonical whole "
            "samples from the prepared split"
        ) from exc

    split = load_eval_split(
        args.database,
        modality=modality,
        n_samples=args.n_samples,
        calib_samples=args.calib_samples,
    )
    targets = list(split.test)
    references = list(load_base_samples(args.database, modality=modality))
    if not targets:
        raise ValueError("the persisted held-out eval split is empty")
    if not references:
        raise ValueError("the prepared base split is empty")

    source_to_reference: Dict[int, int] = {}
    for reference_id, reference in enumerate(references):
        source_index = int(reference.source_index)
        if source_index in source_to_reference:
            raise ValueError(f"duplicate base source index {source_index}")
        source_to_reference[source_index] = reference_id

    chunk_provenance = _load_chunk_provenance(args.database, modality)
    missing = sorted(set(chunk_provenance) - set(source_to_reference))
    if missing:
        raise ValueError(
            "base chunk provenance is inconsistent with reconstructed full base "
            f"samples; missing source indices {missing[:10]}"
        )

    retriever = None
    if args.candidate_policy == "retrieve":
        retriever = _load_retriever(
            args.database,
            modality,
            meta,
            args.embed_model,
            args.embed_device,
            args.embed_batch_size,
        )

    try:
        from compression.baselines import create_codec, create_delta_codec
        from compression.rac_index import FixedIndexCoder
    except (ModuleNotFoundError, ImportError) as exc:
        raise RuntimeError(
            "baseline codec adapters are unavailable; expected "
            "compression.baselines.create_codec/create_delta_codec"
        ) from exc

    zstd_codec = None
    zstd_codec_version = ""
    try:
        zstd_codec = _construct_codec(
            create_codec, "zstd", timeout=args.timeout, tmp_dir=args.tmp_dir
        )
        zstd_codec_version = _safe_codec_version(zstd_codec)
    except Exception as exc:
        if not args.keep_going:
            raise
        print(
            "adaptive zstd fallback unavailable; patch-only rows will still "
            f"run: {type(exc).__name__}: {exc}"
        )
    delta_codecs: Dict[str, Any] = {}
    delta_codec_versions: Dict[str, str] = {}
    codec_init_errors: Dict[str, BaseException] = {}
    for name in codec_names:
        try:
            delta_codecs[name] = _construct_codec(
                create_delta_codec,
                name,
                timeout=args.timeout,
                tmp_dir=args.tmp_dir,
            )
            delta_codec_versions[name] = _safe_codec_version(delta_codecs[name])
        except Exception as exc:
            if not args.keep_going:
                raise
            codec_init_errors[name] = exc
            print(f"{name}: unavailable: {type(exc).__name__}: {exc}")
    # Reference ids address *whole base samples*, not retrieval chunks.
    index_coder = FixedIndexCoder(len(references))

    print(
        f"Loaded {len(targets)} persisted held-out {modality} targets and "
        f"{len(references)} full base references | chunks={len(chunk_provenance)} "
        f"| calibration excluded={len(split.calibration)} "
        f"| candidate-policy={args.candidate_policy} | m={args.m}"
    )

    rows: List[DeltaResult] = []
    all_reference_ids = list(range(len(references)))
    for target_number, sample in enumerate(targets, start=1):
        retrieval_t0 = time.perf_counter()
        if retriever is None:
            reference_ids = all_reference_ids
        else:
            reference_ids = _retrieve_reference_ids(
                retriever,
                _query_for_retriever(sample, modality, meta),
                chunk_provenance,
                source_to_reference,
                args.m,
                args.retrieval_pool,
            )
        retrieval_s = time.perf_counter() - retrieval_t0

        print(
            f"[{target_number}/{len(targets)}] {sample.sample_id}: "
            f"{len(sample.data)} B, {len(reference_ids)} references"
        )
        for codec_name in codec_names:
            if codec_name in codec_init_errors:
                rows.extend(
                    _error_rows(
                        sample,
                        modality,
                        codec_name,
                        len(reference_ids),
                        len(references),
                        retrieval_s,
                        codec_init_errors[codec_name],
                    )
                )
                continue
            delta_codec = delta_codecs[codec_name]
            try:
                rows.extend(
                    _evaluate_codec(
                        sample=sample,
                        modality=modality,
                        codec_name=codec_name,
                        delta_codec=delta_codec,
                        delta_codec_version=delta_codec_versions[codec_name],
                        zstd_codec=zstd_codec,
                        zstd_codec_version=zstd_codec_version,
                        references=references,
                        reference_ids=reference_ids,
                        index_coder=index_coder,
                        retrieval_s=retrieval_s,
                        keep_going=args.keep_going,
                    )
                )
            except Exception as exc:
                if not args.keep_going:
                    raise
                print(f"  {codec_name}: ERROR: {type(exc).__name__}: {exc}")
                rows.extend(
                    _error_rows(
                        sample,
                        modality,
                        codec_name,
                        len(reference_ids),
                        len(references),
                        retrieval_s,
                        exc,
                    )
                )

    _print_summary(rows)
    if args.output:
        _save_csv(rows, args.output)
        print(f"CSV -> {args.output}")
    return rows


if __name__ == "__main__":
    main()
