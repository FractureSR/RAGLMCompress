"""Text loading and preprocessing utilities for compression evaluation.

Dataset loaders are registered by name; ``load_text_documents`` auto-detects
the dataset from the path and dispatches to the matching loader.

Adding a new dataset
--------------------
    from utils.text_utils import register_text_loader

    @register_text_loader("my_dataset")
    def _load_my_dataset(path: str, n: Optional[int] = None, skip: int = 0) -> List[str]:
        ...
"""

from __future__ import annotations

import json
import os
import torch
from dataclasses import dataclass
from itertools import islice
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from datasets import DatasetDict, load_dataset, load_from_disk


# ---------------------------------------------------------------------------
# Low-level record helpers (used by rag_utils)
# ---------------------------------------------------------------------------

def extract_document_text(record: Any, text_keys: Sequence[str] = ("text", "content", "body", "document")) -> str:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        raise TypeError(f"Unsupported record type: {type(record)!r}")

    concated_value: str = ""
    for key in text_keys:
        value = record.get(key)
        if value is not None:
            concated_value += str(value)
        else:
            raise KeyError(
                f"No text field {key!r} found, "
                f"available: {list(record.keys())}"
            )
    return concated_value


def chunk_words(text: str, chunk_size: int, chunk_overlap: int = 0) -> List[str]:
    """Split text into fixed-size overlapping word chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    words = text.split()
    if not words:
        return []
    step = chunk_size - chunk_overlap
    return [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), step)]


# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------

TextLoader = Callable[[str, Optional[int], int], List[str]]

_TEXT_LOADERS: Dict[str, TextLoader] = {}


def register_text_loader(name: str):
    """Decorator to register a text dataset loader by name.

    Detection: *name* (hyphens normalised to underscores) must appear as a
    substring of the normalised dataset path.

    Example::

        @register_text_loader("eurlex")
        def _load_eurlex(path, n=None, skip=0):
            return _load_jsonl(path, ("text", "celex_id"), n, skip)
    """
    def decorator(fn: TextLoader) -> TextLoader:
        _TEXT_LOADERS[name] = fn
        return fn
    return decorator


def _find_text_loader(path: str) -> TextLoader:
    key = os.path.basename(os.path.normpath(path)).lower().replace("-", "_")
    for name, loader in _TEXT_LOADERS.items():
        if name.replace("-", "_") in key:
            return loader
    raise ValueError(
        f"No text loader registered for {path!r}.\n"
        f"Known datasets: {sorted(_TEXT_LOADERS)}.\n"
        f"Register a new one with @register_text_loader('name')."
    )


def load_text_documents(
    path: str,
    num_documents: Optional[int] = None,
    skip_documents: int = 0,
) -> List[str]:
    """Dispatch to the registered loader for the dataset at *path*.

    The dataset is identified by matching registered names against the path.
    """
    return _find_text_loader(path)(path, num_documents, skip_documents)


# Backward-compat alias
load_text_documents_from_hf = load_text_documents


# ---------------------------------------------------------------------------
# Shared low-level helpers used by built-in loaders
# ---------------------------------------------------------------------------

def _load_hf(
    path: str,
    text_keys: Sequence[str],
    n: Optional[int],
    skip: int,
    split: str = "train",
    streaming: bool = True,
) -> List[str]:
    hf_markers = ("dataset_info.json", "dataset_dict.json")
    if os.path.isdir(path) and any(os.path.exists(os.path.join(path, m)) for m in hf_markers):
        loaded = load_from_disk(path)
        dataset = loaded[split] if isinstance(loaded, DatasetDict) else loaded
    else:
        dataset = load_dataset(path, split=split, streaming=streaming)

    iterator = iter(dataset)
    if skip:
        iterator = islice(iterator, skip, None)
    if n is not None:
        iterator = islice(iterator, n)

    texts: List[str] = []
    for record in iterator:
        text = extract_document_text(record, text_keys).strip()
        if text:
            texts.append(text)
    return texts


def _load_jsonl(
    path: str,
    text_keys: Sequence[str],
    n: Optional[int],
    skip: int,
) -> List[str]:
    texts: List[str] = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if skipped < skip:
                skipped += 1
                continue
            if n is not None and len(texts) >= n:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = extract_document_text(record, text_keys).strip()
            if text:
                texts.append(text)
    return texts


# ---------------------------------------------------------------------------
# Built-in dataset loaders
# ---------------------------------------------------------------------------

@register_text_loader("cosmopedia")
def _load_cosmopedia(path: str, n: Optional[int] = None, skip: int = 0) -> List[str]:
    return _load_hf(path, ("text",), n, skip)


@register_text_loader("enwiki")
def _load_enwiki(path: str, n: Optional[int] = None, skip: int = 0) -> List[str]:
    return _load_hf(path, ("text",), n, skip)


# ---------------------------------------------------------------------------
# Compression preprocessing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TextChunk:
    """A pre-tokenized document fragment produced by the chunking preprocessor.

    token_ids is the canonical payload — passed directly to the compressor
    without re-tokenization to avoid BPE boundary artefacts.
    doc_idx / chunk_idx allow aggregating chunk-level metrics back to per-doc.
    """
    token_ids:    List[int]
    doc_idx:      int   # index into the original texts list
    chunk_idx:    int   # 0-based position within this document's chunks
    total_chunks: int   # total chunks for this document


def chunk_documents_for_compression(
    texts: List[str],
    tokenizer,
    max_tokens: int,
) -> List[TextChunk]:
    """Split documents into non-overlapping ≤ max_tokens token chunks.

    Pure preprocessing step — no compression logic involved.
    """
    chunks: List[TextChunk] = []
    for doc_idx, text in enumerate(texts):
        ids: List[int] = tokenizer(text, add_special_tokens=False)["input_ids"]
        windows = [ids[s: s + max_tokens] for s in range(0, len(ids), max_tokens)]
        if not windows:
            windows = [[]]
        n = len(windows)
        for chunk_idx, window in enumerate(windows):
            chunks.append(TextChunk(
                token_ids=window,
                doc_idx=doc_idx,
                chunk_idx=chunk_idx,
                total_chunks=n,
            ))
    return chunks


# Note: Possibly padding for bs > 1 could introduce new precision problems
def pad_token_ids(
    token_id_lists: List[List[int]],
    pad_id: int,
    device=None,
) -> Tuple["torch.Tensor", "torch.Tensor"]:  # type: ignore[name-defined]
    """Pad pre-tokenized token ID lists into (input_ids, attention_mask) tensors."""
    max_len = max((len(ids) for ids in token_id_lists), default=1)
    B = len(token_id_lists)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros(B, max_len, dtype=torch.long)
    for i, ids in enumerate(token_id_lists):
        n = len(ids)
        if n:
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            attn_mask[i, :n] = 1
    if device is not None:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
    return input_ids, attn_mask
