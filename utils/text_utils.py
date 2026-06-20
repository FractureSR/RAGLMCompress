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
import tqdm

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
    key = os.path.normpath(path).lower().replace("-", "_").replace(os.sep, "/")
    for name, loader in _TEXT_LOADERS.items():
        if name.replace("-", "_").lower() in key:
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


@register_text_loader("codeparrot_github_code/C.jsonl")
def _load_codeparrot_github_code_C(path: str, n: Optional[int] = None, skip: int = 0) -> List[str]:
    return _load_jsonl(path, ("code",), n, skip)


@register_text_loader("codeparrot_github_code/Java.jsonl")
def _load_codeparrot_github_code_Java(path: str, n: Optional[int] = None, skip: int = 0) -> List[str]:
    return _load_jsonl(path, ("code",), n, skip)


@register_text_loader("codeparrot_github_code/Python.jsonl")
def _load_codeparrot_github_code_Python(path: str, n: Optional[int] = None, skip: int = 0) -> List[str]:
    return _load_jsonl(path, ("code",), n, skip)

# ---------------------------------------------------------------------------
# Compression preprocessing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TextChunk:
    """A pre-tokenized document fragment produced by the chunking preprocessor.

    token_ids is the canonical payload — passed directly to the compressor
    without re-tokenization to avoid BPE boundary artefacts.
    doc_idx / chunk_idx allow aggregating chunk-level metrics back to per-doc.
    text is the decoded chunk (populated only when ``decode=True``), used when the
    chunk is consumed as text — e.g. embedded by a *different* tokenizer.
    """
    token_ids:    List[int]
    doc_idx:      int   # index into the original texts list
    chunk_idx:    int   # 0-based position within this document's chunks
    total_chunks: int   # total chunks for this document
    text:         Optional[str] = None


def chunk_documents_for_compression(
    texts: List[str],
    tokenizer,
    max_tokens: int,
    chunk_overlap: int = 0,
    decode: bool = False,
) -> List[TextChunk]:
    """Split documents into ≤ max_tokens token windows (length set by *tokenizer*).

    ``chunk_overlap`` > 0 gives sliding windows (use for retrieval units); keep it
    0 for compression data so the chunks partition the document exactly. Set
    ``decode=True`` to also fill each chunk's ``text`` field — needed when the
    chunk is consumed as text (e.g. a retrieval unit embedded by a *different*
    tokenizer). Pure preprocessing step — no compression logic involved.
    """
    if chunk_overlap < 0 or chunk_overlap >= max_tokens:
        raise ValueError("require 0 <= chunk_overlap < max_tokens")
    step = max_tokens - chunk_overlap
    chunks: List[TextChunk] = []
    iterator = enumerate(texts)
    iterator = tqdm.tqdm(iterator, total=len(texts), desc="chunking", unit="doc")
    for doc_idx, text in iterator:
        # verbose=False: long docs exceed model_max_length, but we only chunk them
        # (never feed the whole thing to the model), so the length warning is noise.
        ids: List[int] = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
        windows = [ids[s: s + max_tokens] for s in range(0, len(ids), step)]
        if not windows:
            windows = [[]]
        n = len(windows)
        for chunk_idx, window in enumerate(windows):
            chunks.append(TextChunk(
                token_ids=window,
                doc_idx=doc_idx,
                chunk_idx=chunk_idx,
                total_chunks=n,
                text=tokenizer.decode(window, skip_special_tokens=True) if decode else None,
            ))
    return chunks


# ---------------------------------------------------------------------------
# Text retrieval factories (text instantiation of utils/rag_utils.py)
# ---------------------------------------------------------------------------

import re

_TOKEN_RE = re.compile(
    r"(?:c\+\+|c#|\.net|[a-z0-9]+(?:[._+#-][a-z0-9]+)*)"
)

def bm25_tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


def _last_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the last *real* token's hidden state (Qwen3-Embedding's recipe).

    Handles both left- and right-padded batches: with left padding the last
    column is always a real token; otherwise index by each row's true length.
    """
    left_padded = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padded:
        return last_hidden[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), lengths]


def make_text_dense_embedder(
    model_path: str = "Qwen/Qwen3-Embedding-0.6B",
    device: Optional[str] = None,
    batch_size: int = 32,
    query_prompt: Optional[str] = None,
    max_length: int = 8192,
):
    """Build an ``embed(items) -> np.ndarray`` closure backed by raw 🤗 transformers.

    Used as the ``embed`` callable for :class:`utils.rag_utils.DenseScorer`.
    Replicates the official Qwen3-Embedding recipe — last-token pooling + L2
    normalisation over ``AutoModel`` hidden states — so no sentence-transformers
    dependency (and no Hub probe for its optional config files) is needed.
    ``query_prompt`` (if given) is prepended to each item — Qwen3-Embedding uses
    an "Instruct: …\\nQuery: …" template for asymmetric retrieval; leave None for
    symmetric pattern-similarity retrieval.
    """
    import numpy as np
    from transformers import AutoModel, AutoTokenizer

    on_cuda = device is not None and str(device).startswith("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    # bf16 on GPU halves activation memory (attention dominates for long,
    # token-dense inputs like code) with negligible retrieval-quality impact.
    model = AutoModel.from_pretrained(
        model_path, dtype=torch.bfloat16 if on_cuda else torch.float32)
    if device is not None:
        model = model.to(device)
    model.eval()

    @torch.no_grad()
    def embed(items: List[str], bs: Optional[int] = None):
        bs = bs or batch_size  # per-call override (e.g. smaller for long queries)
        texts = [str(x) for x in items]
        if query_prompt:
            texts = [query_prompt + t for t in texts]
        batches = range(0, len(texts), bs)
        batches = tqdm.tqdm(batches, desc="embed", unit="batch")
        out = []
        for i in batches:
            enc = tokenizer(
                texts[i:i + bs], padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(model.device)
            hidden = model(**enc).last_hidden_state
            emb = _last_token_pool(hidden, enc["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out.append(emb.float().cpu().numpy())
        if not out:
            return np.empty((0, model.config.hidden_size), dtype=np.float32)
        return np.concatenate(out, axis=0)

    return embed


def make_text_retriever(
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B",
    device: Optional[str] = None,
    rrf_k: int = 60,
    batch_size: int = 64,
    query_batch_size: Optional[int] = None,
):
    """Construct an *unbuilt* :class:`Retriever` with BM25 + Qwen3 dense scorers.

    ``batch_size`` embeds the base corpus at build time; ``query_batch_size``
    (default = ``batch_size``) embeds queries at retrieval time — set it smaller
    when queries are much longer than base chunks (avoids GPU OOM). The same
    factory must be used for build and load (callables aren't persisted).
    """
    from utils.rag_utils import Retriever, BM25Scorer, DenseScorer
    bm25 = BM25Scorer(tokenize=bm25_tokenize)
    dense = DenseScorer(embed=make_text_dense_embedder(embed_model, device, batch_size),
                        query_batch_size=query_batch_size)
    return Retriever([bm25, dense], rrf_k=rrf_k)


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
