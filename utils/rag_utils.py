"""
Modality-agnostic retrieval for Retrieval-Augmented Compression (RAC).

This module knows nothing about text, audio, or images.  It deals in opaque
*items* (one per base chunk) and *scorers* that each turn a query item into a
ranking over the base.  Multiple scorer rankings are fused with Reciprocal Rank
Fusion (RRF).

A concrete modality wires this up by supplying scorers built from its own
featurisers (see ``utils/text_utils.py`` for the text instantiation: a BM25
lexical scorer + a Qwen3-Embedding dense scorer).  Audio/image will register
their own scorers from ``audio_utils`` / ``img_utils`` without touching this file.

Typical use
-----------
    retriever = Retriever([bm25_scorer, dense_scorer])
    retriever.build(base_items)              # base_items[i] describes base chunk i
    retriever.save("retriever_cache/enwiki")
    hits = retriever.retrieve(query_item, top_k=16, exclude_ids={query_id})
    # hits -> [(chunk_id, fused_score), ...] best first
"""
from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Callable

import numpy as np
import faiss


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[int, float]]:
    """Fuse several ranked id-lists into one.

    Each ranking is a sequence of chunk ids, best first.  An id at 0-based rank
    ``r`` in a ranking contributes ``weight / (k + r + 1)`` to its fused score.
    Returns ``[(chunk_id, score), ...]`` sorted by score descending.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: Dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for r, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + w / (k + r + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


# ---------------------------------------------------------------------------
# Scorer interface
# ---------------------------------------------------------------------------

class Scorer:
    """A single retrieval signal over the base corpus.

    Subclasses build an index from ``corpus`` (a list of per-chunk items) and
    answer ``rank(query, pool)`` with the ``pool`` best chunk ids, best first.
    ``save``/``load`` persist the index under ``dirpath`` (one subdir per scorer,
    keyed by ``name``).
    """

    name: str = "scorer"

    def build(self, corpus: List[Any]) -> None:
        raise NotImplementedError

    def rank(self, query: Any, pool: int) -> List[int]:
        raise NotImplementedError

    def rank_many(self, queries: Sequence[Any], pool: int) -> List[List[int]]:
        """Rank a batch of queries; returns one id-list per query, best first.

        Default falls back to per-query :meth:`rank`. Subclasses with batchable
        backends (e.g. dense embedding + faiss) should override for efficiency.
        """
        return [self.rank(q, pool) for q in queries]

    def save(self, dirpath: str) -> None:
        raise NotImplementedError

    def load(self, dirpath: str) -> None:
        raise NotImplementedError


class BM25Scorer(Scorer):
    """Lexical scorer over tokenised items, backed by ``bm25s``.

    ``bm25s`` stores BM25 weights as a sparse term-document matrix and scores a
    whole batch of queries via sparse mat-mul, so ``rank_many`` is natively
    batched (far faster than per-query ``rank_bm25`` for long, full-document
    queries). ``tokenize`` maps an item to a token list; it is *not* persisted,
    so supply the same callable when reloading.
    """

    name = "bm25"

    def __init__(self, tokenize: Callable[[Any], List[str]]):
        self.tokenize = tokenize
        self._bm25 = None

    def build(self, corpus: List[Any]) -> None:
        import bm25s
        self._bm25 = bm25s.BM25()
        self._bm25.index([self.tokenize(x) for x in corpus], show_progress=False)

    def rank(self, query: Any, pool: int) -> List[int]:
        return self.rank_many([query], pool)[0]

    def rank_many(self, queries: Sequence[Any], pool: int) -> List[List[int]]:
        # Sparse mat-mul over the whole batch; bm25s shows its own progress bar.
        query_tokens = [self.tokenize(q) for q in queries]
        pool = min(pool, self._bm25.scores["num_docs"])
        results, _ = self._bm25.retrieve(
            query_tokens, k=pool, n_threads=0,
            show_progress=len(query_tokens) > 1, leave_progress=len(query_tokens) > 1)
        return [[int(i) for i in row] for row in results]

    def save(self, dirpath: str) -> None:
        self._bm25.save(os.path.join(dirpath, self.name), show_progress=False)

    def load(self, dirpath: str) -> None:
        import bm25s
        self._bm25 = bm25s.BM25.load(os.path.join(dirpath, self.name), load_vocab=True)


class DenseScorer(Scorer):
    """Semantic scorer: inner-product search via faiss (IndexFlatIP).

    ``embed`` must return L2-normalised float32 arrays so that inner product
    equals cosine similarity.  The same embedder is used at build time and
    query time.  ``embed`` is not persisted — supply it again on reload.
    ``query_batch_size`` (optional) overrides the embed batch size at query time
    only — useful when queries are much longer than base chunks.
    """

    name = "dense"

    def __init__(self, embed: Callable[..., "Any"], query_batch_size: Optional[int] = None):
        self.embed = embed
        self.query_batch_size = query_batch_size
        self._index = None
        self._dim: Optional[int] = None

    def _prep(self, vecs):
        vecs = np.asarray(vecs, dtype="float32")
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        return np.ascontiguousarray(vecs)

    def build(self, corpus: List[Any]) -> None:
        vecs = self._prep(self.embed(corpus))
        self._dim = vecs.shape[1]
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(vecs)

    def rank(self, query: Any, pool: int) -> List[int]:
        return self.rank_many([query], pool)[0]

    def rank_many(self, queries: Sequence[Any], pool: int) -> List[List[int]]:
        # One batched embed (batches internally) + one batched faiss search.
        vecs = self._prep(self.embed(list(queries), self.query_batch_size))  # [Q, dim]
        pool = min(pool, self._index.ntotal)
        _, idx = self._index.search(vecs, pool)           # [Q, pool]
        return [[int(i) for i in row if i >= 0] for row in idx]

    def save(self, dirpath: str) -> None:
        sub = os.path.join(dirpath, self.name)
        os.makedirs(sub, exist_ok=True)
        faiss.write_index(self._index, os.path.join(sub, "index.faiss"))

    def load(self, dirpath: str) -> None:
        sub = os.path.join(dirpath, self.name)
        self._index = faiss.read_index(os.path.join(sub, "index.faiss"))
        self._dim = self._index.d


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """Fuses one or more :class:`Scorer` rankings with RRF.

    The base corpus (``items``) is stored alongside the scorer indices so that
    callers can map returned chunk ids back to their payloads.  ``items`` is a
    list of arbitrary picklable objects — typically the chunk's retrieval text,
    or a ``(chunk_id, text)`` record.
    """

    def __init__(self, scorers: List[Scorer], rrf_k: int = 60,
                 weights: Optional[Sequence[float]] = None):
        self.scorers = scorers
        self.rrf_k = rrf_k
        self.weights = weights
        self.items: List[Any] = []

    # -- build / persist -------------------------------------------------

    def build(self, items: List[Any]) -> None:
        self.items = list(items)
        for s in self.scorers:
            s.build(self.items)

    def save(self, dirpath: str) -> None:
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, "items.pkl"), "wb") as f:
            pickle.dump(self.items, f)
        meta = {"scorers": [s.name for s in self.scorers],
                "rrf_k": self.rrf_k, "weights": self.weights,
                "n_items": len(self.items)}
        with open(os.path.join(dirpath, "retriever.json"), "w") as f:
            json.dump(meta, f, indent=2)
        for s in self.scorers:
            s.save(dirpath)

    def load(self, dirpath: str) -> None:
        with open(os.path.join(dirpath, "items.pkl"), "rb") as f:
            self.items = pickle.load(f)
        for s in self.scorers:
            s.load(dirpath)

    # -- query -----------------------------------------------------------

    def retrieve(
        self,
        query: Any,
        top_k: int,
        pool: Optional[int] = None,
        exclude_ids: Iterable[int] = (),
    ) -> List[Tuple[int, float]]:
        """Return the ``top_k`` fused ``(chunk_id, score)`` hits, best first.

        ``pool`` is how many candidates each scorer contributes before fusion
        (default ``max(5*top_k, 50)``).  ``exclude_ids`` drops chunks (e.g. the
        query's own id when the query is itself a base chunk).
        """
        if pool is None:
            pool = max(5 * top_k, 50)
        exclude = set(exclude_ids)
        # pull a few extra so exclusions don't starve the result
        eff_pool = pool + len(exclude)
        rankings = [s.rank(query, eff_pool) for s in self.scorers]
        return self._fuse(rankings, top_k, exclude)

    def retrieve_many(
        self,
        queries: Sequence[Any],
        top_k: int,
        pool: Optional[int] = None,
        exclude_ids: Optional[Sequence[Iterable[int]]] = None,
    ) -> List[List[Tuple[int, float]]]:
        """Batched :meth:`retrieve` over many queries — one result list per query.

        Equivalent to calling ``retrieve`` per query, but each scorer ranks the
        whole batch via :meth:`Scorer.rank_many` (dense embedding + faiss search
        run in bulk). ``exclude_ids``, if given, is a per-query iterable of ids to
        drop (same length as ``queries``); omit it when nothing is excluded.
        """
        if pool is None:
            pool = max(5 * top_k, 50)
        excludes = [set(e) for e in exclude_ids] if exclude_ids is not None \
            else [set()] * len(queries)
        eff_pool = pool + max((len(e) for e in excludes), default=0)
        # per_scorer[s][q] = ranking of query q by scorer s
        per_scorer = [s.rank_many(list(queries), eff_pool) for s in self.scorers]
        results: List[List[Tuple[int, float]]] = []
        for q in range(len(queries)):
            rankings = [per_scorer[s][q] for s in range(len(self.scorers))]
            results.append(self._fuse(rankings, top_k, excludes[q]))
        return results

    def _fuse(self, rankings, top_k: int, exclude: set) -> List[Tuple[int, float]]:
        fused = reciprocal_rank_fusion(rankings, self.rrf_k, self.weights)
        out: List[Tuple[int, float]] = []
        for cid, score in fused:
            if cid in exclude:
                continue
            out.append((cid, score))
            if len(out) >= top_k:
                break
        return out
