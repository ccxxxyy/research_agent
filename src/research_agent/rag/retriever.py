"""Hybrid retrieval primitives — BM25 sparse index + RRF fusion.

Extracted from ``knowledge_server.py`` so the retrieval pipeline is
independently importable, testable, and demonstrable.  The
``knowledge_server`` still uses these classes but the caller does not
need to stand up an MCP server to exercise retrieval logic.

Components
----------
``BM25Index``
    Thin wrapper around ``rank_bm25.BM25Okapi`` that tokenizes
    documents and maps BM25 scores back to their originating
    ``(content, metadata)`` dicts.

``hybrid_rrf_fuse``
    Weighted Reciprocal-Rank Fusion of dense (vector) and sparse
    (BM25) result lists.  Returns a unified record list sorted by
    fused rank score, deduplicated by ``(source, page, content[:80])``.
"""

from __future__ import annotations

import re
from typing import Any


class BM25Index:
    """BM25Okapi index over a list of ``{content, metadata}`` dicts.

    Tokenization is intentionally simple: lower-case + split on
    non-word characters.  CJK characters survive as single-char
    tokens, which BM25 handles fine for queries that share noun
    phrases with the documents.
    """

    _SPLIT_RE = re.compile(r"\W+", flags=re.UNICODE)

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        from rank_bm25 import BM25Okapi

        self.docs = docs
        self._is_empty: bool = not docs
        tokenized = [self._tokenize(d["content"]) for d in docs]
        if not tokenized:
            tokenized = [[""]]
            self.docs = [{"content": "", "metadata": {}}]
        self._bm25 = BM25Okapi(tokenized)

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        return [t for t in cls._SPLIT_RE.split(text.lower()) if t]

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return ``[(corpus_index, bm25_score)]`` sorted descending."""
        if self._is_empty:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


def hybrid_rrf_fuse(
    vector_hits: list[tuple[dict[str, Any], float]],
    bm25_hits: list[tuple[int, float, dict[str, Any]]],
    *,
    k_rrf: int = 60,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list[dict[str, Any]]:
    """Weighted Reciprocal Rank Fusion of vector + BM25 results.

    Returns one record per UNIQUE document (deduplicated by
    ``(source, page, content[:80])``) carrying:

    * ``content``, ``metadata``
    * ``vector_score`` — raw cosine similarity ([0, 1] post-norm)
    * ``bm25_score`` — raw BM25 (unbounded, model-dependent)
    * ``rrf_score`` — fused rank score (the sort key)
    * ``vector_rank``, ``bm25_rank`` — 1-indexed rank in each list
    """
    fused: dict[str, dict[str, Any]] = {}

    def _key(meta: dict[str, Any], content: str) -> str:
        return f"{meta.get('source', '')}|p={meta.get('page', '?')}|{content[:80]}"

    for rank, (doc, score) in enumerate(vector_hits, start=1):
        k = _key(doc["metadata"], doc["content"])
        rec = fused.setdefault(
            k,
            {
                "content": doc["content"],
                "metadata": doc["metadata"],
                "vector_score": score,
                "bm25_score": 0.0,
                "rrf_score": 0.0,
                "vector_rank": rank,
                "bm25_rank": None,
            },
        )
        rec["vector_score"] = max(rec["vector_score"], score)
        rec["rrf_score"] += vector_weight / (k_rrf + rank)

    for rank, (_, score, doc) in enumerate(bm25_hits, start=1):
        k = _key(doc["metadata"], doc["content"])
        rec = fused.setdefault(
            k,
            {
                "content": doc["content"],
                "metadata": doc["metadata"],
                "vector_score": 0.0,
                "bm25_score": score,
                "rrf_score": 0.0,
                "vector_rank": None,
                "bm25_rank": rank,
            },
        )
        rec["bm25_score"] = max(rec["bm25_score"], score)
        rec["bm25_rank"] = (
            rank if rec["bm25_rank"] is None else min(rec["bm25_rank"], rank)
        )
        rec["rrf_score"] += bm25_weight / (k_rrf + rank)

    return sorted(fused.values(), key=lambda r: r["rrf_score"], reverse=True)


__all__ = ["BM25Index", "hybrid_rrf_fuse"]
