"""Hybrid retriever combining vector search (semantic) and BM25 (keyword).

Phase-3 component, currently dormant
------------------------------------
This module dates from Phase-3 when the project ran on Chroma.
``langchain_chroma`` is no longer a project dependency (we migrated
the production knowledge plane to FAISS — see
``mcp_servers/knowledge_server.py``), so the type hint on
``__init__`` is intentionally a forward reference (``"Any"`` for
practical purposes); the class still works against any vector store
that exposes ``asimilarity_search`` / ``aadd_documents``.

The class is kept importable so the (now optional) Phase-3
``/research`` route and its test fixtures keep type-checking, but at
runtime it is only instantiated when ``langchain_chroma`` happens to
be installed (see ``main.py`` lifespan).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from loguru import logger
from rank_bm25 import BM25Okapi

if TYPE_CHECKING:  # pragma: no cover
    # Imported only for type checkers; ``langchain_chroma`` is not a
    # runtime dependency anymore.
    from langchain_chroma import Chroma


class HybridRetriever:
    """Merges vector similarity search with BM25 keyword search.

    Reciprocal Rank Fusion (RRF) combines the two result lists into
    a single ranked list, balancing semantic understanding with
    exact keyword matching.
    """

    def __init__(
        self,
        vectorstore: "Chroma | Any",
        documents: list[Document] | None = None,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ) -> None:
        self._vectorstore = vectorstore
        self._bm25_weight = bm25_weight
        self._vector_weight = vector_weight
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[Document] = []

        if documents:
            self._build_bm25_index(documents)

    def _build_bm25_index(self, documents: list[Document]) -> None:
        """Build BM25 index from document corpus."""
        self._bm25_docs = documents
        tokenized = [doc.page_content.split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built with {} documents", len(documents))

    async def search(self, query: str, top_k: int = 5) -> list[Document]:
        """Execute hybrid search and return fused results."""
        vector_results = await self._vector_search(query, top_k=top_k * 2)
        bm25_results = self._bm25_search(query, top_k=top_k * 2)

        fused = self._reciprocal_rank_fusion(
            [vector_results, bm25_results],
            weights=[self._vector_weight, self._bm25_weight],
        )

        results = fused[:top_k]
        logger.info(
            "Hybrid search: query='{}' → {} vector + {} bm25 → {} fused",
            query[:50], len(vector_results), len(bm25_results), len(results),
        )
        return results

    async def _vector_search(self, query: str, top_k: int) -> list[Document]:
        return await self._vectorstore.asimilarity_search(query, k=top_k)

    def _bm25_search(self, query: str, top_k: int) -> list[Document]:
        if self._bm25 is None:
            return []
        tokenized_query = query.split()
        scores = self._bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._bm25_docs[i] for i in ranked_indices[:top_k]]

    @staticmethod
    def _reciprocal_rank_fusion(
        result_lists: list[list[Document]],
        weights: list[float],
        k: int = 60,
    ) -> list[Document]:
        """Combine multiple ranked lists using weighted RRF.

        RRF score = sum(weight / (k + rank)) for each list.
        """
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for results, weight in zip(result_lists, weights):
            for rank, doc in enumerate(results):
                doc_id = doc.metadata.get("source", "") + doc.page_content[:100]
                doc_map[doc_id] = doc
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + weight / (k + rank + 1)

        sorted_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids]

    async def add_documents(self, documents: list[Document]) -> None:
        """Add new documents to both vector store and BM25 index."""
        await self._vectorstore.aadd_documents(documents)
        self._bm25_docs.extend(documents)
        self._build_bm25_index(self._bm25_docs)
        logger.info("Added {} documents to hybrid index", len(documents))
