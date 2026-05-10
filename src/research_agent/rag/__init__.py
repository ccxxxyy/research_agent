"""RAG building blocks for PDF ingest and the knowledge-base search pipeline.

Production hybrid retrieval (FAISS + BM25 + rerank) lives in
``mcp_servers.knowledge_server`` and the in-process ``knowledge_*``
tools — not in this package's public exports.

**Removed (legacy Phase-3):** ``HybridRetriever``, ``RetrievalGrader``,
``QueryRewriter`` and the Chroma-backed graph that used them. Corrective
behaviour is handled by the ``knowledge_expert`` over ``knowledge_search``
signals.

Embeddings for the knowledge pipeline live only in
``mcp_servers.knowledge_server._get_embedder()`` (bge-small-zh). The
``chunker`` / ``loader`` modules remain as standalone helpers; they are
not re-exported here. This package publicly exports
``CrossEncoderReranker`` for ``knowledge_server.search``.
"""

from research_agent.rag.reranker import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
)

__all__ = [
    "CrossEncoderReranker",
    "DEFAULT_RERANKER_MODEL",
]
