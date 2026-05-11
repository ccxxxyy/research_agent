"""RAG building blocks — cross-encoder reranker for knowledge search.

This package's sole production export is ``CrossEncoderReranker``,
used by ``mcp_servers.knowledge_server._search()`` to rerank hybrid
retrieval (FAISS + BM25 + RRF) candidates.

All other RAG primitives (PDF loading, chunking, embedding) live in
``mcp_servers.knowledge_server`` as private helpers — they are tightly
coupled to the FAISS index format and not shared.
"""

from research_agent.rag.reranker import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
)

__all__ = [
    "CrossEncoderReranker",
    "DEFAULT_RERANKER_MODEL",
]
