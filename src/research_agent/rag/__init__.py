"""RAG building blocks — retrieval, grading, rewriting, and reranking.

This package exports the core Corrective-RAG pipeline components as
independently testable classes:

* ``BM25Index`` — sparse BM25 index over tokenized documents.
* ``hybrid_rrf_fuse`` — weighted Reciprocal-Rank Fusion of dense +
  sparse retrieval lists.
* ``RetrievalGrader`` — three-bucket quality classifier
  (high / medium / low) driving the corrective loop.
* ``QueryRewriter`` — LLM-based query rewriter for low-quality hits.
* ``CrossEncoderReranker`` — local cross-encoder for result reranking.

The ``knowledge_server`` MCP tool module delegates to these classes
for its retrieval pipeline.
"""

from research_agent.rag.grader import RetrievalGrader
from research_agent.rag.query_rewriter import QueryRewriter
from research_agent.rag.reranker import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
)
from research_agent.rag.retriever import BM25Index, hybrid_rrf_fuse

__all__ = [
    "BM25Index",
    "CrossEncoderReranker",
    "DEFAULT_RERANKER_MODEL",
    "QueryRewriter",
    "RetrievalGrader",
    "hybrid_rrf_fuse",
]
