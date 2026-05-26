"""RAG 构建模块 — 检索、评分、查询重写与重排序。

本包将 Corrective-RAG 流水线的核心组件导出为可独立测试的类：

* ``BM25Index`` — 基于分词文档的稀疏 BM25 索引。
* ``hybrid_rrf_fuse`` — 稠密检索 + 稀疏检索结果的加权 RRF（Reciprocal-Rank Fusion）融合。
* ``RetrievalGrader`` — 三档质量分类器（high / medium / low），驱动纠正循环。
* ``QueryRewriter`` — 基于 LLM 的查询重写器，用于低质量命中场景。
* ``CrossEncoderReranker`` — 本地交叉编码器，用于结果重排序。

``knowledge_server`` MCP 工具模块将检索流水线委托给这些类。
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
