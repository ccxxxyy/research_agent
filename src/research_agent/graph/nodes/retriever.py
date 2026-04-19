"""Retriever node — searches knowledge base and web for relevant information."""

from __future__ import annotations

from loguru import logger

from research_agent.graph.state import ResearchPhase, ResearchState
from research_agent.rag.grader import RetrievalGrader
from research_agent.rag.query_rewriter import QueryRewriter
from research_agent.rag.retriever import HybridRetriever


async def retrieve_node(
    state: ResearchState,
    *,
    retriever: HybridRetriever,
) -> dict:
    """Execute hybrid search (vector + BM25) over the knowledge base."""
    query = state["query"]
    logger.info("Retriever: searching for query='{}'", query)

    queries = state.get("retrieval_queries") or [query]
    all_docs = []

    for q in queries:
        docs = await retriever.search(q, top_k=5)
        all_docs.extend(docs)

    return {
        "retrieved_documents": all_docs,
        "phase": ResearchPhase.RETRIEVING,
        "active_agent": "retriever",
    }


async def grade_retrieval_node(
    state: ResearchState,
    *,
    grader: RetrievalGrader,
) -> dict:
    """Evaluate whether retrieved documents are relevant to the query."""
    grade = await grader.grade(
        query=state["query"],
        documents=state.get("retrieved_documents", []),
    )

    logger.info("Grader: retrieval quality = {}", grade)

    retry_count = state.get("retrieval_retry_count", 0)
    return {
        "retrieval_grade": grade,
        "retrieval_retry_count": retry_count + (1 if grade == "irrelevant" else 0),
    }


async def rewrite_query_node(
    state: ResearchState,
    *,
    rewriter: QueryRewriter,
) -> dict:
    """Rewrite the query to improve retrieval quality (Corrective RAG)."""
    retrieval_queries = state.get("retrieval_queries", [])
    original = retrieval_queries[-1] if retrieval_queries else state["query"]
    grade = state.get("retrieval_grade", "")
    rewritten = await rewriter.rewrite(original, feedback=grade)

    logger.info("Rewriter: '{}' → '{}'", original, rewritten)

    return {
        "retrieval_queries": [*retrieval_queries, rewritten],
        "retrieved_documents": [],
    }
