"""Reranker — re-scores retrieved documents for improved relevance ordering."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from loguru import logger

from research_agent.llm.provider import ModelRouter
from research_agent.llm.tier import AgentName

RERANKER_PROMPT = """\
You are a document relevance scorer. Given a query and a document,
score the document's relevance from 0.0 (completely irrelevant) to 1.0
(perfectly relevant).

Respond in JSON: {"score": 0.85, "reason": "brief explanation"}
"""


class LLMReranker:
    """LLM-based reranker that re-scores documents against the query.

    For production, consider replacing with a cross-encoder model
    (e.g. BAAI/bge-reranker-base) for better latency.
    """

    def __init__(self, model_router: ModelRouter, top_k: int = 5) -> None:
        self._model = model_router.for_agent(AgentName.RAG_GRADER)
        self._parser = JsonOutputParser()
        self._top_k = top_k

    async def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """Re-score and re-order documents by relevance to query."""
        if len(documents) <= 1:
            return documents

        scored: list[tuple[float, Document]] = []

        for doc in documents:
            score = await self._score_document(query, doc)
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [doc for _, doc in scored[:self._top_k]]

        logger.info(
            "Reranked {} docs → top {} (scores: {:.2f} ~ {:.2f})",
            len(documents), len(results),
            scored[0][0] if scored else 0,
            scored[-1][0] if scored else 0,
        )
        return results

    async def _score_document(self, query: str, doc: Document) -> float:
        messages = [
            SystemMessage(content=RERANKER_PROMPT),
            HumanMessage(
                content=f"Query: {query}\n\nDocument:\n{doc.page_content[:800]}"
            ),
        ]
        response = await self._model.ainvoke(messages)
        try:
            parsed = await self._parser.aparse(response.content)
            return float(parsed.get("score", 0.0))
        except Exception:
            return 0.5
