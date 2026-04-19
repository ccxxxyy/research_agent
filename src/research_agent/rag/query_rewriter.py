"""Query rewriter — reformulates queries when retrieval quality is low."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_agent.llm.provider import ModelRouter
from research_agent.llm.tier import AgentName

REWRITER_SYSTEM_PROMPT = """\
You are a search query optimization expert. When a retrieval query returns
irrelevant results, you rewrite it to improve search quality.

Strategies:
1. Expand ambiguous terms with more specific keywords
2. Add domain-specific terminology
3. Break compound queries into focused sub-queries
4. Remove noise words that dilute search relevance
5. Try alternative phrasings or synonyms

Respond with ONLY the rewritten query, nothing else.
"""


class QueryRewriter:
    """Rewrites search queries to improve retrieval quality."""

    def __init__(self, model_router: ModelRouter) -> None:
        self._model = model_router.for_agent(AgentName.QUERY_REWRITER)

    async def rewrite(self, query: str, feedback: str = "") -> str:
        """Rewrite a query based on retrieval feedback."""
        context = f"Original query: {query}"
        if feedback:
            context += f"\nRetrieval feedback: {feedback}"
        context += "\n\nRewrite this query to improve search relevance."

        messages = [
            SystemMessage(content=REWRITER_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]

        response = await self._model.ainvoke(messages)
        rewritten = response.content.strip().strip('"').strip("'")

        logger.info("Query rewritten: '{}' → '{}'", query[:50], rewritten[:50])
        return rewritten
