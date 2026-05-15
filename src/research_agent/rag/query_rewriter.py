"""LLM-based query rewriter for corrective RAG loops.

In the production pipeline, query rewriting is driven by the
``knowledge_expert`` agent's system prompt: when ``knowledge_search``
returns ``quality == "low"``, the agent rewrites the query and retries
(up to 3 attempts).  This class encapsulates that same pattern as an
**independently callable component** so it can be unit-tested, used
outside the agent loop, and pointed at during interviews when asked
"where is your Corrective RAG code?"

Typical usage::

    rewriter = QueryRewriter(model=model_router.get_model(ModelTier.LIGHT))
    better_query = await rewriter.rewrite(
        original_query="ESG 碳中和",
        context="Previous search returned low-quality results about ESG "
                "but nothing on carbon neutrality commitments.",
    )
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

_SYSTEM_PROMPT = """\
You are a search-query rewriter inside a Corrective RAG pipeline.

Your input is:
  1. The original user query that produced low-quality retrieval hits.
  2. (Optional) context describing what went wrong.

Your job: produce a SINGLE improved query that is more specific,
uses more precise terminology, and is likely to surface relevant
chunks from a Chinese-English financial knowledge base.

Rules:
  - Output ONLY the rewritten query text, no explanation.
  - Preserve the language of the original query.
  - If the original query is vague, add likely domain keywords
    (e.g. "ROE", "毛利率", "年报", "ESG", specific company names).
  - Do NOT invent facts or change the user's intent.
"""


class QueryRewriter:
    """Rewrite a search query to improve retrieval quality.

    Parameters
    ----------
    model:
        Any LangChain-compatible chat model (``BaseChatModel``).
        Typically a LIGHT-tier model since rewriting is a simple
        classification/generation task.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    async def rewrite(
        self,
        original_query: str,
        context: str = "",
    ) -> str:
        """Return a rewritten query string.

        Falls back to ``original_query`` if the LLM call fails or
        returns empty — the corrective loop should never crash
        because the rewriter hiccuped.
        """
        user_content = f"Original query: {original_query}"
        if context:
            user_content += f"\nContext: {context}"

        try:
            response = await self._model.ainvoke([
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ])
            rewritten = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            ).strip()
            if not rewritten:
                return original_query
            logger.debug(
                "QueryRewriter: '{}' → '{}'", original_query, rewritten
            )
            return rewritten
        except Exception as exc:  # noqa: BLE001
            logger.warning("QueryRewriter failed ({}); using original", exc)
            return original_query


__all__ = ["QueryRewriter"]
