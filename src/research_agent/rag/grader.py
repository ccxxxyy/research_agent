"""Retrieval quality grader — core of Corrective RAG."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from loguru import logger

from research_agent.llm.provider import ModelRouter
from research_agent.llm.tier import AgentName

GRADER_SYSTEM_PROMPT = """\
You are a retrieval quality evaluator. Given a user query and a list of
retrieved documents, assess whether the documents contain information
relevant to answering the query.

Respond in JSON:
{
    "grade": "relevant" or "irrelevant",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of your assessment"
}

Criteria for "relevant":
- Documents contain factual information that directly addresses the query
- Documents provide sufficient context to generate a meaningful answer
- At least 2 out of the retrieved documents are on-topic

Criteria for "irrelevant":
- Documents are off-topic or only tangentially related
- Information is too vague or outdated to be useful
- Retrieved content appears to be noise or boilerplate
"""


class RetrievalGrader:
    """Evaluates retrieval quality and triggers corrective actions."""

    def __init__(self, model_router: ModelRouter) -> None:
        self._model = model_router.for_agent(AgentName.RAG_GRADER)
        self._parser = JsonOutputParser()

    async def grade(self, query: str, documents: list[Document]) -> str:
        """Grade retrieved documents as 'relevant' or 'irrelevant'."""
        if not documents:
            logger.warning("No documents to grade")
            return "irrelevant"

        doc_texts = "\n\n---\n\n".join(
            f"[Doc {i + 1}] {doc.page_content[:500]}"
            for i, doc in enumerate(documents[:10])
        )

        messages = [
            SystemMessage(content=GRADER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Query: {query}\n\nRetrieved documents:\n{doc_texts}"
            ),
        ]

        response = await self._model.ainvoke(messages)

        try:
            parsed = await self._parser.aparse(response.content)
            grade = parsed.get("grade", "irrelevant")
            confidence = parsed.get("confidence", 0.0)
            logger.info("Retrieval grade: {} (confidence={:.2f})", grade, confidence)
            return grade
        except Exception:
            logger.warning("Failed to parse grader output, defaulting to 'relevant'")
            return "relevant"
