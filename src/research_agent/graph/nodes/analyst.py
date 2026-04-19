"""Analyst node — extracts structured data and generates analysis."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_agent.graph.state import ResearchPhase, ResearchState
from research_agent.llm.provider import ModelRouter
from research_agent.llm.tier import AgentName

ANALYST_SYSTEM_PROMPT = """\
You are a senior data analyst specializing in deep research. Your role is to:

1. Analyze the retrieved documents and extract key data points
2. Identify patterns, trends, and relationships in the data
3. Produce a structured analysis with clear sections
4. Highlight any data gaps or inconsistencies

Output your analysis in a structured format with clear headings.
Cite specific data points from the source documents.
"""


async def analyst_node(
    state: ResearchState,
    *,
    model_router: ModelRouter,
) -> dict:
    """Analyze retrieved documents and extract structured insights."""
    retrieved_documents = state.get("retrieved_documents", [])
    logger.info("Analyst: processing {} documents", len(retrieved_documents))

    model = model_router.for_agent(AgentName.ANALYST)

    doc_texts = "\n\n---\n\n".join(
        f"[Document {i + 1}]\n{doc.page_content}"
        for i, doc in enumerate(retrieved_documents)
    )

    messages = [
        SystemMessage(content=ANALYST_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Research query: {state['query']}\n\n"
            f"Retrieved documents:\n{doc_texts}\n\n"
            "Provide your structured analysis."
        ),
    ]

    response = await model.ainvoke(messages)

    return {
        "analysis_result": str(response.content),
        "phase": ResearchPhase.ANALYZING,
        "active_agent": "analyst",
    }
