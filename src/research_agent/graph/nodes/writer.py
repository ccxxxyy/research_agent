"""Writer node — generates and refines the final research report."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_agent.graph.state import ResearchPhase, ResearchState
from research_agent.llm.provider import ModelRouter
from research_agent.llm.tier import AgentName

WRITER_SYSTEM_PROMPT = """\
You are an expert research report writer. Your role is to:

1. Synthesize the analysis into a well-structured research report
2. Present findings with clear logic flow and evidence
3. Include an executive summary, key findings, detailed analysis, and conclusions
4. Use professional tone appropriate for the research domain
5. Cite source documents where applicable

Structure your report with:
- Executive Summary
- Key Findings
- Detailed Analysis
- Conclusions & Recommendations
"""

REVISION_PROMPT_TEMPLATE = """\
Your previous draft received the following feedback:

Quality score: {score:.2f}/1.0
Feedback: {feedback}

Please revise the report to address these issues while maintaining the overall structure.
Only make targeted improvements — do not rewrite sections that were already good.

Previous draft:
{draft}
"""


async def writer_node(
    state: ResearchState,
    *,
    model_router: ModelRouter,
) -> dict:
    """Generate or revise the research report based on analysis and feedback."""
    model = model_router.for_agent(AgentName.WRITER)

    reflection_count = state.get("reflection_count", 0)
    draft = state.get("draft_report", "")
    is_revision = reflection_count > 0 and draft

    if is_revision:
        logger.info("Writer: revising draft (round {})", reflection_count)
        user_content = REVISION_PROMPT_TEMPLATE.format(
            score=state.get("quality_score", 0.0),
            feedback=state.get("quality_feedback", ""),
            draft=draft,
        )
    else:
        logger.info("Writer: generating initial draft")
        user_content = (
            f"Research query: {state['query']}\n\n"
            f"Analysis:\n{state.get('analysis_result', '')}\n\n"
        )
        human_feedback = state.get("human_feedback", "")
        if human_feedback:
            user_content += f"User feedback to incorporate:\n{human_feedback}\n\n"
        user_content += "Write the research report."

    messages = [
        SystemMessage(content=WRITER_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    response = await model.ainvoke(messages)

    return {
        "draft_report": str(response.content),
        "phase": ResearchPhase.WRITING,
        "active_agent": "writer",
    }


async def finalize_node(state: ResearchState) -> dict:
    """Promote the approved draft to final report."""
    logger.info(
        "Finalizer: report approved (score={:.2f}, rounds={})",
        state.get("quality_score", 0.0),
        state.get("reflection_count", 0),
    )
    return {
        "final_report": state.get("draft_report", ""),
        "phase": ResearchPhase.COMPLETED,
    }
