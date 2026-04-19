"""Reasoner node — performs deep reasoning and quality evaluation (Reflection)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from loguru import logger

from research_agent.graph.state import ResearchPhase, ResearchState
from research_agent.llm.provider import ModelRouter
from research_agent.llm.tier import AgentName

REASONER_SYSTEM_PROMPT = """\
You are a critical reasoning expert. Your role is to:

1. Evaluate the analysis and draft report for logical consistency
2. Check for unsupported claims or hallucinations
3. Assess completeness — are all aspects of the query addressed?
4. Provide an overall quality score (0.0 to 1.0)
5. Give specific, actionable feedback for improvement

Respond in JSON format:
{
    "quality_score": 0.85,
    "reasoning": "Your detailed reasoning...",
    "feedback": "Specific improvement suggestions...",
    "issues": ["issue1", "issue2"]
}
"""


async def reasoner_node(
    state: ResearchState,
    *,
    model_router: ModelRouter,
) -> dict:
    """Evaluate the draft report quality and provide feedback for reflection."""
    reflection_count = state.get("reflection_count", 0)
    logger.info("Reasoner: evaluating draft (reflection round {})", reflection_count + 1)

    model = model_router.for_agent(AgentName.REASONER)
    parser = JsonOutputParser()

    messages = [
        SystemMessage(content=REASONER_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Original query: {state['query']}\n\n"
            f"Analysis:\n{state.get('analysis_result', '')}\n\n"
            f"Draft report:\n{state.get('draft_report', '')}\n\n"
            "Evaluate the quality and provide your assessment."
        ),
    ]

    response = await model.ainvoke(messages)
    raw_content = str(response.content)

    try:
        parsed = await parser.aparse(raw_content)
        score = float(parsed.get("quality_score", 0.0))
        feedback = parsed.get("feedback", "")
        reasoning = parsed.get("reasoning", "")
    except (KeyError, ValueError, TypeError):
        score = 0.5
        feedback = raw_content
        reasoning = ""

    logger.info("Reasoner: quality_score={:.2f}", score)

    return {
        "quality_score": score,
        "quality_feedback": feedback,
        "reasoning_result": reasoning,
        "reflection_count": reflection_count + 1,
        "phase": ResearchPhase.REFLECTING,
        "active_agent": "reasoner",
    }
