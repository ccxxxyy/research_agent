"""Conditional edge functions for graph routing decisions."""

from __future__ import annotations

from typing import Literal

from research_agent.graph.state import ResearchState


def should_retry_retrieval(state: ResearchState) -> Literal["rewrite_query", "analyze"]:
    """After grading retrieval quality, decide whether to retry or proceed."""
    grade = state.get("retrieval_grade", "relevant")
    retry_count = state.get("retrieval_retry_count", 0)
    max_retries = state.get("max_retrieval_retries", 3)

    if grade == "irrelevant" and retry_count < max_retries:
        return "rewrite_query"
    return "analyze"


def should_reflect(state: ResearchState) -> Literal["reflect", "finalize"]:
    """After writing a draft, decide if reflection/revision is needed."""
    score = state.get("quality_score", 0.0)
    count = state.get("reflection_count", 0)
    max_ref = state.get("max_reflections", 3)

    if score >= 0.8 or count >= max_ref:
        return "finalize"
    return "reflect"


def check_error(state: ResearchState) -> Literal["handle_error", "continue"]:
    """Route to error handler if an error occurred."""
    if state.get("error"):
        return "handle_error"
    return "continue"
