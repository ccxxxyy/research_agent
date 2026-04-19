"""Model tier definitions and agent-to-model mapping."""

from __future__ import annotations

from enum import Enum


class ModelTier(str, Enum):
    """Task complexity tiers for model routing."""

    LIGHT = "light"    # Classification, extraction, formatting, grading
    MEDIUM = "medium"  # Summarization, analysis, evaluation
    HEAVY = "heavy"    # Deep reasoning, report writing, planning


class AgentName(str, Enum):
    """Registered agent identifiers for model routing."""

    SUPERVISOR = "supervisor"
    RETRIEVER = "retriever"
    ANALYST = "analyst"
    REASONER = "reasoner"
    WRITER = "writer"
    RAG_GRADER = "rag_grader"
    QUERY_REWRITER = "query_rewriter"


AGENT_TIER_MAP: dict[AgentName, ModelTier] = {
    AgentName.SUPERVISOR: ModelTier.HEAVY,
    AgentName.RETRIEVER: ModelTier.LIGHT,
    AgentName.ANALYST: ModelTier.MEDIUM,
    AgentName.REASONER: ModelTier.HEAVY,
    AgentName.WRITER: ModelTier.HEAVY,
    AgentName.RAG_GRADER: ModelTier.LIGHT,
    AgentName.QUERY_REWRITER: ModelTier.LIGHT,
}

FALLBACK_CHAIN: dict[ModelTier, ModelTier] = {
    ModelTier.HEAVY: ModelTier.MEDIUM,
    ModelTier.MEDIUM: ModelTier.LIGHT,
}
