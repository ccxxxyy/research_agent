"""Shared graph state flowing through the multi-agent pipeline."""

from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, Any, NotRequired

from langchain_core.documents import Document
from langgraph.graph import MessagesState


class ResearchPhase(str, Enum):
    """Current phase of the research pipeline."""

    PLANNING = "planning"
    RETRIEVING = "retrieving"
    ANALYZING = "analyzing"
    REASONING = "reasoning"
    WRITING = "writing"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchState(MessagesState):
    """Shared state across all agents in the research graph.

    Extends MessagesState to get built-in message list handling with
    the add-only reducer (messages are appended, never overwritten).

    All fields beyond ``messages`` (inherited) are optional because the
    graph populates them progressively as it executes.
    """

    # Task definition
    query: str
    research_plan: NotRequired[list[str]]

    # Current execution phase (for SSE streaming)
    phase: NotRequired[ResearchPhase]

    # Retrieval results
    retrieved_documents: NotRequired[Annotated[list[Document], operator.add]]
    retrieval_queries: NotRequired[list[str]]

    # RAG quality control
    retrieval_grade: NotRequired[str]
    retrieval_retry_count: NotRequired[int]
    max_retrieval_retries: NotRequired[int]

    # Analysis results
    analysis_result: NotRequired[str]
    extracted_data: NotRequired[dict[str, Any]]

    # Reasoning / reflection
    reasoning_result: NotRequired[str]
    quality_score: NotRequired[float]
    quality_feedback: NotRequired[str]
    reflection_count: NotRequired[int]
    max_reflections: NotRequired[int]

    # Final output
    draft_report: NotRequired[str]
    final_report: NotRequired[str]

    # Human-in-the-loop
    human_feedback: NotRequired[str]
    requires_approval: NotRequired[bool]

    # Metadata
    active_agent: NotRequired[str]
    error: NotRequired[str]
