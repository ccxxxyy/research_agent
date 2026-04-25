"""Pydantic request/response schemas for all API endpoints."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------- Research ----------

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Research question")
    thread_id: str | None = Field(None, description="Resume an existing research thread")
    user_id: str = Field(default="anonymous", description="User identifier for memory")


class ResearchPhaseEvent(str, Enum):
    PLANNING = "planning"
    RETRIEVING = "retrieving"
    ANALYZING = "analyzing"
    WRITING = "writing"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchSSEEvent(BaseModel):
    """Server-Sent Event payload for streaming research progress."""
    phase: ResearchPhaseEvent
    agent: str = ""
    content: str = ""
    metadata: dict[str, Any] = {}


class ResearchResponse(BaseModel):
    thread_id: str
    status: str
    final_report: str = ""
    quality_score: float = 0.0
    reflection_rounds: int = 0
    usage: dict[str, Any] = {}


class ResearchStateResponse(BaseModel):
    thread_id: str
    current_phase: str
    next_nodes: list[str] = []
    completed_nodes: list[str] = []
    can_resume: bool = False


# ---------- Chat ----------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: str | None = None
    user_id: str = "anonymous"


class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    sources: list[dict[str, str]] = []


# ---------- Supervisor (Phase 3) ----------


class SupervisorChatRequest(BaseModel):
    """Request body for the minimal multi-agent supervisor."""

    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: str | None = Field(
        None,
        description="Conversation thread; omit to start a new isolated session.",
    )
    recursion_limit: int | None = Field(
        None,
        ge=4,
        le=50,
        description="Optional LangGraph recursion cap (defaults to framework default).",
    )


class SupervisorChatResponse(BaseModel):
    reply: str
    thread_id: str
    message_count: int = 0


# ---------- Research Supervisor (Phase 4.5) ----------


class ResearchSupervisorRequest(BaseModel):
    """Request body for the financial-research supervisor.

    Accepts a free-form Chinese or English question. The supervisor
    decides internally which specialists (``data_expert``,
    ``report_expert``, ``coder_expert``) to route the subtasks to.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Natural-language research question.",
    )
    thread_id: str | None = Field(
        None,
        description=(
            "Conversation thread; omit to start a new isolated "
            "session. Reusing the same thread_id across calls "
            "resumes the prior conversation from its checkpoint."
        ),
    )
    recursion_limit: int | None = Field(
        None,
        ge=4,
        le=80,
        description=(
            "Optional LangGraph recursion cap. The default (framework "
            "value of 25) is already generous for 3-specialist routes; "
            "raise only if you need >5 sequential hand-offs."
        ),
    )


class ResearchSupervisorResponse(BaseModel):
    """JSON response for the non-streaming research endpoint.

    ``specialists_reached`` is derived from the outer-state
    ``transfer_to_*`` hand-off trace. Because the graph compiles with
    ``output_mode="last_message"``, specialists' internal
    ``fin_*``/``pdf_*``/``code_*`` tool calls stay inside their
    subgraphs and are NOT surfaced here; the reply content itself is
    the canonical evidence that specialists did their work.
    """

    reply: str = Field(..., description="Final supervisor answer.")
    thread_id: str
    specialists_reached: list[str] = Field(
        default_factory=list,
        description="Distinct specialists the supervisor routed to.",
    )
    message_count: int = 0


class ResearchSupervisorSSEPhase(str, Enum):
    """Phase tags for SSE events emitted during streaming.

    * ``handoff``  — supervisor called ``transfer_to_<specialist>``.
    * ``update``   — a node produced a state update (specialist or
                     supervisor). Content is the most recent
                     assistant message from that update, truncated.
    * ``final``    — the supervisor produced its final user-visible
                     answer. Full content included.
    * ``error``    — graph invocation raised. Content is the short
                     error message; the client should stop consuming.
    * ``done``     — stream terminator. Always emitted last.
    """

    HANDOFF = "handoff"
    UPDATE = "update"
    FINAL = "final"
    ERROR = "error"
    DONE = "done"


class ResearchSupervisorSSEEvent(BaseModel):
    """Single SSE event payload for research-supervisor streaming."""

    phase: ResearchSupervisorSSEPhase
    node: str = Field(
        "",
        description="Graph node that produced the update (e.g. supervisor / data_expert).",
    )
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------- Knowledge ----------

class KnowledgeUploadRequest(BaseModel):
    collection: str = "default"


class KnowledgeDocument(BaseModel):
    id: str
    content_preview: str
    metadata: dict[str, Any] = {}
    created_at: datetime | None = None


class KnowledgeListResponse(BaseModel):
    collection: str
    total: int
    documents: list[KnowledgeDocument]


# ---------- Health ----------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = ""
    services: dict[str, str] = {}


# ---------- Resume / Approve ----------

class ResumeRequest(BaseModel):
    """Resume a paused or failed research task."""
    pass


class ApproveRequest(BaseModel):
    """Approve a paused research task and optionally inject feedback."""
    feedback: str = ""
