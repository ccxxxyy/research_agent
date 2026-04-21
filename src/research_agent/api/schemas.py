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
