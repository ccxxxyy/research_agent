"""Long-term memory management endpoints.

Exposes the MemoryManager as a REST surface so frontends can:
- View a user's research history
- Set / update user preferences
- Query accumulated domain knowledge

These endpoints complement the automatic memory lifecycle in the
supervisor routes (which save research results on completion).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

from research_agent.api.dependencies import MemoryDep
from research_agent.memory.manager import MemoryNamespace

router = APIRouter(prefix="/api/memory", tags=["memory"])


# =====================================================================
# Request / Response models
# =====================================================================


class SavePreferenceRequest(BaseModel):
    key: str = Field(..., min_length=1, max_length=128)
    content: str = Field(..., min_length=1, max_length=2000)


class SavePreferenceResponse(BaseModel):
    user_id: str
    key: str
    saved: bool = True


class MemoryItem(BaseModel):
    key: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class ResearchHistoryResponse(BaseModel):
    user_id: str
    items: list[dict[str, Any]]


class UserContextResponse(BaseModel):
    user_id: str
    preferences: list[dict[str, Any]]
    recent_research: list[dict[str, Any]]


# =====================================================================
# Endpoints
# =====================================================================


@router.get("/context", response_model=UserContextResponse)
async def get_user_context(
    memory: MemoryDep,
    x_user_id: str = Header(..., alias="X-User-ID"),
) -> UserContextResponse:
    """Retrieve full user context (preferences + recent research history).

    This is the same context the research supervisor injects as a
    preamble before each invocation.
    """
    ctx = await memory.get_user_context(x_user_id)
    return UserContextResponse(
        user_id=x_user_id,
        preferences=ctx.get("preferences", []),
        recent_research=ctx.get("recent_research", []),
    )


@router.get("/history", response_model=ResearchHistoryResponse)
async def get_research_history(
    memory: MemoryDep,
    x_user_id: str = Header(..., alias="X-User-ID"),
    limit: int = 10,
) -> ResearchHistoryResponse:
    """List the user's saved research results (most recent first)."""
    items = await memory.search_memories(
        user_id=x_user_id,
        namespace=MemoryNamespace.RESEARCH_HISTORY,
        limit=min(limit, 50),
    )
    return ResearchHistoryResponse(user_id=x_user_id, items=items)


@router.post(
    "/preferences",
    response_model=SavePreferenceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def save_preference(
    body: SavePreferenceRequest,
    memory: MemoryDep,
    x_user_id: str = Header(..., alias="X-User-ID"),
) -> SavePreferenceResponse:
    """Save or update a user preference.

    Preferences are injected into the supervisor prompt so it can
    personalize answers (e.g. preferred language, analysis depth,
    favorite sectors).
    """
    await memory.save_memory(
        user_id=x_user_id,
        namespace=MemoryNamespace.USER_PREFERENCES,
        key=body.key,
        value={"content": body.content},
    )
    return SavePreferenceResponse(user_id=x_user_id, key=body.key)


@router.delete("/preferences/{key}")
async def delete_preference(
    key: str,
    memory: MemoryDep,
    x_user_id: str = Header(..., alias="X-User-ID"),
) -> dict[str, Any]:
    """Remove a user preference by key."""
    try:
        ns = (x_user_id, MemoryNamespace.USER_PREFERENCES)
        await memory._store.adelete(ns, key)
        return {"user_id": x_user_id, "key": key, "deleted": True}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete preference: {exc}",
        ) from exc
