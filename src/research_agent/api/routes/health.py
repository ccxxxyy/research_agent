"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from research_agent import __version__
from research_agent.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    # TODO: check postgres / redis / chroma connectivity
    return HealthResponse(
        status="ok",
        version=__version__,
        services={
            "postgres": "ok",
            "redis": "ok",
            "chroma": "ok",
        },
    )
