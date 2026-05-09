"""Health check endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request

from research_agent import __version__
from research_agent.api.schemas import HealthResponse

router = APIRouter(tags=["health"])

_KNOWLEDGE_DB_DIR = Path("data/knowledge_db")


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Probe real service dependencies and report honest status."""
    services: dict[str, str] = {}

    # Postgres — reuse the lightweight TCP probe
    from research_agent.memory._pg_reachability import is_postgres_reachable

    settings = request.app.state.settings
    pg_uri = settings.database.postgres_sync_uri
    services["postgres"] = "ok" if is_postgres_reachable(pg_uri) else "unreachable"

    # Redis
    try:
        import redis as redis_lib

        r = redis_lib.from_url(settings.database.redis_url, socket_connect_timeout=2)
        r.ping()
        services["redis"] = "ok"
    except Exception:
        services["redis"] = "unreachable"

    # Knowledge DB (FAISS directory exists)
    services["knowledge_db"] = "ok" if _KNOWLEDGE_DB_DIR.is_dir() else "not_initialized"

    # Research supervisor graph availability
    graph = getattr(request.app.state, "research_supervisor_graph", None)
    services["research_supervisor"] = "ok" if graph is not None else "unavailable"

    overall = "ok" if all(v == "ok" for v in services.values()) else "degraded"

    return HealthResponse(
        status=overall,
        version=__version__,
        services=services,
    )
