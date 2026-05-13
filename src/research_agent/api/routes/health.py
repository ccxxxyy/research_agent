"""Health check endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request

from research_agent import __version__
from research_agent.api.schemas import HealthResponse
from research_agent.config import get_settings

router = APIRouter(tags=["health"])

_KNOWLEDGE_DB_DIR = Path("data/knowledge_db")


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Probe real service dependencies and report honest status.

    Reads settings via the module-level ``get_settings()`` (an
    ``@lru_cache``'d factory) rather than through ``request.app.state``
    so the probe stays usable when called BEFORE the ASGI lifespan has
    run — for example under ``httpx.ASGITransport`` in integration
    tests, or by a Kubernetes startup probe that fires before the
    lifespan ``on_startup`` hook has finished initialising
    long-running resources.
    """
    services: dict[str, str] = {}

    # Postgres — reuse the lightweight TCP probe
    from research_agent.memory._pg_reachability import is_postgres_reachable

    settings = get_settings()
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

    # Research supervisor graph availability. ``app.state`` may not
    # exist yet (lifespan hasn't run) — ``getattr`` makes this safe.
    graph = getattr(request.app.state, "research_supervisor_graph", None)
    services["research_supervisor"] = "ok" if graph is not None else "unavailable"

    # Checkpointer backend — report which tier is active so operators
    # know when short-term memory has silently degraded to in-memory.
    checkpointer = getattr(request.app.state, "checkpointer", None)
    if checkpointer is not None:
        backend = type(checkpointer).__name__
        services["checkpointer"] = f"ok ({backend})"
    else:
        services["checkpointer"] = "unavailable"

    # Memory store backend — same rationale as checkpointer.
    memory_store = getattr(request.app.state, "memory_store", None)
    if memory_store is not None:
        backend = type(memory_store).__name__
        services["memory_store"] = f"ok ({backend})"
    else:
        services["memory_store"] = "unavailable"

    # Aggregate liveness. Postgres/Redis are first-class production
    # dependencies; the rest are advisory. We report ``ok`` as long as
    # the data plane is up; missing optional services degrade the
    # response but do not flip a green dashboard to red.
    critical = ("postgres", "redis")
    overall = (
        "ok" if all(services.get(k) == "ok" for k in critical) else "degraded"
    )

    return HealthResponse(
        status=overall,
        version=__version__,
        services=services,
    )
