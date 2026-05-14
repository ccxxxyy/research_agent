"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from research_agent.api.routes import health, knowledge, memory, sentiment, supervisor
from research_agent.config import get_settings
from research_agent.observability.logging import setup_logging


async def _try_build_research_supervisor(model_router, checkpointer, settings=None):
    """Best-effort compile of the Phase-4.5 / 4.6 research supervisor.

    Tool discovery runs four loaders in parallel: three MCP-stdio
    subprocesses (fin_data, pdf_report, code) plus the in-process
    knowledge-tools loader. If any of them fails (missing dependency,
    network timeout, etc.) we degrade gracefully: the surviving
    specialists are still wired in, and ``get_research_supervisor_graph``
    surfaces a 503 only when **every** specialist's tool discovery
    failed. Startup never dies because of an unavailable backend.

    Returns the compiled graph, or ``None`` if no specialist could
    be loaded.
    """
    from research_agent.graph.research_supervisor import build_research_supervisor
    from research_agent.mcp_servers.client_factory import (
        load_code_server_tools,
        load_fin_data_server_tools,
        load_knowledge_tools_inproc,
        load_news_sentiment_server_tools,
        load_news_server_tools,
        load_pdf_report_server_tools,
    )

    # NOTE: ``load_knowledge_tools_inproc`` is the in-process replacement
    # for the (deprecated) MCP-stdio ``load_knowledge_server_tools``.
    # The other four loaders still spawn MCP subprocesses — those
    # servers' import chains are light enough that the stdio path is
    # stable. See ``knowledge_server.py`` for why knowledge is special.
    timeout = float(getattr(settings, "mcp_tool_discovery_timeout", 30.0))
    results = await asyncio.gather(
        asyncio.wait_for(load_fin_data_server_tools(), timeout=timeout),
        asyncio.wait_for(load_pdf_report_server_tools(), timeout=timeout),
        asyncio.wait_for(load_code_server_tools(), timeout=timeout),
        asyncio.wait_for(load_knowledge_tools_inproc(), timeout=timeout),
        asyncio.wait_for(load_news_server_tools(), timeout=timeout),
        asyncio.wait_for(load_news_sentiment_server_tools(), timeout=timeout),
        return_exceptions=True,
    )
    names = (
        "fin_data_server",
        "pdf_report_server",
        "code_server",
        "knowledge_tools_inproc",
        "news_server",
        "news_sentiment_server",
    )
    tools: dict[str, list] = {}
    for name, r in zip(names, results):
        if isinstance(r, Exception):
            logger.warning("Tool discovery failed for {}: {}", name, r)
            tools[name] = []
        else:
            tools[name] = list(r)
            logger.info("Tools discovered for {}: {}", name, len(tools[name]))

    if not any(tools.values()):
        logger.error(
            "All tool sources failed to provide tools; "
            "research supervisor will be unavailable."
        )
        return None

    # ``settings`` is optional so this helper stays unit-testable in
    # isolation. In the production lifespan we always pass it.
    reflect = bool(getattr(settings, "reflection_enabled", False))
    pass_threshold = float(getattr(settings, "reflection_pass_threshold", 0.85))
    max_iter = int(getattr(settings, "reflection_max_iterations", 2))

    try:
        return build_research_supervisor(
            model_router=model_router,
            data_tools=tools["fin_data_server"] or None,
            report_tools=tools["pdf_report_server"] or None,
            coder_tools=tools["code_server"] or None,
            knowledge_tools=tools["knowledge_tools_inproc"] or None,
            news_tools=tools["news_server"] or None,
            sentiment_tools=tools["news_sentiment_server"] or None,
            checkpointer=checkpointer,
            enable_reflection=reflect,
            reflection_pass_threshold=pass_threshold,
            reflection_max_iterations=max_iter,
        )
    except Exception:  # noqa: BLE001
        # A crash here (e.g. misconfigured model router) should not
        # take down the entire API — the minimal supervisor and RAG
        # pipeline may still be usable.
        logger.exception("Failed to compile research_supervisor; route will 503.")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and tear down application resources."""
    settings = get_settings()
    setup_logging(
        settings.observability.log_level,
        log_file_path=settings.observability.log_file_path,
    )

    from research_agent.memory.checkpointer import init_checkpointer
    from research_agent.memory.store import init_memory_store

    checkpoint_sqlite = settings.checkpoint_sqlite_path.strip()
    checkpoint_sqlite_arg: Path | str | None = (
        checkpoint_sqlite if checkpoint_sqlite else None
    )
    store_sqlite = settings.memory_store_sqlite_path.strip()
    store_sqlite_arg: Path | str | None = (
        store_sqlite if store_sqlite else None
    )

    checkpointer = await init_checkpointer(
        settings.database.postgres_sync_uri,
        sqlite_path=checkpoint_sqlite_arg,
    )
    memory_store = await init_memory_store(
        settings.database.postgres_sync_uri,
        sqlite_path=store_sqlite_arg,
    )

    from research_agent.graph.minimal_supervisor import build_minimal_supervisor
    from research_agent.llm.provider import ModelRouter

    model_router = ModelRouter(settings.llm)

    supervisor_graph = build_minimal_supervisor(
        model_router=model_router,
        checkpointer=checkpointer,
    )

    research_supervisor_graph = await _try_build_research_supervisor(
        model_router=model_router,
        checkpointer=checkpointer,
        settings=settings,
    )

    app.state.supervisor_graph = supervisor_graph
    app.state.research_supervisor_graph = research_supervisor_graph
    app.state.model_router = model_router
    app.state.memory_store = memory_store
    app.state.checkpointer = checkpointer
    app.state.settings = settings

    yield

    # --- Graceful shutdown: release resources in reverse order ---
    logger.info("Shutting down: releasing resources...")

    # Close memory store connection pool
    if hasattr(memory_store, "conn") and hasattr(memory_store.conn, "close"):
        try:
            memory_store.conn.close()
            logger.info("Memory store connection pool closed.")
        except Exception as e:
            logger.warning("Error closing memory store pool: {}", e)

    # Close checkpointer connection pool / sqlite connection
    if hasattr(checkpointer, "conn"):
        conn = checkpointer.conn
        if hasattr(conn, "close"):
            try:
                if asyncio.iscoroutinefunction(getattr(conn, "close", None)):
                    await conn.close()
                else:
                    conn.close()
                logger.info("Checkpointer connection closed.")
            except Exception as e:
                logger.warning("Error closing checkpointer connection: {}", e)

    logger.info("Shutdown complete.")


def _parse_cors_origins(raw: str) -> list[str]:
    """Parse comma-separated CORS origins or wildcard."""
    if raw.strip() == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Research Agent",
        description="Multi-agent deep research system with LangGraph, MCP, and Agentic RAG",
        version="0.1.0",
        lifespan=lifespan,
    )

    settings = get_settings()
    origins = _parse_cors_origins(settings.cors_origins)

    # Middleware stack (execution order is bottom-to-top). Outermost
    # middleware runs first: RequestTimeout → Auth → RateLimit → CORS
    # → route handler.
    app.add_middleware(
        CORSMiddleware,  # type: ignore[arg-type]
        allow_origins=origins,
        allow_credentials=origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from research_agent.api.middleware import (
        AuthMiddleware,
        RateLimitMiddleware,
        RequestTimeoutMiddleware,
    )

    app.add_middleware(
        RequestTimeoutMiddleware,
        timeout_seconds=float(settings.http_request_timeout_seconds),
    )
    app.add_middleware(
        RateLimitMiddleware,
        max_rpm=settings.rate_limit_rpm,
        redis_url=settings.database.redis_url or None,
    )
    app.add_middleware(AuthMiddleware, secret_key=settings.api_secret_key)

    app.include_router(health.router)
    app.include_router(knowledge.router)
    app.include_router(memory.router)
    app.include_router(sentiment.router)
    app.include_router(supervisor.router)

    return app


app = create_app()


def cli() -> None:
    settings = get_settings()
    uvicorn.run(
        "research_agent.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.is_dev,
    )


if __name__ == "__main__":
    cli()
