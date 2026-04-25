"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from research_agent.api.routes import chat, health, knowledge, research, supervisor
from research_agent.config import get_settings
from research_agent.observability.logging import setup_logging


async def _try_build_research_supervisor(model_router, checkpointer):
    """Best-effort compile of the Phase-4.5 research supervisor.

    MCP tool discovery spawns three stdio subprocesses in parallel.
    If any of them fails (missing dependency, network timeout, etc.)
    we degrade gracefully: the surviving specialists are still
    wired in, and ``get_research_supervisor_graph`` surfaces a 503
    only when *all three* failed. Startup never dies because of MCP.

    Returns the compiled graph, or ``None`` if no specialist could
    be loaded.
    """
    from research_agent.graph.research_supervisor import build_research_supervisor
    from research_agent.mcp_servers.client_factory import (
        load_code_server_tools,
        load_fin_data_server_tools,
        load_pdf_report_server_tools,
    )

    results = await asyncio.gather(
        load_fin_data_server_tools(),
        load_pdf_report_server_tools(),
        load_code_server_tools(),
        return_exceptions=True,
    )
    names = ("fin_data_server", "pdf_report_server", "code_server")
    tools: dict[str, list] = {}
    for name, r in zip(names, results):
        if isinstance(r, Exception):
            logger.warning("MCP tool discovery failed for {}: {}", name, r)
            tools[name] = []
        else:
            tools[name] = list(r)
            logger.info("MCP tools discovered for {}: {}", name, len(tools[name]))

    if not any(tools.values()):
        logger.error(
            "All three MCP servers failed to provide tools; "
            "research supervisor will be unavailable."
        )
        return None

    try:
        return build_research_supervisor(
            model_router=model_router,
            data_tools=tools["fin_data_server"] or None,
            report_tools=tools["pdf_report_server"] or None,
            coder_tools=tools["code_server"] or None,
            checkpointer=checkpointer,
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
    setup_logging(settings.observability.log_level)

    from research_agent.memory.checkpointer import init_checkpointer
    from research_agent.memory.store import init_memory_store

    checkpointer = await init_checkpointer(settings.database.postgres_sync_uri)
    memory_store = await init_memory_store(settings.database.postgres_sync_uri)

    from research_agent.graph.minimal_supervisor import build_minimal_supervisor
    from research_agent.graph.supervisor import build_research_graph
    from research_agent.llm.provider import ModelRouter
    from research_agent.rag.embedder import create_embeddings
    from research_agent.rag.retriever import HybridRetriever

    from langchain_chroma import Chroma

    model_router = ModelRouter(settings.llm)

    embeddings = create_embeddings(settings.llm)
    vectorstore = Chroma(
        collection_name="research_agent",
        embedding_function=embeddings,
    )
    hybrid_retriever = HybridRetriever(vectorstore=vectorstore)

    graph = build_research_graph(
        model_router=model_router,
        hybrid_retriever=hybrid_retriever,
        checkpointer=checkpointer,
        memory_store=memory_store,
    )

    supervisor_graph = build_minimal_supervisor(
        model_router=model_router,
        checkpointer=checkpointer,
    )

    research_supervisor_graph = await _try_build_research_supervisor(
        model_router=model_router, checkpointer=checkpointer
    )

    app.state.graph = graph
    app.state.supervisor_graph = supervisor_graph
    app.state.research_supervisor_graph = research_supervisor_graph
    app.state.model_router = model_router
    app.state.memory_store = memory_store
    app.state.settings = settings

    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Research Agent",
        description="Multi-agent deep research system with LangGraph, MCP, and Agentic RAG",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,  # type: ignore[arg-type]
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(research.router)
    app.include_router(chat.router)
    app.include_router(knowledge.router)
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
