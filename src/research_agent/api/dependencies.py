"""FastAPI dependency injection — provides shared resources to route handlers."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from langgraph.graph.state import CompiledStateGraph

from research_agent.config import Settings
from research_agent.llm.provider import ModelRouter
from research_agent.memory.manager import MemoryManager


def get_graph(request: Request) -> CompiledStateGraph:
    """Return the Phase-3 RAG-backed research graph.

    The graph is optional — if ``langchain_chroma`` is not installed
    (post FAISS migration) the lifespan leaves this attribute as
    ``None`` and we surface a 503 here. Use the Phase-4 ``/supervisor``
    or ``/knowledge`` routes instead, which are backed by FAISS.
    """
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Phase-3 research graph is not available: its Chroma "
                "vector store dependency was removed during the FAISS "
                "migration. Use POST /supervisor or the /knowledge "
                "endpoints (Phase-4.5 / 4.6) instead."
            ),
        )
    return graph


def get_supervisor_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.supervisor_graph


def get_research_supervisor_graph(request: Request) -> CompiledStateGraph:
    """Return the Phase-4.5 financial-research supervisor graph.

    The lifespan tries to compile this graph eagerly; if MCP tool
    discovery fails (e.g. network is down, ``uv`` is missing, etc.)
    the attribute is left unset and we surface a 503 here rather
    than an opaque ``AttributeError`` inside the route handler. A
    503 (vs. 500) signals to clients that the server is reachable
    but the downstream MCP dependency is not ready.
    """
    graph = getattr(request.app.state, "research_supervisor_graph", None)
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Research supervisor is not available: MCP tool "
                "discovery failed during application startup. Check "
                "the server logs for the underlying error."
            ),
        )
    return graph


def get_model_router(request: Request) -> ModelRouter:
    return request.app.state.model_router


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_memory_manager(request: Request) -> MemoryManager:
    store = getattr(request.app.state, "memory_store", None)
    if store is None:
        from langgraph.store.memory import InMemoryStore
        store = InMemoryStore()
    return MemoryManager(store)


GraphDep = Annotated[CompiledStateGraph, Depends(get_graph)]
SupervisorGraphDep = Annotated[CompiledStateGraph, Depends(get_supervisor_graph)]
ResearchSupervisorGraphDep = Annotated[
    CompiledStateGraph, Depends(get_research_supervisor_graph)
]
ModelRouterDep = Annotated[ModelRouter, Depends(get_model_router)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
MemoryDep = Annotated[MemoryManager, Depends(get_memory_manager)]
