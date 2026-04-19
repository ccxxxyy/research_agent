"""FastAPI dependency injection — provides shared resources to route handlers."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from langgraph.graph.state import CompiledStateGraph

from research_agent.config import Settings
from research_agent.llm.provider import ModelRouter
from research_agent.memory.manager import MemoryManager


def get_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.graph


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
ModelRouterDep = Annotated[ModelRouter, Depends(get_model_router)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
MemoryDep = Annotated[MemoryManager, Depends(get_memory_manager)]
