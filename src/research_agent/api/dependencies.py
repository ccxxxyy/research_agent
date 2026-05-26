"""FastAPI 依赖注入 - 为路由处理器提供共享资源。"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from langgraph.graph.state import CompiledStateGraph

from research_agent.config import Settings
from research_agent.llm.provider import ModelRouter
from research_agent.memory.manager import MemoryManager


def get_supervisor_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.supervisor_graph


def get_research_supervisor_graph(request: Request) -> CompiledStateGraph:
    """返回金融研究 supervisor 图。

    lifespan 会尽早编译此图；若 MCP 工具发现失败（如网络断开、``uv``未安装等），该属性将保持未设置状态，此处返回 503 而非路由处理器内部不透明的 ``AttributeError``。
    503（非 500）向客户端表明服务器可达但下游 MCP 依赖尚未就绪。
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
        from loguru import logger

        store = InMemoryStore()
        request.app.state.memory_store = store
        logger.warning(
            "memory_store was not initialised by lifespan; "
            "created a shared InMemoryStore fallback (non-persistent)."
        )
    return MemoryManager(store)


SupervisorGraphDep = Annotated[CompiledStateGraph, Depends(get_supervisor_graph)]
ResearchSupervisorGraphDep = Annotated[
    CompiledStateGraph, Depends(get_research_supervisor_graph)
]
ModelRouterDep = Annotated[ModelRouter, Depends(get_model_router)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
MemoryDep = Annotated[MemoryManager, Depends(get_memory_manager)]
