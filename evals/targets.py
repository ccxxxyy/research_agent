"""评估目标：运行真实的 supervisor 图并返回结构化输出。

目标完全绕过 FastAPI — 直接调用 ``graph.ainvoke``，因此评估运行不受认证、速率限制或 HTTP 开销影响。

单次 ``build_eval_environment`` 调用在每个评估会话中初始化图 + 记忆一次；``supervisor_target`` 是 ``langsmith.evaluate`` 逐样本调用的可调用对象。
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

from research_agent.api.routes.supervisor import (
    _final_assistant_text,
    _specialists_reached,
)
from research_agent.memory.manager import MemoryManager

_GRAPH = None
_MEMORY: MemoryManager | None = None


async def build_eval_environment() -> None:
    """一次性初始化：编译研究 supervisor 和记忆管理器。

    必须在首次 ``supervisor_target`` 调用之前执行。使用与生产 lifespan 相同的 ``_try_build_research_supervisor`` 路径，以确保图拓扑完全一致。
    """
    global _GRAPH, _MEMORY  # noqa: PLW0603

    from research_agent.config import get_settings
    from research_agent.llm.provider import ModelRouter
    from research_agent.main import _try_build_research_supervisor
    from research_agent.memory.checkpointer import init_checkpointer

    settings = get_settings()
    model_router = ModelRouter(settings.llm)
    checkpointer = await init_checkpointer(settings.database.postgres_sync_uri)
    store = InMemoryStore()

    _GRAPH, _ = await _try_build_research_supervisor(
        model_router=model_router,
        checkpointer=checkpointer,
        settings=settings,
    )
    if _GRAPH is None:
        raise RuntimeError(
            "Failed to build research supervisor graph for evaluation. "
            "Check MCP tool availability and LLM API keys."
        )
    _MEMORY = MemoryManager(store)


async def supervisor_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """通过 supervisor 运行一个评估样本。

    Args:
        inputs: 匹配数据集 schema 的字典 —
            ``{"query": str, "user_id": str, ...}``。

    Returns:
        ``{"reply": str, "specialists_reached": list[str],
          "memory_saved": bool, "thread_id": str}``
    """
    if _GRAPH is None or _MEMORY is None:
        raise RuntimeError("Call build_eval_environment() before running targets.")

    query = inputs["query"]
    user_id = inputs.get("user_id", "anonymous")
    thread_id = str(uuid.uuid4())

    config: dict = {"configurable": {"thread_id": thread_id}}

    result = await _GRAPH.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )
    messages = result.get("messages", [])
    reply = _final_assistant_text(messages)
    specialists = _specialists_reached(messages)

    memory_saved = False
    if user_id != "anonymous" and reply.strip():
        try:
            await _MEMORY.save_research_result(
                user_id=user_id,
                query=query,
                summary=reply,
                thread_id=thread_id,
            )
            saved_back = await _MEMORY.get_memory(
                user_id=user_id,
                namespace="research_history",
                key=thread_id,
            )
            memory_saved = saved_back is not None
        except Exception:  # noqa: BLE001
            memory_saved = False

    return {
        "reply": reply,
        "specialists_reached": specialists,
        "memory_saved": memory_saved,
        "thread_id": thread_id,
    }
