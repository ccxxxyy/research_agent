"""Evaluation target: run the real supervisor graph and return structured outputs.

The target bypasses FastAPI entirely — it calls ``graph.ainvoke`` directly
so evaluation runs are not affected by auth, rate limiting, or HTTP overhead.

A single ``build_eval_environment`` call initialises the graph + memory once
per evaluation session; ``supervisor_target`` is the per-example callable
that ``langsmith.evaluate`` invokes.
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

from research_agent.api.routes.supervisor import (
    _final_assistant_text,
    _specialists_reached,
)
from research_agent.memory.manager import MemoryManager


_GRAPH = None
_MEMORY: MemoryManager | None = None


async def build_eval_environment() -> None:
    """One-time setup: compile the research supervisor and memory manager.

    Must be called before the first ``supervisor_target`` invocation.
    Uses the same ``_try_build_research_supervisor`` path as the
    production lifespan so the graph topology is identical.
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

    _GRAPH = await _try_build_research_supervisor(
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
    """Run one evaluation example through the supervisor.

    Args:
        inputs: A dict matching the dataset schema —
            ``{"query": str, "user_id": str, ...}``.

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
