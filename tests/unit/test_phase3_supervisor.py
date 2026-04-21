"""Phase-3: minimal supervisor graph compiles without live LLM calls."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver

from research_agent.config import LLMConfig
from research_agent.graph.minimal_supervisor import build_minimal_supervisor
from research_agent.llm.provider import ModelRouter


def test_build_minimal_supervisor_returns_compiled_graph() -> None:
    cfg = LLMConfig(
        deepseek_api_key="sk-test-not-used-at-compile-time",
        deepseek_api_base="https://example.invalid/v1",
    )
    router = ModelRouter(cfg)
    graph = build_minimal_supervisor(model_router=router, checkpointer=MemorySaver())

    assert hasattr(graph, "ainvoke")
    assert hasattr(graph, "get_graph")
