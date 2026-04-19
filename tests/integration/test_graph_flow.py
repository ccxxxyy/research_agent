"""Integration tests for the full LangGraph research pipeline.

These tests require LLM API keys and validate the end-to-end graph execution.
Run with: pytest tests/integration/ -m integration
"""

import pytest
from langchain_core.runnables import RunnableConfig

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestGraphFlow:
    @pytest.mark.asyncio
    async def test_graph_compiles(self, model_router, checkpointer, memory_store):
        """Verify the graph compiles without errors."""
        from research_agent.graph.supervisor import build_research_graph

        graph = build_research_graph(
            model_router=model_router,
            checkpointer=checkpointer,
            memory_store=memory_store,
        )
        assert graph is not None

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires LLM API keys")
    async def test_full_research_flow(self, model_router, checkpointer, memory_store):
        """Full end-to-end test of the research pipeline."""
        from research_agent.graph.supervisor import build_research_graph

        graph = build_research_graph(
            model_router=model_router,
            checkpointer=checkpointer,
            memory_store=memory_store,
        )

        config: RunnableConfig = {"configurable": {"thread_id": "test-thread-001"}}
        result = await graph.ainvoke(
            {"query": "What are the latest trends in AI Agent development?"},
            config=config,
        )

        assert result.get("phase") == "completed"
        assert result.get("final_report")
        assert result.get("quality_score", 0) > 0
