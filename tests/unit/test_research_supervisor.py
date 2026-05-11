"""Phase-4.4 unit tests — research supervisor graph compiles & wires correctly.

These tests deliberately avoid:
  * real MCP subprocesses (no ``load_*_server_tools()`` calls),
  * real network / LLM traffic (no ``graph.ainvoke``).

The goal is to guard the *wiring contract* of ``build_research_supervisor``:
the set of specialists, input validation, and prompt adaptation. Heavy
end-to-end coverage lives in:

  * ``tests/unit/test_mcp_fin_data_server.py``
  * ``tests/unit/test_mcp_pdf_report_server.py``
  * ``scripts/demo_financial_research.py`` (manual / nightly)
"""

from __future__ import annotations

import pytest
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import MemorySaver

from research_agent.agents.specialists import (
    SPECIALIST_BUILDERS,
    build_data_expert,
    build_knowledge_expert,
    build_news_expert,
    build_report_expert,
)
from research_agent.config import LLMConfig
from research_agent.graph.research_supervisor import (
    SUPERVISOR_PROMPT_CODER,
    SUPERVISOR_PROMPT_DATA,
    SUPERVISOR_PROMPT_KNOWLEDGE,
    SUPERVISOR_PROMPT_NEWS,
    SUPERVISOR_PROMPT_REPORT,
    _build_supervisor_prompt,
    build_research_supervisor,
)
from research_agent.llm.provider import ModelRouter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_tool(name: str) -> BaseTool:
    """Create a minimal BaseTool suitable for ``create_react_agent``.

    Using ``@tool`` keeps us on the same contract ``create_react_agent``
    expects without dragging in a real MCP subprocess. The function body
    is never invoked in these tests — we only check that the graph
    compiles with the tools attached.
    """

    @tool(name)
    def _t(payload: str) -> str:
        """Fake tool; intentionally does nothing."""
        return payload

    return _t  # type: ignore[return-value]


@pytest.fixture
def fake_data_tools() -> list[BaseTool]:
    return [
        _fake_tool("fin_search_stock_by_name"),
        _fake_tool("fin_get_stock_basic_info"),
        _fake_tool("fin_get_stock_price_history"),
        _fake_tool("fin_get_financial_abstract"),
        _fake_tool("fin_get_financial_indicators"),
    ]


@pytest.fixture
def fake_report_tools() -> list[BaseTool]:
    return [
        _fake_tool("pdf_search_announcements"),
        _fake_tool("pdf_download_pdf"),
        _fake_tool("pdf_extract_pdf_metadata"),
        _fake_tool("pdf_parse_pdf_pages"),
    ]


@pytest.fixture
def fake_coder_tools() -> list[BaseTool]:
    return [_fake_tool("code_execute_python")]


@pytest.fixture
def fake_knowledge_tools() -> list[BaseTool]:
    return [
        _fake_tool("knowledge_ingest_pdf"),
        _fake_tool("knowledge_search"),
        _fake_tool("knowledge_list_collections"),
        _fake_tool("knowledge_delete_collection"),
    ]


@pytest.fixture
def fake_news_tools() -> list[BaseTool]:
    return [
        _fake_tool("news_get_stock_news"),
        _fake_tool("news_get_market_telegraph"),
        _fake_tool("news_get_hot_keywords"),
        _fake_tool("news_get_economic_news"),
        _fake_tool("news_get_xueqiu_discussion_hot_rank"),
    ]


@pytest.fixture
def router() -> ModelRouter:
    # Fake credentials — compile-time only, no network calls happen here.
    cfg = LLMConfig(
        deepseek_api_key="sk-test-not-used-at-compile-time",
        deepseek_api_base="https://example.invalid/v1",
    )
    return ModelRouter(cfg)


# ---------------------------------------------------------------------------
# Specialist builder guard-rails
# ---------------------------------------------------------------------------


class TestSpecialistBuilders:
    def test_registry_contains_new_specialists(self) -> None:
        assert "data_expert" in SPECIALIST_BUILDERS
        assert "report_expert" in SPECIALIST_BUILDERS
        assert "knowledge_expert" in SPECIALIST_BUILDERS
        assert "news_expert" in SPECIALIST_BUILDERS
        assert SPECIALIST_BUILDERS["data_expert"] is build_data_expert
        assert SPECIALIST_BUILDERS["report_expert"] is build_report_expert
        assert SPECIALIST_BUILDERS["knowledge_expert"] is build_knowledge_expert
        assert SPECIALIST_BUILDERS["news_expert"] is build_news_expert

    def test_data_expert_rejects_empty_tools(self, router: ModelRouter) -> None:
        with pytest.raises(ValueError, match="fin_data_server"):
            build_data_expert(router, [])

    def test_report_expert_rejects_empty_tools(self, router: ModelRouter) -> None:
        with pytest.raises(ValueError, match="pdf_report_server"):
            build_report_expert(router, [])

    def test_knowledge_expert_rejects_empty_tools(self, router: ModelRouter) -> None:
        with pytest.raises(ValueError, match="knowledge_server"):
            build_knowledge_expert(router, [])

    def test_news_expert_rejects_empty_tools(self, router: ModelRouter) -> None:
        with pytest.raises(ValueError, match="news_server"):
            build_news_expert(router, [])

    def test_data_expert_compiles_with_tools(
        self, router: ModelRouter, fake_data_tools: list[BaseTool]
    ) -> None:
        agent = build_data_expert(router, fake_data_tools)
        # ``create_react_agent`` returns a compiled graph with ``ainvoke``.
        assert hasattr(agent, "ainvoke")
        # The ``name`` is what ``langgraph_supervisor`` keys off of.
        assert getattr(agent, "name", None) == "data_expert"

    def test_report_expert_compiles_with_tools(
        self, router: ModelRouter, fake_report_tools: list[BaseTool]
    ) -> None:
        agent = build_report_expert(router, fake_report_tools)
        assert hasattr(agent, "ainvoke")
        assert getattr(agent, "name", None) == "report_expert"

    def test_knowledge_expert_compiles_with_tools(
        self, router: ModelRouter, fake_knowledge_tools: list[BaseTool]
    ) -> None:
        agent = build_knowledge_expert(router, fake_knowledge_tools)
        assert hasattr(agent, "ainvoke")
        assert getattr(agent, "name", None) == "knowledge_expert"

    def test_news_expert_compiles_with_tools(
        self, router: ModelRouter, fake_news_tools: list[BaseTool]
    ) -> None:
        agent = build_news_expert(router, fake_news_tools)
        assert hasattr(agent, "ainvoke")
        assert getattr(agent, "name", None) == "news_expert"


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


class TestSupervisorPrompt:
    def test_all_specialists_listed(self) -> None:
        prompt = _build_supervisor_prompt(
            has_data=True,
            has_report=True,
            has_coder=True,
            has_knowledge=True,
            has_news=True,
            has_sentiment=True,
        )
        assert "data_expert" in prompt
        assert "report_expert" in prompt
        assert "coder_expert" in prompt
        assert "knowledge_expert" in prompt
        assert "news_expert" in prompt
        assert "sentiment_expert" in prompt
        # Section texts actually appended.
        assert SUPERVISOR_PROMPT_DATA in prompt
        assert SUPERVISOR_PROMPT_REPORT in prompt
        assert SUPERVISOR_PROMPT_CODER in prompt
        assert SUPERVISOR_PROMPT_KNOWLEDGE in prompt
        assert SUPERVISOR_PROMPT_NEWS in prompt

    def test_missing_specialists_are_not_mentioned(self) -> None:
        """Mentioning a specialist that doesn't exist in the team would
        cause the supervisor to emit ``transfer_to_<missing>`` tool
        calls that fail at runtime. This is the property we guard."""
        prompt = _build_supervisor_prompt(
            has_data=True,
            has_report=False,
            has_coder=False,
            has_knowledge=False,
            has_news=False,
            has_sentiment=False,
        )
        assert "data_expert" in prompt
        assert "report_expert" not in prompt
        assert "coder_expert" not in prompt
        assert "knowledge_expert" not in prompt
        assert "news_expert" not in prompt
        assert "sentiment_expert" not in prompt

    def test_knowledge_only_prompt_omits_other_experts(self) -> None:
        """Symmetric guard for the knowledge_expert-only team."""
        prompt = _build_supervisor_prompt(
            has_data=False,
            has_report=False,
            has_coder=False,
            has_knowledge=True,
            has_news=False,
            has_sentiment=False,
        )
        assert "knowledge_expert" in prompt
        assert SUPERVISOR_PROMPT_KNOWLEDGE in prompt
        assert "data_expert" not in prompt
        assert "report_expert" not in prompt
        assert "coder_expert" not in prompt
        assert "news_expert" not in prompt
        assert "sentiment_expert" not in prompt

    def test_news_only_prompt_omits_other_experts(self) -> None:
        """Symmetric guard for the news_expert-only team."""
        prompt = _build_supervisor_prompt(
            has_data=False,
            has_report=False,
            has_coder=False,
            has_knowledge=False,
            has_news=True,
            has_sentiment=False,
        )
        assert "news_expert" in prompt
        assert SUPERVISOR_PROMPT_NEWS in prompt
        assert "data_expert" not in prompt
        assert "report_expert" not in prompt
        assert "coder_expert" not in prompt
        assert "knowledge_expert" not in prompt
        assert "sentiment_expert" not in prompt

    def test_rules_always_included(self) -> None:
        """The routing rules ("hand off one subtask at a time", "never
        invent numbers", etc.) are non-negotiable — they must appear
        regardless of team composition."""
        # Each tuple is one (has_data, has_report, has_coder,
        # has_knowledge, has_news, has_sentiment) combination — six
        # single-specialist setups plus one full-team setup.
        for flags in [
            (True, False, False, False, False, False),
            (False, True, False, False, False, False),
            (False, False, True, False, False, False),
            (False, False, False, True, False, False),
            (False, False, False, False, True, False),
            (False, False, False, False, False, True),
            (True, True, True, True, True, True),
        ]:
            prompt = _build_supervisor_prompt(
                has_data=flags[0],
                has_report=flags[1],
                has_coder=flags[2],
                has_knowledge=flags[3],
                has_news=flags[4],
                has_sentiment=flags[5],
            )
            assert "hand off" in prompt.lower() or "hand-off" in prompt.lower()
            assert "Never invent" in prompt

    def test_anti_hallucination_rules_present(self) -> None:
        prompt = _build_supervisor_prompt(
            has_data=True,
            has_report=True,
            has_coder=True,
            has_knowledge=True,
            has_news=True,
            has_sentiment=True,
        )
        assert "NEVER claim" in prompt
        assert "unavailable" in prompt.lower()
        assert "NEVER substitute" in prompt
        assert "NEVER perform arithmetic" in prompt

    def test_anti_hallucination_present_single_specialist(self) -> None:
        prompt = _build_supervisor_prompt(
            has_data=True,
            has_report=False,
            has_coder=False,
            has_knowledge=False,
            has_news=False,
            has_sentiment=False,
        )
        assert "NEVER claim" in prompt
        assert "NEVER substitute" in prompt

    def test_sub_question_decomposition_guidance(self) -> None:
        prompt = _build_supervisor_prompt(
            has_data=True,
            has_report=True,
            has_coder=True,
            has_knowledge=True,
            has_news=True,
            has_sentiment=True,
        )
        lower = prompt.lower()
        assert "numbered step" in lower or "numbered steps" in lower

    def test_self_check_before_final_answer(self) -> None:
        prompt = _build_supervisor_prompt(
            has_data=True,
            has_report=True,
            has_coder=True,
            has_knowledge=True,
            has_news=True,
            has_sentiment=True,
        )
        lower = prompt.lower()
        assert "self-check" in lower or "re-read" in lower


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


class TestBuildResearchSupervisor:
    def test_full_team_compiles(
        self,
        router: ModelRouter,
        fake_data_tools: list[BaseTool],
        fake_report_tools: list[BaseTool],
        fake_coder_tools: list[BaseTool],
        fake_knowledge_tools: list[BaseTool],
        fake_news_tools: list[BaseTool],
    ) -> None:
        graph = build_research_supervisor(
            model_router=router,
            data_tools=fake_data_tools,
            report_tools=fake_report_tools,
            coder_tools=fake_coder_tools,
            knowledge_tools=fake_knowledge_tools,
            news_tools=fake_news_tools,
            checkpointer=MemorySaver(),
        )
        assert hasattr(graph, "ainvoke")
        assert hasattr(graph, "get_graph")

        # All five specialists should appear as subgraph nodes.
        # ``get_graph()`` returns a nx-like object; its ``nodes`` is a dict.
        node_names = set(graph.get_graph().nodes.keys())
        assert "data_expert" in node_names
        assert "report_expert" in node_names
        assert "coder_expert" in node_names
        assert "knowledge_expert" in node_names
        assert "news_expert" in node_names

    def test_data_only_compiles(
        self, router: ModelRouter, fake_data_tools: list[BaseTool]
    ) -> None:
        graph = build_research_supervisor(
            model_router=router, data_tools=fake_data_tools
        )
        node_names = set(graph.get_graph().nodes.keys())
        assert "data_expert" in node_names
        assert "report_expert" not in node_names
        assert "coder_expert" not in node_names
        assert "knowledge_expert" not in node_names
        assert "news_expert" not in node_names

    def test_report_only_compiles(
        self, router: ModelRouter, fake_report_tools: list[BaseTool]
    ) -> None:
        graph = build_research_supervisor(
            model_router=router, report_tools=fake_report_tools
        )
        node_names = set(graph.get_graph().nodes.keys())
        assert "report_expert" in node_names
        assert "data_expert" not in node_names
        assert "knowledge_expert" not in node_names
        assert "news_expert" not in node_names

    def test_knowledge_only_compiles(
        self, router: ModelRouter, fake_knowledge_tools: list[BaseTool]
    ) -> None:
        """A knowledge-only team is the smoke configuration for the
        Phase-4.6 RAG closure — guard that it compiles standalone."""
        graph = build_research_supervisor(
            model_router=router, knowledge_tools=fake_knowledge_tools
        )
        node_names = set(graph.get_graph().nodes.keys())
        assert "knowledge_expert" in node_names
        assert "data_expert" not in node_names
        assert "report_expert" not in node_names
        assert "coder_expert" not in node_names
        assert "news_expert" not in node_names

    def test_news_only_compiles(
        self, router: ModelRouter, fake_news_tools: list[BaseTool]
    ) -> None:
        """A news-only team is the smoke configuration for the news
        plane — guard that it compiles standalone too."""
        graph = build_research_supervisor(
            model_router=router, news_tools=fake_news_tools
        )
        node_names = set(graph.get_graph().nodes.keys())
        assert "news_expert" in node_names
        assert "data_expert" not in node_names
        assert "report_expert" not in node_names
        assert "coder_expert" not in node_names
        assert "knowledge_expert" not in node_names

    def test_empty_inputs_raise(self, router: ModelRouter) -> None:
        with pytest.raises(ValueError, match="at least one specialist"):
            build_research_supervisor(model_router=router)

    def test_all_empty_lists_raise(self, router: ModelRouter) -> None:
        """Passing empty lists is semantically identical to passing
        nothing — the builder should reject both uniformly."""
        with pytest.raises(ValueError, match="at least one specialist"):
            build_research_supervisor(
                model_router=router,
                data_tools=[],
                report_tools=[],
                coder_tools=[],
                knowledge_tools=[],
                news_tools=[],
            )

    def test_enable_reflection_wraps_graph(
        self, router: ModelRouter, fake_data_tools: list[BaseTool]
    ) -> None:
        """``enable_reflection=True`` replaces the supervisor with a
        parent graph that exposes ``supervisor`` and ``reflection``
        as top-level nodes. This is the structural contract: callers
        keep the same ``ainvoke`` API but the DAG now reflects the
        two-stage pipeline."""
        graph = build_research_supervisor(
            model_router=router,
            data_tools=fake_data_tools,
            enable_reflection=True,
        )
        node_names = set(graph.get_graph().nodes.keys())
        assert "supervisor" in node_names
        assert "reflection" in node_names
        # When reflection is on, the inner supervisor is encapsulated;
        # specialist nodes should NOT leak into the parent topology
        # (they live one level deeper, inside the supervisor node).
        assert "data_expert" not in node_names

    def test_reflection_off_keeps_legacy_topology(
        self, router: ModelRouter, fake_data_tools: list[BaseTool]
    ) -> None:
        """The default (``enable_reflection=False``) must NOT introduce
        the reflection wrapper. The legacy supervisor already names
        its router node ``supervisor`` (that's a ``langgraph_supervisor``
        contract), so the marker that distinguishes "wrapped" from
        "not wrapped" is the presence of the ``reflection`` node, plus
        the specialists being top-level (not encapsulated)."""
        graph = build_research_supervisor(
            model_router=router, data_tools=fake_data_tools
        )
        node_names = set(graph.get_graph().nodes.keys())
        # Specialists are at the top level in the unwrapped graph.
        assert "data_expert" in node_names
        # The reflection node is what the wrapper adds — absence
        # confirms we're on the legacy topology.
        assert "reflection" not in node_names
