"""单元测试 — 验证 research supervisor 图的编译与连线正确性。

这些测试刻意避免了：
  * 真实 MCP 子进程（不调用 ``load_*_server_tools()``），
  * 真实网络 / LLM 流量（不调用 ``graph.ainvoke``）。

目标是守护 ``build_research_supervisor`` 的*连线契约*：专家集合、输入校验和提示词适配。重度端到端覆盖位于：

  * ``tests/unit/test_mcp_fin_data_server.py``
  * ``tests/unit/test_mcp_pdf_report_server.py``
  * ``scripts/demo_financial_research.py``（手动 / 夜间构建）
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
    build_us_data_expert,
    build_us_filing_expert,
    build_us_news_expert,
    build_us_sentiment_expert,
)
from research_agent.config import LLMConfig
from research_agent.graph.research_supervisor import (
    SUPERVISOR_PROMPT_CODER,
    SUPERVISOR_PROMPT_DATA,
    SUPERVISOR_PROMPT_FUND,
    SUPERVISOR_PROMPT_KNOWLEDGE,
    SUPERVISOR_PROMPT_NEWS,
    SUPERVISOR_PROMPT_REPORT,
    SUPERVISOR_PROMPT_US_DATA,
    SUPERVISOR_PROMPT_US_FILING,
    SUPERVISOR_PROMPT_US_NEWS,
    SUPERVISOR_PROMPT_US_SENTIMENT,
    _build_supervisor_prompt,
    build_research_supervisor,
)
from research_agent.llm.provider import ModelRouter

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _fake_tool(name: str) -> BaseTool:
    """创建一个适用于 ``create_agent`` 的最小 BaseTool。

    使用 ``@tool`` 装饰器保持与 ``create_agent`` 期望的相同契约，
    而无需引入真实的 MCP 子进程。函数体在这些测试中不会被调用 —仅检查图能否在附加这些工具后成功编译。
    """

    @tool(name)
    def _t(payload: str) -> str:
        """假工具；故意不执行任何操作。"""
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
def fake_us_data_tools() -> list[BaseTool]:
    return [
        _fake_tool("us_get_market_status"),
        _fake_tool("us_search_ticker"),
        _fake_tool("us_get_quote"),
        _fake_tool("us_get_price_history"),
        _fake_tool("us_get_basic_info"),
        _fake_tool("us_get_index_quotes"),
        _fake_tool("us_get_etf_overview"),
        _fake_tool("us_get_etf_holdings"),
        _fake_tool("us_get_etf_sector_weights"),
    ]


@pytest.fixture
def fake_us_filing_tools() -> list[BaseTool]:
    return [
        _fake_tool("us_filing_resolve_cik"),
        _fake_tool("us_filing_search_filings"),
        _fake_tool("us_filing_download_filing"),
        _fake_tool("us_filing_extract_filing_metadata"),
        _fake_tool("us_filing_parse_filing_text"),
    ]


@pytest.fixture
def fake_us_news_tools() -> list[BaseTool]:
    return [
        _fake_tool("us_news_get_ticker_news"),
        _fake_tool("us_news_get_market_news"),
        _fake_tool("us_news_get_etf_news"),
        _fake_tool("us_news_get_recent_8k_headlines"),
    ]


@pytest.fixture
def fake_us_sentiment_tools() -> list[BaseTool]:
    return [
        _fake_tool("us_sentiment_analyze_text_sentiment"),
        _fake_tool("us_sentiment_get_ticker_sentiment_report"),
    ]


@pytest.fixture
def router() -> ModelRouter:
    # 假凭据 — 仅用于编译阶段，此处不会发生网络调用。
    cfg = LLMConfig(
        deepseek_api_key="sk-test-not-used-at-compile-time",
        deepseek_api_base="https://example.invalid/v1",
    )
    return ModelRouter(cfg)


# ---------------------------------------------------------------------------
# 专家构建器的护栏测试
# ---------------------------------------------------------------------------


class TestSpecialistBuilders:
    def test_registry_contains_new_specialists(self) -> None:
        assert "data_expert" in SPECIALIST_BUILDERS
        assert "us_data_expert" in SPECIALIST_BUILDERS
        assert "us_filing_expert" in SPECIALIST_BUILDERS
        assert "us_news_expert" in SPECIALIST_BUILDERS
        assert "us_sentiment_expert" in SPECIALIST_BUILDERS
        assert "report_expert" in SPECIALIST_BUILDERS
        assert "knowledge_expert" in SPECIALIST_BUILDERS
        assert "news_expert" in SPECIALIST_BUILDERS
        assert SPECIALIST_BUILDERS["data_expert"] is build_data_expert
        assert SPECIALIST_BUILDERS["us_data_expert"] is build_us_data_expert
        assert SPECIALIST_BUILDERS["us_filing_expert"] is build_us_filing_expert
        assert SPECIALIST_BUILDERS["us_news_expert"] is build_us_news_expert
        assert SPECIALIST_BUILDERS["us_sentiment_expert"] is build_us_sentiment_expert
        assert SPECIALIST_BUILDERS["report_expert"] is build_report_expert
        assert SPECIALIST_BUILDERS["knowledge_expert"] is build_knowledge_expert
        assert SPECIALIST_BUILDERS["news_expert"] is build_news_expert

    def test_data_expert_rejects_empty_tools(self, router: ModelRouter) -> None:
        with pytest.raises(ValueError, match="fin_data_server"):
            build_data_expert(router, [])

    def test_us_data_expert_rejects_empty_tools(self, router: ModelRouter) -> None:
        with pytest.raises(ValueError, match="us_data_server"):
            build_us_data_expert(router, [])

    def test_us_filing_expert_rejects_empty_tools(self, router: ModelRouter) -> None:
        with pytest.raises(ValueError, match="us_filing_server"):
            build_us_filing_expert(router, [])

    def test_us_news_expert_rejects_empty_tools(self, router: ModelRouter) -> None:
        with pytest.raises(ValueError, match="us_news_server"):
            build_us_news_expert(router, [])

    def test_us_sentiment_expert_rejects_empty_tools(self, router: ModelRouter) -> None:
        with pytest.raises(ValueError, match="us_sentiment_server"):
            build_us_sentiment_expert(router, [])

    def test_report_expert_rejects_empty_tools(self, router: ModelRouter) -> None:
        with pytest.raises(ValueError, match="pdf_report_server"):
            build_report_expert(router, [])

    def test_knowledge_expert_rejects_empty_tools(self, router: ModelRouter) -> None:
        with pytest.raises(ValueError, match="load_knowledge_tools_inproc"):
            build_knowledge_expert(router, [])

    def test_news_expert_rejects_empty_tools(self, router: ModelRouter) -> None:
        with pytest.raises(ValueError, match="news_server"):
            build_news_expert(router, [])

    def test_data_expert_compiles_with_tools(
        self, router: ModelRouter, fake_data_tools: list[BaseTool]
    ) -> None:
        agent = build_data_expert(router, fake_data_tools)
        # ``create_agent`` 返回一个包含 ``ainvoke`` 的已编译图。
        assert hasattr(agent, "ainvoke")
        # ``name`` 是 ``langgraph_supervisor`` 用来匹配的键。
        assert getattr(agent, "name", None) == "data_expert"

    def test_us_data_expert_compiles_with_tools(
        self, router: ModelRouter, fake_us_data_tools: list[BaseTool]
    ) -> None:
        agent = build_us_data_expert(router, fake_us_data_tools)
        assert hasattr(agent, "ainvoke")
        assert getattr(agent, "name", None) == "us_data_expert"

    def test_us_filing_expert_compiles_with_tools(
        self, router: ModelRouter, fake_us_filing_tools: list[BaseTool]
    ) -> None:
        agent = build_us_filing_expert(router, fake_us_filing_tools)
        assert hasattr(agent, "ainvoke")
        assert getattr(agent, "name", None) == "us_filing_expert"

    def test_us_news_expert_compiles_with_tools(
        self, router: ModelRouter, fake_us_news_tools: list[BaseTool]
    ) -> None:
        agent = build_us_news_expert(router, fake_us_news_tools)
        assert hasattr(agent, "ainvoke")
        assert getattr(agent, "name", None) == "us_news_expert"

    def test_us_sentiment_expert_compiles_with_tools(
        self, router: ModelRouter, fake_us_sentiment_tools: list[BaseTool]
    ) -> None:
        agent = build_us_sentiment_expert(router, fake_us_sentiment_tools)
        assert hasattr(agent, "ainvoke")
        assert getattr(agent, "name", None) == "us_sentiment_expert"

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
# 提示词组装
# ---------------------------------------------------------------------------


class TestSupervisorPrompt:
    def test_all_specialists_listed(self) -> None:
        prompt = _build_supervisor_prompt(
            has_data=True,
            has_us_data=True,
            has_us_filing=True,
            has_us_news=True,
            has_us_sentiment=True,
            has_report=True,
            has_coder=True,
            has_knowledge=True,
            has_news=True,
            has_sentiment=True,
            has_fund=True,
        )
        # 使用 "  - <name>" 避免 us_*_expert 子串命中裸名。
        assert "  - data_expert" in prompt
        assert "us_data_expert" in prompt
        assert "us_filing_expert" in prompt
        assert "us_news_expert" in prompt
        assert "us_sentiment_expert" in prompt
        assert "report_expert" in prompt
        assert "coder_expert" in prompt
        assert "knowledge_expert" in prompt
        assert "  - news_expert" in prompt
        assert "  - sentiment_expert" in prompt
        assert "fund_expert" in prompt
        # 各节文本确实被追加。
        assert SUPERVISOR_PROMPT_DATA in prompt
        assert SUPERVISOR_PROMPT_US_DATA in prompt
        assert SUPERVISOR_PROMPT_US_FILING in prompt
        assert SUPERVISOR_PROMPT_US_NEWS in prompt
        assert SUPERVISOR_PROMPT_US_SENTIMENT in prompt
        assert SUPERVISOR_PROMPT_REPORT in prompt
        assert SUPERVISOR_PROMPT_CODER in prompt
        assert SUPERVISOR_PROMPT_KNOWLEDGE in prompt
        assert SUPERVISOR_PROMPT_NEWS in prompt
        assert SUPERVISOR_PROMPT_FUND in prompt

    def test_missing_specialists_are_not_mentioned(self) -> None:
        """提及团队中不存在的专家会导致 supervisor 发出在运行时失败的
        ``transfer_to_<missing>`` 工具调用。这就是要守护的属性。"""
        prompt = _build_supervisor_prompt(
            has_data=True,
            has_us_data=False,
            has_us_filing=False,
            has_us_news=False,
            has_us_sentiment=False,
            has_report=False,
            has_coder=False,
            has_knowledge=False,
            has_news=False,
            has_sentiment=False,
            has_fund=False,
        )
        assert "  - data_expert" in prompt
        assert "us_data_expert" not in prompt
        assert "us_filing_expert" not in prompt
        assert "us_news_expert" not in prompt
        assert "us_sentiment_expert" not in prompt
        assert "report_expert" not in prompt
        assert "coder_expert" not in prompt
        assert "knowledge_expert" not in prompt
        assert "  - news_expert" not in prompt
        assert "  - sentiment_expert" not in prompt
        assert "fund_expert" not in prompt

    def test_us_only_prompt_omits_cn_data_expert(self) -> None:
        prompt = _build_supervisor_prompt(
            has_data=False,
            has_us_data=True,
            has_us_filing=False,
            has_report=False,
            has_coder=False,
            has_knowledge=False,
            has_news=False,
            has_sentiment=False,
            has_fund=False,
        )
        assert "us_data_expert" in prompt
        assert SUPERVISOR_PROMPT_US_DATA in prompt
        assert "  - data_expert" not in prompt
        assert SUPERVISOR_PROMPT_DATA not in prompt

    def test_us_filing_only_prompt_omits_cn_report_expert(self) -> None:
        prompt = _build_supervisor_prompt(
            has_data=False,
            has_us_data=False,
            has_us_filing=True,
            has_report=False,
            has_coder=False,
            has_knowledge=False,
            has_news=False,
            has_sentiment=False,
            has_fund=False,
        )
        assert "us_filing_expert" in prompt
        assert SUPERVISOR_PROMPT_US_FILING in prompt
        assert "report_expert" not in prompt
        assert SUPERVISOR_PROMPT_REPORT not in prompt

    def test_knowledge_only_prompt_omits_other_experts(self) -> None:
        """针对仅含 knowledge_expert 团队的对称守护测试。"""
        prompt = _build_supervisor_prompt(
            has_data=False,
            has_us_data=False,
            has_us_filing=False,
            has_report=False,
            has_coder=False,
            has_knowledge=True,
            has_news=False,
            has_sentiment=False,
            has_fund=False,
        )
        assert "knowledge_expert" in prompt
        assert SUPERVISOR_PROMPT_KNOWLEDGE in prompt
        assert "  - data_expert" not in prompt
        assert "us_data_expert" not in prompt
        assert "us_filing_expert" not in prompt
        assert "report_expert" not in prompt
        assert "coder_expert" not in prompt
        assert "news_expert" not in prompt
        assert "sentiment_expert" not in prompt
        assert "fund_expert" not in prompt

    def test_news_only_prompt_omits_other_experts(self) -> None:
        """针对仅含 news_expert 团队的对称守护测试。"""
        prompt = _build_supervisor_prompt(
            has_data=False,
            has_us_data=False,
            has_us_filing=False,
            has_report=False,
            has_coder=False,
            has_knowledge=False,
            has_news=True,
            has_sentiment=False,
            has_fund=False,
        )
        assert "  - news_expert" in prompt
        assert SUPERVISOR_PROMPT_NEWS in prompt
        assert "  - data_expert" not in prompt
        assert "us_data_expert" not in prompt
        assert "us_filing_expert" not in prompt
        assert "us_news_expert" not in prompt
        assert "us_sentiment_expert" not in prompt
        assert "report_expert" not in prompt
        assert "coder_expert" not in prompt
        assert "knowledge_expert" not in prompt
        assert "  - sentiment_expert" not in prompt
        assert "fund_expert" not in prompt

    def test_us_news_only_prompt_omits_cn_news_expert(self) -> None:
        prompt = _build_supervisor_prompt(
            has_data=False,
            has_us_data=False,
            has_us_filing=False,
            has_us_news=True,
            has_us_sentiment=False,
            has_report=False,
            has_coder=False,
            has_knowledge=False,
            has_news=False,
            has_sentiment=False,
            has_fund=False,
        )
        assert "us_news_expert" in prompt
        assert SUPERVISOR_PROMPT_US_NEWS in prompt
        assert "  - news_expert" not in prompt
        assert SUPERVISOR_PROMPT_NEWS not in prompt

    def test_us_sentiment_only_prompt_omits_cn_sentiment_expert(self) -> None:
        prompt = _build_supervisor_prompt(
            has_data=False,
            has_us_data=False,
            has_us_filing=False,
            has_us_news=False,
            has_us_sentiment=True,
            has_report=False,
            has_coder=False,
            has_knowledge=False,
            has_news=False,
            has_sentiment=False,
            has_fund=False,
        )
        assert "us_sentiment_expert" in prompt
        assert SUPERVISOR_PROMPT_US_SENTIMENT in prompt
        assert "  - sentiment_expert" not in prompt

    def test_rules_always_included(self) -> None:
        """路由规则（"每次只移交一个子任务"、"不可捏造数字"等）是不可协商的 —
        无论团队组成如何，它们都必须出现。"""
        # 元组: (data, us_data, us_filing, us_news, us_sentiment, report, coder, knowledge, news, sentiment, fund)
        for flags in [
            (True, False, False, False, False, False, False, False, False, False, False),
            (False, True, False, False, False, False, False, False, False, False, False),
            (False, False, True, False, False, False, False, False, False, False, False),
            (False, False, False, True, False, False, False, False, False, False, False),
            (False, False, False, False, True, False, False, False, False, False, False),
            (False, False, False, False, False, True, False, False, False, False, False),
            (False, False, False, False, False, False, True, False, False, False, False),
            (False, False, False, False, False, False, False, True, False, False, False),
            (False, False, False, False, False, False, False, False, True, False, False),
            (False, False, False, False, False, False, False, False, False, True, False),
            (False, False, False, False, False, False, False, False, False, False, True),
            (True, True, True, True, True, True, True, True, True, True, True),
        ]:
            prompt = _build_supervisor_prompt(
                has_data=flags[0],
                has_us_data=flags[1],
                has_us_filing=flags[2],
                has_us_news=flags[3],
                has_us_sentiment=flags[4],
                has_report=flags[5],
                has_coder=flags[6],
                has_knowledge=flags[7],
                has_news=flags[8],
                has_sentiment=flags[9],
                has_fund=flags[10],
            )
            assert "移交" in prompt
            assert "不要编造" in prompt

    def test_anti_hallucination_rules_present(self) -> None:
        prompt = _build_supervisor_prompt(
            has_data=True,
            has_report=True,
            has_coder=True,
            has_knowledge=True,
            has_news=True,
            has_sentiment=True,
        )
        assert "绝不声称" in prompt
        assert "不可用" in prompt
        assert "绝不用你自己的知识替代" in prompt
        assert "绝不自己做算术" in prompt

    def test_anti_hallucination_present_single_specialist(self) -> None:
        prompt = _build_supervisor_prompt(
            has_data=True,
            has_report=False,
            has_coder=False,
            has_knowledge=False,
            has_news=False,
            has_sentiment=False,
        )
        assert "绝不声称" in prompt
        assert "绝不用你自己的知识替代" in prompt

    def test_sub_question_decomposition_guidance(self) -> None:
        prompt = _build_supervisor_prompt(
            has_data=True,
            has_report=True,
            has_coder=True,
            has_knowledge=True,
            has_news=True,
            has_sentiment=True,
        )
        assert "编号步骤" in prompt

    def test_self_check_before_final_answer(self) -> None:
        prompt = _build_supervisor_prompt(
            has_data=True,
            has_report=True,
            has_coder=True,
            has_knowledge=True,
            has_news=True,
            has_sentiment=True,
        )
        assert "自检" in prompt or "重新阅读" in prompt


# ---------------------------------------------------------------------------
# 图构建器
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

        # 所有五个专家都应作为子图节点出现。
        # ``get_graph()`` 返回一个类似 nx 的对象；其 ``nodes`` 是一个字典。
        node_names = set(graph.get_graph().nodes.keys())
        assert "data_expert" in node_names
        assert "report_expert" in node_names
        assert "coder_expert" in node_names
        assert "knowledge_expert" in node_names
        assert "news_expert" in node_names

    def test_data_only_compiles(self, router: ModelRouter, fake_data_tools: list[BaseTool]) -> None:
        graph = build_research_supervisor(model_router=router, data_tools=fake_data_tools)
        node_names = set(graph.get_graph().nodes.keys())
        assert "data_expert" in node_names
        assert "report_expert" not in node_names
        assert "coder_expert" not in node_names
        assert "knowledge_expert" not in node_names
        assert "news_expert" not in node_names

    def test_report_only_compiles(
        self, router: ModelRouter, fake_report_tools: list[BaseTool]
    ) -> None:
        graph = build_research_supervisor(model_router=router, report_tools=fake_report_tools)
        node_names = set(graph.get_graph().nodes.keys())
        assert "report_expert" in node_names
        assert "data_expert" not in node_names
        assert "knowledge_expert" not in node_names
        assert "news_expert" not in node_names

    def test_knowledge_only_compiles(
        self, router: ModelRouter, fake_knowledge_tools: list[BaseTool]
    ) -> None:
        """仅含 knowledge 的团队是 Phase-4.6 RAG 闭环的冒烟配置 —守护其能独立编译。"""
        graph = build_research_supervisor(model_router=router, knowledge_tools=fake_knowledge_tools)
        node_names = set(graph.get_graph().nodes.keys())
        assert "knowledge_expert" in node_names
        assert "data_expert" not in node_names
        assert "report_expert" not in node_names
        assert "coder_expert" not in node_names
        assert "news_expert" not in node_names

    def test_news_only_compiles(self, router: ModelRouter, fake_news_tools: list[BaseTool]) -> None:
        """仅含 news 的团队是新闻平面的冒烟配置 — 同样守护其能独立编译。"""
        graph = build_research_supervisor(model_router=router, news_tools=fake_news_tools)
        node_names = set(graph.get_graph().nodes.keys())
        assert "news_expert" in node_names
        assert "data_expert" not in node_names
        assert "report_expert" not in node_names
        assert "coder_expert" not in node_names
        assert "knowledge_expert" not in node_names

    def test_us_only_compiles(
        self, router: ModelRouter, fake_us_data_tools: list[BaseTool]
    ) -> None:
        graph = build_research_supervisor(model_router=router, us_data_tools=fake_us_data_tools)
        node_names = set(graph.get_graph().nodes.keys())
        assert "us_data_expert" in node_names
        assert "data_expert" not in node_names
        assert "news_expert" not in node_names

    def test_us_filing_only_compiles(
        self, router: ModelRouter, fake_us_filing_tools: list[BaseTool]
    ) -> None:
        graph = build_research_supervisor(model_router=router, us_filing_tools=fake_us_filing_tools)
        node_names = set(graph.get_graph().nodes.keys())
        assert "us_filing_expert" in node_names
        assert "report_expert" not in node_names
        assert "data_expert" not in node_names

    def test_us_news_only_compiles(
        self, router: ModelRouter, fake_us_news_tools: list[BaseTool]
    ) -> None:
        graph = build_research_supervisor(model_router=router, us_news_tools=fake_us_news_tools)
        node_names = set(graph.get_graph().nodes.keys())
        assert "us_news_expert" in node_names
        assert "news_expert" not in node_names

    def test_us_sentiment_only_compiles(
        self, router: ModelRouter, fake_us_sentiment_tools: list[BaseTool]
    ) -> None:
        graph = build_research_supervisor(
            model_router=router, us_sentiment_tools=fake_us_sentiment_tools
        )
        node_names = set(graph.get_graph().nodes.keys())
        assert "us_sentiment_expert" in node_names
        assert "sentiment_expert" not in node_names

    def test_empty_inputs_raise(self, router: ModelRouter) -> None:
        with pytest.raises(ValueError, match="至少需要一个专家的工具列表非空"):
            build_research_supervisor(model_router=router)

    def test_all_empty_lists_raise(self, router: ModelRouter) -> None:
        """传递空列表在语义上等同于不传递任何内容 — 构建器应统一拒绝两者。"""
        with pytest.raises(ValueError, match="至少需要一个专家的工具列表非空"):
            build_research_supervisor(
                model_router=router,
                data_tools=[],
                us_data_tools=[],
                us_filing_tools=[],
                us_news_tools=[],
                us_sentiment_tools=[],
                report_tools=[],
                coder_tools=[],
                knowledge_tools=[],
                news_tools=[],
            )

    def test_enable_reflection_wraps_graph(
        self, router: ModelRouter, fake_data_tools: list[BaseTool]
    ) -> None:
        """``enable_reflection=True`` 将 supervisor 替换为一个父级图，
        将 ``supervisor`` 和 ``reflection`` 暴露为顶层节点。这是结构性契约：调用方保持相同的 ``ainvoke`` API，但 DAG 现在反映了两阶段流水线。"""
        graph = build_research_supervisor(
            model_router=router,
            data_tools=fake_data_tools,
            enable_reflection=True,
        )
        node_names = set(graph.get_graph().nodes.keys())
        assert "supervisor" in node_names
        assert "reflection" in node_names
        # 当反射开启时，内部 supervisor 被封装；
        # 专家节点不应泄漏到父级拓扑中（它们位于更深一层，在 supervisor 节点内部）。
        assert "data_expert" not in node_names

    def test_reflection_off_keeps_legacy_topology(
        self, router: ModelRouter, fake_data_tools: list[BaseTool]
    ) -> None:
        """默认设置（``enable_reflection=False``）不得引入反射包装器。 旧版 supervisor 已将其路由节点命名为 ``supervisor``（这是``langgraph_supervisor`` 的契约），
        因此区分"已包装"与"未包装"的标志是 ``reflection`` 节点的存在，以及专家节点位于顶层（未被封装）。"""
        graph = build_research_supervisor(model_router=router, data_tools=fake_data_tools)
        node_names = set(graph.get_graph().nodes.keys())
        # 在未包装的图中，专家节点位于顶层。
        assert "data_expert" in node_names
        # 反射节点是包装器添加的 — 其缺失确认处于旧版拓扑上。
        assert "reflection" not in node_names
