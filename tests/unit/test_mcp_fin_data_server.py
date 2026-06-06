"""MCP ``fin_data_server`` 往返测试。

为什么此测试文件如此组织
------------------------------------------
``fin_data_server`` 是所有专家的数据平面。
每次 MCP 工具调用通常会启动一个全新子进程，而该子进程在能够响应之前需要支付不小的启动开销：

- ``pandas`` 顶层导入：约 1.5 秒
- ``akshare`` 首次工具调用时的懒加载导入：约 3-5 秒
- 一次性 ``stock_info_a_code_name()`` 股票花名册获取（用于 ``search_stock_by_name``）：约 6 秒

如果每个测试方法都创建自己的客户端和子进程，仅启动开销就会轻松超过 60 秒
。因此每个测试打开*个``client.session(...)``上下文，然后在该单一子进程中执行所有需要覆盖的工具契约。
这将总开销固定在每个测试函数大约一次 akshare 预热（约 10 秒），整个文件即使在慢速链路上也能在 40 秒内完成。

所有测试访问实时 HTTP 端点（akshare 镜像的东财/雪球/新浪），因此标记为 ``network``；离线 CI 应使用 ``pytest -m 'not network'`` 运行。
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient

from research_agent.mcp_servers.client_factory import (
    FIN_DATA_SERVER_MODULE,
    extract_text_content,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langchain_core.tools import BaseTool

pytestmark = pytest.mark.network

# 工具名称由 ``load_mcp_tools(session)`` 返回——即原始名称，
# 不带 ``MultiServerMCPClient.get_tools()`` 通过 ``tool_name_prefix=True`` 添加的 ``fin_`` 前缀。
# 前缀是客户端层的关注点，而非服务器层的，在此绕过客户端以便在多次工具调用间复用同一会话。
# 面向 Agent z ``load_fin_data_server_tools()`` 辅助函数保留 ``fin_`` 前缀用于 supervisor 消歧——
# 该路径由端到端冒烟测试 ``scripts/smoke_test_fin_data_mcp.py`` 单独验证。
EXPECTED_TOOL_NAMES: set[str] = {
    "get_stock_basic_info",
    "get_stock_price_history",
    "get_financial_abstract",
    "get_financial_indicators",
    "search_stock_by_name",
}

# 宁德时代是一只大盘、高流动性、上市时间长的股票；其财务历史足够稳定，即使 akshare 升级上游抓取目标，测试仍保持确定性。
SAMPLE_SYMBOL = "300750"
SAMPLE_NAME_KEYWORD = "宁德"


@asynccontextmanager
async def _open_session() -> AsyncIterator[dict[str, BaseTool]]:
    """启动一个 ``fin_data_server`` 子进程并 yield 其工具。

    此处返回的工具绑定到已打开的会话，因此 ``async with`` 块内任意数量的 ``ainvoke(...)`` 调用都复用*同一个*子进程。这是快速路径。
    """
    client = MultiServerMCPClient(
        {
            "fin": {
                "command": sys.executable,
                "args": ["-m", FIN_DATA_SERVER_MODULE],
                "transport": "stdio",
            }
        },
        tool_name_prefix=True,
    )
    async with client.session("fin") as session:
        # 延迟导入以避免在收集阶段支付此开销。
        from langchain_mcp_adapters.tools import load_mcp_tools

        tools = await load_mcp_tools(session)
        yield {t.name: t for t in tools}


def _parse(raw: object) -> dict[str, Any]:
    """解码 MCP 工具返回的 JSON 内容块。"""
    return json.loads(extract_text_content(raw))


# ---------------------------------------------------------------------
# 测试 1：发现 + 轻量级过滤工具（无 akshare HTTP 请求）
# ---------------------------------------------------------------------
async def test_discovery_and_search() -> None:
    """所有五个工具均已发布且 ``search_stock_by_name`` 正常工作。

    整合验证：
      - MCP 握手 + 工具 schema 往返
      - 基于内存中 A 股花名册的关键词搜索
      - 空关键词的输入校验
    """
    async with _open_session() as tools:
        assert EXPECTED_TOOL_NAMES.issubset(tools.keys()), (
            f"missing tools: {EXPECTED_TOOL_NAMES - tools.keys()}"
        )

        hit = _parse(
            await tools["search_stock_by_name"].ainvoke(
                {"keyword": SAMPLE_NAME_KEYWORD, "limit": 5}
            )
        )
        assert "error" not in hit, hit
        codes = {m["code"] for m in hit["matches"]}
        assert SAMPLE_SYMBOL in codes, (
            f"expected {SAMPLE_SYMBOL} among matches for "
            f"{SAMPLE_NAME_KEYWORD!r}, got {hit['matches']}"
        )

        bad = _parse(await tools["search_stock_by_name"].ainvoke({"keyword": "   "}))
        assert "error" in bad
        assert "non-empty" in bad["error"] or "ValueError" in bad["error"]


# ---------------------------------------------------------------------
# 测试 2：财务报表契约（不涉及不稳定的 push2 端点）
# ---------------------------------------------------------------------
async def test_financial_statement_tools() -> None:
    """``get_financial_abstract`` 和 ``get_financial_indicators`` 契约测试。

    这些工具访问新浪和东财的*报表*端点（不是 push2），这些端点是稳定的，因此可以对返回结构和数值合理性进行断言而不会产生不稳定性。
    """
    async with _open_session() as tools:
        abs_payload = _parse(
            await tools["get_financial_abstract"].ainvoke(
                {"symbol": SAMPLE_SYMBOL, "last_n_periods": 2}
            )
        )
        assert "error" not in abs_payload, abs_payload
        assert abs_payload["symbol"] == SAMPLE_SYMBOL
        assert isinstance(abs_payload["periods"], list) and 1 <= len(abs_payload["periods"]) <= 2
        assert isinstance(abs_payload["metrics"], dict) and abs_payload["metrics"]
        assert any("营业" in k or "净利润" in k for k in abs_payload["metrics"]), (
            f"expected revenue/profit metric, got {list(abs_payload['metrics'])}"
        )
        for values in abs_payload["metrics"].values():
            assert isinstance(values, list)
            assert len(values) == len(abs_payload["periods"])

        bad_periods = _parse(
            await tools["get_financial_abstract"].ainvoke(
                {"symbol": SAMPLE_SYMBOL, "last_n_periods": 99}
            )
        )
        assert "error" in bad_periods
        assert "last_n_periods" in bad_periods["error"] or "ValueError" in bad_periods["error"]

        ind_payload = _parse(
            await tools["get_financial_indicators"].ainvoke(
                {"symbol": SAMPLE_SYMBOL, "start_year": "2024"}
            )
        )
        assert "error" not in ind_payload, ind_payload
        assert isinstance(ind_payload["ratios"], dict) and ind_payload["ratios"]
        for values in ind_payload["ratios"].values():
            assert len(values) == len(ind_payload["periods"])


# ---------------------------------------------------------------------
# 测试 3：多源降级契约（push2 端点）
# ---------------------------------------------------------------------
async def test_basic_info_and_price_history_fallback() -> None:
    """两个基于 push2 的工具返回结构化数据或结构化失败。

    位于 ``push2*.eastmoney.com`` 的两个端点不可靠；失败时级联到雪球/新浪。此测试不要求主数据源成功——
    只要求：
      1. 成功响应携带文档允许列表中的 ``source`` 标签。
      2. 完全中断以 ``{error, attempts}`` 形式呈现，而非 Python 异常从子进程中冒泡。
      3. 超出范围的 ``days`` 在工具边界被拒绝。
    """
    async with _open_session() as tools:
        basic = _parse(await tools["get_stock_basic_info"].ainvoke({"symbol": SAMPLE_SYMBOL}))
        if "error" in basic:
            assert "attempts" in basic, basic
        else:
            assert basic["source"] in {"eastmoney", "xueqiu", "local_cache"}, basic
            assert isinstance(basic["info"], dict) and basic["info"]

        price = _parse(
            await tools["get_stock_price_history"].ainvoke({"symbol": SAMPLE_SYMBOL, "days": 15})
        )
        if "error" in price:
            assert "attempts" in price, price
        else:
            assert price["source"] in {"eastmoney", "sina"}, price
            summary = price["summary"]
            assert summary["sessions"] >= 1
            assert summary["high"] >= summary["low"] > 0
            assert "pct_change" in summary
            assert len(price["bars"]) == summary["sessions"]

        bad_days = _parse(
            await tools["get_stock_price_history"].ainvoke({"symbol": SAMPLE_SYMBOL, "days": 9999})
        )
        assert "error" in bad_days
        assert "days" in bad_days["error"] or "ValueError" in bad_days["error"]
