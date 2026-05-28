"""P2：MCP ``news_server`` 往返测试。

为什么此测试文件如此组织
------------------------------------------
``news_server`` 是研究流水线的新闻/情绪平面。与 ``fin_data_server``类似，它底层调用 ``akshare``，
因此每次 MCP 工具调用在返回结果之前都需要支付相同的固定启动开销（``pandas`` + ``akshare`` 懒加载导入约 2 秒）。

因此采用与金融数据测试相同的单会话模式：每个测试函数打开一个``client.session(...)`` 上下文，并在该上下文中执行所有需要覆盖的工具。
这将开销固定在每个测试约一次 akshare 预热（慢速链路约 5 秒）。

所有测试访问实时 HTTP 端点（东方财富 / 财联社 / 百度财经 / 雪球），
因此标记为 ``network``；离线 CI 应使用``pytest -m 'not network'`` 运行。
单个上游提供商的网络中断不应导致测试套件失败 — 每个工具在上游故障时返回结构化的``{"error": ..., "context": ...}`` 形式，
断言"有效载荷或结构化错误"，而非"仅有效载荷"。

不对新闻内容进行断言（那会使套件变得脆弱，因为真实新闻来去不定）。仅断言：
  - 工具发现（恰好五个预期工具）
  - 响应 schema（必需键存在、类型正确）
  - 优雅的失败路径（无效参数、未知股票代码）
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient

from research_agent.mcp_servers.client_factory import (
    NEWS_SERVER_MODULE,
    extract_text_content,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langchain_core.tools import BaseTool

pytestmark = pytest.mark.network

# 工具名称由 ``load_mcp_tools(session)`` 返回 — 原始名称，不带``MultiServerMCPClient.get_tools()`` 在生产路径中添加的 ``news_`` 前缀。
# 与 ``test_mcp_fin_data_server.py`` 相同的原理：前缀是客户端层的关注点。
EXPECTED_TOOL_NAMES: set[str] = {
    "get_stock_news",
    "get_market_telegraph",
    "get_hot_keywords",
    "get_economic_news",
    "get_xueqiu_discussion_hot_rank",
}

# 宁德时代 — 与金融数据测试相同的锚定股票代码。上市时间长，散户论坛和新闻报道覆盖面广，因此空载荷意味着工具 bug 而非"本周无新闻"的现实情况。
SAMPLE_SYMBOL = "300750"


@asynccontextmanager
async def _open_session() -> AsyncIterator[dict[str, BaseTool]]:
    """启动一个 ``news_server`` 子进程并 yield 其工具。

    工具绑定到已打开的会话，因此 ``async with`` 块内任意数量的 ``ainvoke(...)`` 调用都复用*同一个*子进程。这是快速路径。
    """
    client = MultiServerMCPClient(
        {
            "news": {
                "command": sys.executable,
                "args": ["-m", NEWS_SERVER_MODULE],
                "transport": "stdio",
            }
        },
        tool_name_prefix=True,
    )
    async with client.session("news") as session:
        from langchain_mcp_adapters.tools import load_mcp_tools

        tools = await load_mcp_tools(session)
        yield {t.name: t for t in tools}


def _parse(raw: object) -> dict[str, Any]:
    """解码 MCP 工具返回的 JSON 内容块。"""
    return json.loads(extract_text_content(raw))


# ---------------------------------------------------------------------
# 测试 1：发现 + 最简单的个股新闻往返
# ---------------------------------------------------------------------
async def test_discovery_and_stock_news() -> None:
    """所有五个工具均已发布，``get_stock_news`` 返回有效数据帧。

    整合验证：
      - MCP 握手 + 工具 schema 往返
      - 东方财富个股新闻载荷形状
      - ``limit`` 被遵守/限制
    """
    async with _open_session() as tools:
        assert EXPECTED_TOOL_NAMES.issubset(tools.keys()), (
            f"missing tools: {EXPECTED_TOOL_NAMES - tools.keys()}"
        )

        payload = _parse(
            await tools["get_stock_news"].ainvoke(
                {"symbol": SAMPLE_SYMBOL, "limit": 5}
            )
        )
        if "error" in payload:
            assert "context" in payload, payload
            return  # upstream blip — accept structured failure
        assert payload["symbol"] == SAMPLE_SYMBOL
        assert payload["source"] == "eastmoney"
        assert isinstance(payload["news"], list)
        assert payload["count"] == len(payload["news"])
        assert len(payload["news"]) <= 5, "limit=5 must be honoured"
        if payload["news"]:
            row = payload["news"][0]
            assert isinstance(row, dict) and row, (
                "每条新闻应为非空的 列名→值 字典"
            )


# ---------------------------------------------------------------------
# 测试 2：电报契约 — 快讯流 + 分类校验
# ---------------------------------------------------------------------
async def test_market_telegraph_contract() -> None:
    """``get_market_telegraph`` 往返 + 分类白名单强制执行。

    财联社端点仅支持 ``全部`` 和 ``重点`` 过滤器。无效分类必须在工具边界以结构化错误形式暴露（在访问网络之前）—— 这样可以 防止 LLM 浪费 token 探测无效分类。
    """
    async with _open_session() as tools:
        payload = _parse(
            await tools["get_market_telegraph"].ainvoke(
                {"category": "全部", "limit": 5}
            )
        )
        if "error" in payload:
            assert "context" in payload, payload
        else:
            assert payload["category"] == "全部"
            assert payload["source"] == "cls"
            assert isinstance(payload["telegraph"], list)
            assert payload["count"] == len(payload["telegraph"])
            assert len(payload["telegraph"]) <= 5

        bad = _parse(
            await tools["get_market_telegraph"].ainvoke(
                {"category": "宏观"}  # 不在白名单中
            )
        )
        assert "error" in bad
        assert "category" in bad["error"] or "ValueError" in bad["error"]


# ---------------------------------------------------------------------
# 测试 3：热词股票代码规范化 + 经济新闻日期校验
# ---------------------------------------------------------------------
async def test_hot_keywords_and_economic_news() -> None:
    """一个会话中验证两个契约。

    1. ``get_hot_keywords`` 必须接受纯 6 位股票代码，并在调用 akshare 之前内部规范化为 ``SH``/``SZ`` 前缀形式。不应期望 LLM 了解前缀细节 — 是为什么此规范化逻辑放在工具中。
    2. ``get_economic_news`` 必须在边界拒绝格式错误的 ``date`` 参数（不希望吞掉拼写错误并静默查询"今天"；那会隐藏 Agent 提示词中的 bug）。
    """
    async with _open_session() as tools:
        payload = _parse(
            await tools["get_hot_keywords"].ainvoke(
                {"symbol": SAMPLE_SYMBOL, "limit": 5}
            )
        )
        if "error" in payload:
            assert "context" in payload, payload
        else:
            assert payload["source"] == "eastmoney"
            assert isinstance(payload["keywords"], list)
            assert payload["count"] == len(payload["keywords"])
            assert len(payload["keywords"]) <= 5
            # 在线路上规范化回大写前缀形式：
            assert payload["symbol"].upper().startswith(("SH", "SZ"))
            assert SAMPLE_SYMBOL in payload["symbol"]

        econ = _parse(
            await tools["get_economic_news"].ainvoke({"limit": 5})
        )
        if "error" not in econ:
            assert econ["source"] == "baidu"
            assert isinstance(econ["news"], list)
            assert econ["count"] == len(econ["news"])
            assert len(econ["news"]) <= 5
            assert econ["date"].isdigit() and len(econ["date"]) == 8
        else:
            assert "context" in econ, econ

        bad = _parse(
            await tools["get_economic_news"].ainvoke({"date": "2026-05-08"})
        )
        assert "error" in bad
        assert "date" in bad["error"] or "YYYYMMDD" in bad["error"]


# ---------------------------------------------------------------------
# 测试 4：雪球讨论榜 — 无效排名（快速）+ 可选的实时调用
# ---------------------------------------------------------------------
async def test_xueqiu_discussion_rank_contract() -> None:
    """``get_xueqiu_discussion_hot_rank`` 校验 + schema 验证。

    无效 ``ranking`` 必须在任何 HTTP 请求之前报错。有效调用会访问雪球且可能较慢（akshare 中的全量筛选器分页）。
    """
    async with _open_session() as tools:
        bad = _parse(
            await tools["get_xueqiu_discussion_hot_rank"].ainvoke(
                {"ranking": "全天热帖"}
            )
        )
        assert "error" in bad
        assert "ranking" in bad["error"] or "ValueError" in bad["error"]

        payload = _parse(
            await tools["get_xueqiu_discussion_hot_rank"].ainvoke(
                {"ranking": "最热门", "limit": 5}
            )
        )
        if "error" in payload:
            assert "context" in payload, payload
            return
        assert payload["ranking"] == "最热门"
        assert payload["source"] == "xueqiu"
        assert isinstance(payload["stocks"], list)
        assert payload["count"] == len(payload["stocks"])
        assert len(payload["stocks"]) <= 5
        if payload["stocks"]:
            row = payload["stocks"][0]
            assert "讨论量" in row or "最新价" in row
