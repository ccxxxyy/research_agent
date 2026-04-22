"""Phase-4.1: MCP ``fin_data_server`` round-trip tests.

Why this test file is shaped the way it is
------------------------------------------
``fin_data_server`` is the data plane for every Phase-4 specialist.
Each MCP tool invocation ordinarily spawns a **fresh subprocess**,
and that subprocess pays a non-trivial startup cost before it can
answer anything:

- ``pandas`` top-level import: ~1.5 s
- ``akshare`` lazy import on first tool call: ~3-5 s
- One-off ``stock_info_a_code_name()`` roster fetch (used by
  ``search_stock_by_name``): ~6 s

If each test method created its own client and its own subprocess we
would easily blow past 60 s just on startup overhead. Instead every
test opens **one** ``client.session(...)`` context, then exercises as
many tools as it needs to cover a contract inside that single
subprocess. This pins the total cost to roughly one akshare warm-up
per test function (~10 s), and the whole file completes in under
40 s even over a slow link.

All tests hit live HTTP endpoints (akshare mirrors of 东财/雪球/新浪),
so they are tagged ``network``; offline CI should run with
``pytest -m 'not network'``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from research_agent.mcp_servers.client_factory import (
    FIN_DATA_SERVER_MODULE,
    extract_text_content,
)

pytestmark = pytest.mark.network

# Tool names as returned by ``load_mcp_tools(session)`` — i.e. RAW,
# without the ``fin_`` prefix that ``MultiServerMCPClient.get_tools()``
# would add via ``tool_name_prefix=True``. Prefixing is a client-layer
# concern, not a server-layer one, and we bypass the client here so a
# single session can be reused across tool calls. The Agent-facing
# ``load_fin_data_server_tools()`` helper keeps the ``fin_`` prefix for
# supervisor disambiguation — that path is exercised separately by the
# end-to-end smoke test in ``scripts/smoke_test_fin_data_mcp.py``.
EXPECTED_TOOL_NAMES: set[str] = {
    "get_stock_basic_info",
    "get_stock_price_history",
    "get_financial_abstract",
    "get_financial_indicators",
    "search_stock_by_name",
}

# 宁德时代 is a large, liquid, long-listed ticker; its financial
# history is stable enough that tests remain deterministic even if
# akshare upgrades its upstream scrape targets.
SAMPLE_SYMBOL = "300750"
SAMPLE_NAME_KEYWORD = "宁德"


@asynccontextmanager
async def _open_session() -> AsyncIterator[dict[str, BaseTool]]:
    """Launch one ``fin_data_server`` subprocess and yield its tools.

    The tools returned here are bound to the open session, so any
    number of ``ainvoke(...)`` calls inside the ``async with`` block
    reuse the *same* subprocess. This is the fast path.
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
        # Lazy import to avoid paying this cost at collection time.
        from langchain_mcp_adapters.tools import load_mcp_tools

        tools = await load_mcp_tools(session)
        yield {t.name: t for t in tools}


def _parse(raw: object) -> dict[str, Any]:
    """Decode the JSON content block an MCP tool returns."""
    return json.loads(extract_text_content(raw))


# ---------------------------------------------------------------------
# Test 1: discovery + the cheap filter-style tool (no akshare HTTP)
# ---------------------------------------------------------------------
async def test_discovery_and_search() -> None:
    """All five tools are advertised and ``search_stock_by_name`` works.

    Consolidates:
      - MCP handshake + tool schema round-trip
      - Keyword search against the in-memory A-share roster
      - Input validation on empty keyword
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

        bad = _parse(
            await tools["search_stock_by_name"].ainvoke({"keyword": "   "})
        )
        assert "error" in bad
        assert "non-empty" in bad["error"] or "ValueError" in bad["error"]


# ---------------------------------------------------------------------
# Test 2: financial-statement contracts (no flaky push2 endpoints)
# ---------------------------------------------------------------------
async def test_financial_statement_tools() -> None:
    """``get_financial_abstract`` and ``get_financial_indicators`` contracts.

    These tools hit 新浪 and 东财 *report* endpoints (NOT push2), which
    are stable, so we can assert on shape + value sanity without
    flakiness.
    """
    async with _open_session() as tools:
        abs_payload = _parse(
            await tools["get_financial_abstract"].ainvoke(
                {"symbol": SAMPLE_SYMBOL, "last_n_periods": 2}
            )
        )
        assert "error" not in abs_payload, abs_payload
        assert abs_payload["symbol"] == SAMPLE_SYMBOL
        assert (
            isinstance(abs_payload["periods"], list)
            and 1 <= len(abs_payload["periods"]) <= 2
        )
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
        assert (
            "last_n_periods" in bad_periods["error"]
            or "ValueError" in bad_periods["error"]
        )

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
# Test 3: multi-source fallback contract (push2 endpoints)
# ---------------------------------------------------------------------
async def test_basic_info_and_price_history_fallback() -> None:
    """Both push2-backed tools return structured data OR a structured failure.

    The two endpoints that sit on ``push2*.eastmoney.com`` are
    unreliable; we cascade to 雪球/新浪 on failure. This test does NOT
    require the primary to succeed — it only requires that:
      1. A successful response carries a ``source`` tag in the
         documented allow-list.
      2. A complete outage surfaces as ``{error, attempts}``, not a
         Python exception bubbling out of the subprocess.
      3. Out-of-range ``days`` is rejected at the tool boundary.
    """
    async with _open_session() as tools:
        basic = _parse(
            await tools["get_stock_basic_info"].ainvoke({"symbol": SAMPLE_SYMBOL})
        )
        if "error" in basic:
            assert "attempts" in basic, basic
        else:
            assert basic["source"] in {"eastmoney", "xueqiu"}, basic
            assert isinstance(basic["info"], dict) and basic["info"]

        price = _parse(
            await tools["get_stock_price_history"].ainvoke(
                {"symbol": SAMPLE_SYMBOL, "days": 15}
            )
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
            await tools["get_stock_price_history"].ainvoke(
                {"symbol": SAMPLE_SYMBOL, "days": 9999}
            )
        )
        assert "error" in bad_days
        assert "days" in bad_days["error"] or "ValueError" in bad_days["error"]
