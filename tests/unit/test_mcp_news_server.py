"""P2: MCP ``news_server`` round-trip tests.

Why this test file is shaped the way it is
------------------------------------------
``news_server`` is the news / sentiment plane of the research
pipeline. Like ``fin_data_server``, it shells out to ``akshare``
under the hood, so each MCP tool invocation pays the same fixed
startup cost (~2 s for ``pandas`` + ``akshare`` lazy import) before
it can return anything.

We therefore use the same single-session pattern as the fin-data
tests: open ONE ``client.session(...)`` context per test function and
exercise every tool we want to cover inside that context. This pins
the cost to ~one akshare warm-up per test (~5 s on a slow link).

Every test hits live HTTP endpoints (东方财富 / 财联社 / 百度财经 /
雪球),
so they are tagged ``network``; offline CI should run with
``pytest -m 'not network'``. Network outages on individual upstream
providers must not fail the test suite — each tool returns a
structured ``{"error": ..., "context": ...}`` shape on upstream
failure, and we assert "either valid payload OR structured error",
never "valid payload only".

We do NOT assert on news content (that would make the suite flaky as
real news comes and goes). We only assert on:
  - tool discovery (exactly five expected tools)
  - response schema (required keys present, correct types)
  - graceful failure paths (invalid args, unknown ticker)
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
    NEWS_SERVER_MODULE,
    extract_text_content,
)

pytestmark = pytest.mark.network

# Tool names as returned by ``load_mcp_tools(session)`` — RAW, without
# the ``news_`` prefix that ``MultiServerMCPClient.get_tools()`` adds
# in the production path. Same rationale as in
# ``test_mcp_fin_data_server.py``: prefixing is a client-layer concern.
EXPECTED_TOOL_NAMES: set[str] = {
    "get_stock_news",
    "get_market_telegraph",
    "get_hot_keywords",
    "get_economic_news",
    "get_xueqiu_discussion_hot_rank",
}

# 宁德时代 — same anchor ticker as the fin-data tests. Long-listed,
# highly covered by retail forums + news desks, so empty payloads
# would be a tool bug rather than a "no news this week" reality.
SAMPLE_SYMBOL = "300750"


@asynccontextmanager
async def _open_session() -> AsyncIterator[dict[str, BaseTool]]:
    """Launch one ``news_server`` subprocess and yield its tools.

    Tools are bound to the open session, so any number of
    ``ainvoke(...)`` calls inside the ``async with`` block reuse the
    *same* subprocess. This is the fast path.
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
    """Decode the JSON content block an MCP tool returns."""
    return json.loads(extract_text_content(raw))


# ---------------------------------------------------------------------
# Test 1: discovery + the simplest stock-news round-trip
# ---------------------------------------------------------------------
async def test_discovery_and_stock_news() -> None:
    """All five tools advertised, ``get_stock_news`` returns a valid frame.

    Consolidates:
      - MCP handshake + tool schema round-trip
      - 东方财富 individual-stock news payload shape
      - ``limit`` honoured / capped
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
                "each news row should be a non-empty dict of column→value"
            )


# ---------------------------------------------------------------------
# Test 2: telegraph contract — flash feed + category validation
# ---------------------------------------------------------------------
async def test_market_telegraph_contract() -> None:
    """``get_market_telegraph`` round-trip + category-allowlist enforcement.

    The 财联社 endpoint only supports ``全部`` and ``重点`` filters.
    A bad category must surface as a structured error at the tool
    boundary BEFORE we hit the network — that's how we keep the LLM
    from wasting tokens probing invalid categories.
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
                {"category": "宏观"}  # not in allow-list
            )
        )
        assert "error" in bad
        assert "category" in bad["error"] or "ValueError" in bad["error"]


# ---------------------------------------------------------------------
# Test 3: hot-keyword symbol normalisation + economic-news date guard
# ---------------------------------------------------------------------
async def test_hot_keywords_and_economic_news() -> None:
    """Two contracts in one session.

    1. ``get_hot_keywords`` must accept a plain 6-digit ticker and
       internally normalise to ``SH``/``SZ``-prefixed form before
       calling akshare. The LLM should not be expected to know about
       the prefix quirk — that's why this normalisation lives in the
       tool.
    2. ``get_economic_news`` must reject malformed ``date`` arguments
       at the boundary (we do NOT want to swallow a typo and silently
       query "today"; that hides bugs in the agent prompt).
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
            # Normalised back to upper-case prefixed form on the wire:
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
# Test 4: 雪球讨论榜 — invalid ranking (fast) + optional live call
# ---------------------------------------------------------------------
async def test_xueqiu_discussion_rank_contract() -> None:
    """``get_xueqiu_discussion_hot_rank`` validation + schema.

    Invalid ``ranking`` must error before any HTTP. A valid call hits
    xueqiu and may be slow (full screener pagination in akshare).
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
