"""自选 → Agent memory / 问答前导注入。"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.store.memory import InMemoryStore

from research_agent.api.routes import supervisor as supervisor_route
from research_agent.market.types import Market, MarketResolution
from research_agent.memory.manager import MemoryManager, MemoryNamespace
from research_agent.memory.watchlist_store import (
    WATCHLIST_MEMORY_KEY,
    WatchlistStore,
    format_watchlist_context,
    snapshot_watchlist,
    sync_watchlist_to_memory,
)


def test_format_watchlist_context_empty():
    assert format_watchlist_context([], []) == ""


def test_format_watchlist_context_both_markets():
    text = format_watchlist_context(
        [{"symbol": "600519", "display_name": "贵州茅台", "asset_class": "equity"}],
        [{"symbol": "AAPL", "display_name": "Apple", "asset_class": "equity"}],
    )
    assert "CN_A" in text
    assert "600519" in text
    assert "AAPL" in text
    assert "自选" in text


@pytest.mark.asyncio
async def test_sync_watchlist_to_memory(tmp_path):
    store = WatchlistStore(db_path=tmp_path / "wl.db")
    store.add_item("u1", "CN_A", "600519", display_name="贵州茅台", asset_class="equity")
    store.add_item("u1", "US", "AAPL", display_name="Apple", asset_class="equity")
    memory = MemoryManager(InMemoryStore())

    snap = await sync_watchlist_to_memory(memory, store, "u1")
    assert snap["count"] == 2

    mem = await memory.get_memory("u1", MemoryNamespace.WATCHLIST, WATCHLIST_MEMORY_KEY)
    assert mem is not None
    assert "600519" in mem["content"]
    assert "AAPL" in mem["content"]

    ctx = await memory.get_user_context("u1")
    assert ctx["watchlist"] is not None
    assert ctx["watchlist"]["cn_count"] == 1


@pytest.mark.asyncio
async def test_build_user_context_injects_watchlist(tmp_path):
    store = WatchlistStore(db_path=tmp_path / "wl2.db")
    store.add_item("demo", "US", "CL=F", display_name="WTI", asset_class="future")
    memory = MemoryManager(InMemoryStore())
    resolution = MarketResolution(market=Market.US, source="query", confidence=0.9)

    with (
        patch(
            "research_agent.market.resolve_market",
            new=AsyncMock(return_value=resolution),
        ),
        patch(
            "research_agent.market.format_market_preamble",
            return_value="Market: US",
        ),
    ):
        messages, _ = await supervisor_route._build_user_context_messages(
            memory,
            "demo",
            "看看我的自选",
            watchlist_store=store,
        )

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "CL=F" in messages[0].content
    assert "User dashboard watchlist" in messages[0].content


@pytest.mark.asyncio
async def test_build_user_context_falls_back_to_memory():
    memory = MemoryManager(InMemoryStore())
    await memory.save_memory(
        "demo",
        MemoryNamespace.WATCHLIST,
        WATCHLIST_MEMORY_KEY,
        {
            "content": (
                "User dashboard watchlist (when the user refers to 自选/关注/我的股票 "
                "without naming tickers, prefer these symbols):\n- US: SPY(etf)"
            ),
            "cn_count": 0,
            "us_count": 1,
        },
    )
    resolution = MarketResolution(market=Market.US, source="default", confidence=0.5)

    with (
        patch(
            "research_agent.market.resolve_market",
            new=AsyncMock(return_value=resolution),
        ),
        patch(
            "research_agent.market.format_market_preamble",
            return_value="Market: US",
        ),
    ):
        messages, _ = await supervisor_route._build_user_context_messages(
            memory,
            "demo",
            "自选怎么样",
            watchlist_store=None,
        )

    assert "SPY" in messages[0].content


def test_snapshot_watchlist(tmp_path):
    store = WatchlistStore(db_path=tmp_path / "wl3.db")
    store.add_item("u", "CN_A", "510300", display_name="沪深300ETF", asset_class="etf")
    snap = snapshot_watchlist(store, "u")
    assert snap["count"] == 1
    assert "510300" in snap["content"]
