"""市场判定与偏好（P0）单元测试。"""

from __future__ import annotations

import pytest
from langgraph.store.memory import InMemoryStore

from research_agent.market import (
    PREFERRED_MARKET_KEY,
    AssetClass,
    Market,
    build_mixed_orchestration_plan,
    detect_market_from_query,
    extract_symbols_from_query,
    format_market_preamble,
    parse_market_override,
    parse_preferred_market,
    resolve_market,
    set_user_preferred_market,
)
from research_agent.memory.manager import MemoryManager, MemoryNamespace


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CN_A", Market.CN_A),
        ("us", Market.US),
        ("A股", None),  # 非枚举字符串
        ("USA", Market.US),
        ("", None),
    ],
)
def test_parse_preferred_market(raw: str, expected: Market | None) -> None:
    assert parse_preferred_market(raw) is expected


def test_detect_us_by_chinese_name() -> None:
    r = detect_market_from_query("帮我看看特斯拉最近怎么样")
    assert r.market == Market.US
    assert any(s.ticker == "TSLA" for s in r.symbols)


def test_detect_us_by_ticker() -> None:
    r = detect_market_from_query("What is the outlook for AAPL?")
    assert r.market == Market.US
    assert any(s.ticker == "AAPL" for s in r.symbols)


def test_detect_cn_by_code() -> None:
    r = detect_market_from_query("300750 近五日行情")
    assert r.market == Market.CN_A
    assert any(s.ticker == "300750" for s in r.symbols)


def test_detect_cn_by_name() -> None:
    r = detect_market_from_query("贵州茅台的ROE怎么样")
    assert r.market == Market.CN_A
    assert any(s.ticker == "600519" for s in r.symbols)


def test_detect_mixed() -> None:
    r = detect_market_from_query("中美对比一下宁德时代和特斯拉")
    assert r.market == Market.MIXED


def test_detect_unknown_without_signal() -> None:
    r = detect_market_from_query("今天天气怎么样")
    assert r.market == Market.UNKNOWN


def test_extract_etf_and_index() -> None:
    syms = extract_symbols_from_query("QQQ 和 标普500")
    tickers = {s.ticker for s in syms}
    assert "QQQ" in tickers
    assert "^GSPC" in tickers
    assert any(s.asset_class == AssetClass.ETF for s in syms)
    assert any(s.asset_class == AssetClass.INDEX for s in syms)


@pytest.mark.asyncio
async def test_resolve_falls_back_to_preference() -> None:
    store = InMemoryStore()
    memory = MemoryManager(store)
    await set_user_preferred_market(memory, "u1", Market.US)

    r = await resolve_market("随便聊聊估值方法", memory=memory, user_id="u1")
    assert r.market == Market.US
    assert r.source == "user_preference"


@pytest.mark.asyncio
async def test_resolve_query_beats_preference() -> None:
    store = InMemoryStore()
    memory = MemoryManager(store)
    await set_user_preferred_market(memory, "u1", Market.US)

    r = await resolve_market("茅台今天涨了吗", memory=memory, user_id="u1")
    assert r.market == Market.CN_A
    assert r.source == "query_signal"
    assert r.preferred_market == Market.US


@pytest.mark.asyncio
async def test_resolve_override() -> None:
    r = await resolve_market("特斯拉", override="CN_A")
    assert r.market == Market.CN_A
    assert r.source == "request_override"


@pytest.mark.asyncio
async def test_resolve_default_cn_a() -> None:
    r = await resolve_market("你好")
    assert r.market == Market.CN_A
    assert r.source == "default"


@pytest.mark.asyncio
async def test_resolve_thread_sticky_beats_default() -> None:
    r = await resolve_market("预测下周情况 不同情况发生的概率", sticky_market="US")
    assert r.market == Market.US
    assert r.source == "thread_sticky"


@pytest.mark.asyncio
async def test_resolve_query_beats_thread_sticky() -> None:
    r = await resolve_market("贵州茅台今天怎么样", sticky_market="US")
    assert r.market == Market.CN_A
    assert r.source == "query_signal"


@pytest.mark.asyncio
async def test_resolve_thread_sticky_beats_preference() -> None:
    store = InMemoryStore()
    memory = MemoryManager(store)
    await set_user_preferred_market(memory, "u1", Market.CN_A)
    r = await resolve_market(
        "每一种情况都仔细分析前因后果",
        memory=memory,
        user_id="u1",
        sticky_market="US",
    )
    assert r.market == Market.US
    assert r.source == "thread_sticky"


def test_format_preamble_contains_routing_constraint() -> None:
    r = detect_market_from_query("AAPL")
    text = format_market_preamble(r)
    assert "MarketResolution" in text
    assert "禁止" in text
    assert "US" in text


def test_parse_market_override_accepts_mixed() -> None:
    assert parse_market_override("MIXED") is Market.MIXED
    assert parse_market_override("auto") is None
    assert parse_preferred_market("MIXED") is None


def test_build_mixed_plan_sides() -> None:
    q = "对比宁德时代和特斯拉最近的股价表现"
    r = detect_market_from_query(q)
    plan = build_mixed_orchestration_plan(r, q)
    assert plan is not None
    assert plan.is_comparison is True
    assert {t.side for t in plan.subtasks} >= {Market.CN_A, Market.US}


def test_format_preamble_mixed_includes_orchestration() -> None:
    q = "对比宁德时代和特斯拉最近的股价表现"
    r = detect_market_from_query(q)
    text = format_market_preamble(r, query=q)
    assert r.market == Market.MIXED
    assert "[MixedOrchestration]" in text
    assert "data_expert" in text
    assert "us_data_expert" in text
    assert "mode=comparison" in text


@pytest.mark.asyncio
async def test_resolve_override_mixed_string() -> None:
    r = await resolve_market("随便问问", override="MIXED")
    assert r.market == Market.MIXED
    assert r.source == "request_override"


@pytest.mark.asyncio
async def test_set_preferred_market_rejects_mixed() -> None:
    store = InMemoryStore()
    memory = MemoryManager(store)
    with pytest.raises(ValueError):
        await set_user_preferred_market(memory, "u1", "MIXED")

    item = await memory.get_memory("u1", MemoryNamespace.USER_PREFERENCES, PREFERRED_MARKET_KEY)
    assert item is None
