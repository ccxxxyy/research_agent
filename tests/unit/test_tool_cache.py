"""MCP 工具结果分层 TTL 缓存的单元测试。"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from research_agent.cache.tool_cache import (
    TTL_REALTIME,
    ToolResultCache,
    _MemoryBackend,
    cached_tool,
    get_tool_cache,
    reset_tool_cache_for_tests,
)


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """为本模块显式开启缓存（全局 conftest 默认关闭）。"""
    monkeypatch.setenv("TOOL_CACHE_ENABLED", "true")
    monkeypatch.setenv("TOOL_CACHE_BACKEND", "memory")
    reset_tool_cache_for_tests()
    yield
    reset_tool_cache_for_tests()


@pytest.mark.asyncio
async def test_hit_on_second_call_same_args() -> None:
    calls = {"n": 0}

    @cached_tool(ttl=TTL_REALTIME, namespace="fin")
    async def get_index_quotes() -> dict[str, Any]:
        calls["n"] += 1
        return {"quotes": [{"code": "000001", "price": 3000.0 + calls["n"]}]}

    first = await get_index_quotes()
    second = await get_index_quotes()

    assert calls["n"] == 1
    assert first == second
    assert first["quotes"][0]["price"] == 3001.0
    stats = get_tool_cache().stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


@pytest.mark.asyncio
async def test_different_args_are_separate_keys() -> None:
    calls: list[str] = []

    @cached_tool(ttl=60, namespace="fin")
    async def get_stock_news(symbol: str, limit: int = 20) -> dict[str, Any]:
        calls.append(symbol)
        return {"symbol": symbol, "items": [f"news-{symbol}"]}

    a = await get_stock_news("300750")
    b = await get_stock_news("600519")
    a2 = await get_stock_news("300750")

    assert calls == ["300750", "600519"]
    assert a["symbol"] == "300750"
    assert b["symbol"] == "600519"
    assert a2 == a


@pytest.mark.asyncio
async def test_error_payload_is_not_cached() -> None:
    calls = {"n": 0}

    @cached_tool(ttl=60, namespace="fin")
    async def flaky() -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            return {"error": "ConnectionError: timeout", "context": "flaky"}
        return {"ok": True}

    first = await flaky()
    second = await flaky()

    assert "error" in first
    assert second == {"ok": True}
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_ttl_expiry_triggers_refetch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    clock = {"t": 1000.0}

    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])

    # 重建缓存，使后端在 set/get 时使用被 patch 的 time.monotonic。
    reset_tool_cache_for_tests()
    cache = get_tool_cache()
    assert isinstance(cache._backend, _MemoryBackend)

    @cached_tool(ttl=10, namespace="fin")
    async def tick() -> dict[str, Any]:
        calls["n"] += 1
        return {"n": calls["n"]}

    await tick()
    clock["t"] += 5
    await tick()
    assert calls["n"] == 1

    clock["t"] += 6  # 超过 TTL
    await tick()
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_disabled_cache_always_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOOL_CACHE_ENABLED", "false")
    reset_tool_cache_for_tests()
    calls = {"n": 0}

    @cached_tool(ttl=60, namespace="fin")
    async def ping() -> dict[str, Any]:
        calls["n"] += 1
        return {"ok": True}

    await ping()
    await ping()
    assert calls["n"] == 2
    assert get_tool_cache().enabled is False


@pytest.mark.asyncio
async def test_returned_value_is_deep_copied() -> None:
    """修改返回的 dict 不得污染缓存条目。"""

    @cached_tool(ttl=60, namespace="fin")
    async def payload() -> dict[str, Any]:
        return {"items": [1, 2, 3]}

    first = await payload()
    first["items"].append(99)
    second = await payload()
    assert second["items"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_memory_backend_clear() -> None:
    backend = _MemoryBackend()
    await backend.set("k", {"v": 1}, ttl=60)
    assert await backend.get("k") == {"v": 1}
    await backend.clear()
    assert await backend.get("k") is None


@pytest.mark.asyncio
async def test_concurrent_same_key_only_one_upstream_ideal_not_required() -> None:
    """冒烟：同 key 并发调用不崩溃；至少成功一次上游调用。"""
    calls = {"n": 0}
    gate = asyncio.Event()

    @cached_tool(ttl=60, namespace="fin")
    async def slow() -> dict[str, Any]:
        calls["n"] += 1
        await gate.wait()
        return {"n": calls["n"]}

    async def caller() -> dict[str, Any]:
        return await slow()

    tasks = [asyncio.create_task(caller()) for _ in range(5)]
    await asyncio.sleep(0.05)
    gate.set()
    results = await asyncio.gather(*tasks)
    # 未做 singleflight 时 5 次都可能 miss；只断言全部成功返回。
    assert all(isinstance(r, dict) for r in results)
    assert calls["n"] >= 1


@pytest.mark.asyncio
async def test_tool_result_cache_stats() -> None:
    cache = ToolResultCache(_MemoryBackend(), enabled=True)
    await cache.set("a", {"x": 1}, ttl=30)
    assert await cache.get("a") == {"x": 1}
    assert await cache.get("missing") is None
    await cache.set("err", {"error": "boom"}, ttl=30)
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1
    assert cache.stats()["skips"] == 1
