"""Unit tests for API middleware (auth + rate limiting).

These tests verify:
  * In-memory sliding-window rate limiting (no Redis)
  * Redis-backed sliding-window rate limiting (with fakeredis)
  * Transparent fallback when Redis is unreachable
  * Auth-exempt paths bypass rate limiting
  * Retry-After header on 429 responses
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from research_agent.api.middleware import RateLimitMiddleware


def _build_app(*, max_rpm: int = 3, redis_url: str | None = None) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        RateLimitMiddleware, max_rpm=max_rpm, redis_url=redis_url,
    )

    @app.get("/health")
    async def _health():
        return {"ok": True}

    @app.get("/api/test")
    async def _test():
        return {"msg": "hello"}

    return app


def _find_rate_limit_middleware(app: FastAPI) -> RateLimitMiddleware | None:
    """Walk the ASGI middleware stack to find our RateLimitMiddleware.

    The stack is lazily built by Starlette on the first request, so
    call this *after* at least one request has been made.
    """
    obj: Any = app.middleware_stack
    while obj is not None:
        if isinstance(obj, RateLimitMiddleware):
            return obj
        obj = getattr(obj, "app", None)
    return None


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------


class TestInMemoryRateLimit:
    @pytest.mark.asyncio
    async def test_allows_under_limit(self) -> None:
        app = _build_app(max_rpm=5)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            for _ in range(5):
                r = await client.get("/api/test")
                assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_rejects_over_limit_with_429(self) -> None:
        app = _build_app(max_rpm=2)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.get("/api/test")
            await client.get("/api/test")
            r = await client.get("/api/test")
            assert r.status_code == 429
            assert "Retry-After" in r.headers
            assert "Rate limit exceeded" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_health_exempt_from_limit(self) -> None:
        app = _build_app(max_rpm=1)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.get("/api/test")  # uses the 1 allowed
            for _ in range(5):
                r = await client.get("/health")
                assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_window_expires(self, monkeypatch: pytest.MonkeyPatch) -> None:
        app = _build_app(max_rpm=1)
        fake_time = time.time()

        monkeypatch.setattr(time, "time", lambda: fake_time)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/api/test")
            assert r.status_code == 200
            r = await client.get("/api/test")
            assert r.status_code == 429

            fake_time += 61
            monkeypatch.setattr(time, "time", lambda: fake_time)

            r = await client.get("/api/test")
            assert r.status_code == 200


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------


class TestRedisRateLimit:
    @pytest.mark.asyncio
    async def test_redis_allows_under_limit(self) -> None:
        fakeredis = pytest.importorskip("fakeredis")
        from fakeredis.aioredis import FakeRedis

        app = _build_app(max_rpm=3)

        # Trigger middleware stack build
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.get("/health")

        rl = _find_rate_limit_middleware(app)
        assert rl is not None

        fake_client = FakeRedis(decode_responses=True)
        rl._redis = fake_client
        rl._lua_sha = None

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            for _ in range(3):
                r = await client.get("/api/test")
                assert r.status_code == 200

            r = await client.get("/api/test")
            assert r.status_code == 429

        await fake_client.aclose()

    @pytest.mark.asyncio
    async def test_redis_failure_falls_back_to_memory(self) -> None:
        app = _build_app(max_rpm=3)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.get("/health")

        rl = _find_rate_limit_middleware(app)
        assert rl is not None

        broken_redis = AsyncMock()
        broken_redis.evalsha = AsyncMock(side_effect=ConnectionError("down"))
        broken_redis.script_load = AsyncMock(return_value="fakeSHA")
        rl._redis = broken_redis

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            for _ in range(3):
                r = await client.get("/api/test")
                assert r.status_code == 200

            r = await client.get("/api/test")
            assert r.status_code == 429

    @pytest.mark.asyncio
    async def test_redis_counts_across_requests(self) -> None:
        fakeredis = pytest.importorskip("fakeredis")
        from fakeredis.aioredis import FakeRedis

        app = _build_app(max_rpm=2)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.get("/health")

        rl = _find_rate_limit_middleware(app)
        assert rl is not None

        fake_client = FakeRedis(decode_responses=True)
        rl._redis = fake_client
        rl._lua_sha = None

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r1 = await client.get("/api/test")
            r2 = await client.get("/api/test")
            r3 = await client.get("/api/test")

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 429
        assert int(r3.headers["Retry-After"]) > 0

        await fake_client.aclose()
