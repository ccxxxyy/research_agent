"""API authentication, timeouts, and rate-limiting middleware.

Authentication
--------------
Bearer-token check against the ``API_SECRET_KEY`` env var. When the
key is empty (development default), authentication is **disabled** —
all requests pass through. This avoids breaking local ``curl`` / UI
dev workflows while keeping production locked down with a single env
var flip.

Endpoints exempt from auth:
- ``GET /health`` — must be reachable by orchestrators unconditionally.
- ``GET /docs``, ``GET /openapi.json`` — Swagger UI for dev convenience.

Rate Limiting
-------------
Sliding-window counter keyed by client IP. Two backends:

1. **Redis** (distributed) — when ``redis_url`` is provided and the
   server is reachable. Uses a sorted-set + Lua script for atomic
   per-key counting. Works across multiple app instances behind a
   load balancer.
2. **In-memory** (fallback) — a ``dict`` keyed by client IP. Holds at
   most a few dozen floats per key. A periodic sweep drops keys whose
   timestamps have fully aged out of the window to cap memory under
   churning client-IP scans.

Backend selection happens once at startup; if Redis becomes
unreachable *after* startup, each failing request transparently
falls back to in-memory counting and logs a warning (no 500).

The window is 60 seconds; the cap is ``RATE_LIMIT_RPM`` (default 30
requests/minute). Exceeding the limit returns 429 with a
``Retry-After`` header.

Request-ID tracing
-------------------
``RequestIdMiddleware`` generates (or propagates) a unique trace ID
for every request. The ID is injected into the loguru context so
that log lines emitted during the request are correlated, and is
echoed back in the ``X-Request-ID`` response header.

HTTP request timeout (optional)
--------------------------------
When configured with ``http_request_timeout_seconds > 0``,
``RequestTimeoutMiddleware`` wraps downstream handlers with
``asyncio.wait_for``. Long SSE streams exclude
``/api/supervisor/research/stream``.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable

from loguru import logger

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

_AUTH_EXEMPT_PATHS = frozenset({
    "/health",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
})

_REQUEST_TIMEOUT_EXEMPT_PATHS = _AUTH_EXEMPT_PATHS | frozenset({
    "/api/supervisor/research/stream",
})


class AuthMiddleware(BaseHTTPMiddleware):
    """Reject requests without a valid Bearer token.

    Disabled (pass-through) when ``secret_key`` is empty.
    """

    def __init__(self, app: ASGIApp, *, secret_key: str = "") -> None:
        super().__init__(app)
        self._secret_key = secret_key

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self._secret_key:
            return await call_next(request)

        if request.url.path in _AUTH_EXEMPT_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if auth_header == f"Bearer {self._secret_key}":
            return await call_next(request)

        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key. Provide 'Authorization: Bearer <key>'."},
        )


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Generate or propagate a unique request trace ID.

    If the incoming request carries ``X-Request-ID`` (set by a reverse
    proxy or the client), that value is reused.  Otherwise a UUID-4 is
    minted.  The resolved ID is:

    * stored on ``request.state.request_id`` for downstream handlers,
    * bound into the loguru context (``extra["request_id"]``) so every
      log line emitted during the request carries the trace ID, and
    * echoed back as the ``X-Request-ID`` response header.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        with logger.contextualize(request_id=request_id):
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """Optional ceiling on ASGI handler wall-clock time."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        timeout_seconds: float = 0.0,
    ) -> None:
        super().__init__(app)
        self._timeout = float(timeout_seconds)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if self._timeout <= 0:
            return await call_next(request)
        path = request.url.path
        if path in _REQUEST_TIMEOUT_EXEMPT_PATHS:
            return await call_next(request)
        try:
            return await asyncio.wait_for(call_next(request), timeout=self._timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Request timed out after {:.1f}s: {} {}",
                self._timeout,
                request.method,
                path,
            )
            return JSONResponse(
                status_code=504,
                content={"detail": "Gateway timeout — handler exceeded configured limit."},
            )


_SLIDING_WINDOW_LUA = """\
local key     = KEYS[1]
local now     = tonumber(ARGV[1])
local window  = tonumber(ARGV[2])
local limit   = tonumber(ARGV[3])
local member  = ARGV[4]

redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)
if count >= limit then
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    return {0, oldest[2] or tostring(now - window)}
end
redis.call('ZADD', key, now, member)
redis.call('EXPIRE', key, math.ceil(window))
return {1, tostring(count + 1)}
"""


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter keyed by client IP.

    When ``redis_url`` is provided, uses a Redis sorted-set with
    an atomic Lua script for distributed counting across instances.
    Falls back to in-memory counting when Redis is unavailable.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_rpm: int = 30,
        redis_url: str | None = None,
    ) -> None:
        super().__init__(app)
        self._max_rpm = max_rpm
        self._window = 60.0
        self._key_prefix = "rl:rpm:"

        # In-memory fallback (always available)
        self._requests: dict[str, list[float]] = {}
        self._memory_tick: int = 0

        # Redis backend (best-effort)
        self._redis = None
        self._lua_sha: str | None = None
        if redis_url:
            self._redis = self._try_connect_redis(redis_url)

    # -- Redis bootstrap ---------------------------------------------------

    @staticmethod
    def _try_connect_redis(redis_url: str):
        """Return an ``redis.asyncio.Redis`` instance, or ``None``."""
        try:
            from redis.asyncio import Redis

            client = Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=1,
            )
            logger.info("RateLimitMiddleware: Redis configured ({})", redis_url)
            return client
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RateLimitMiddleware: Redis unavailable ({}), using in-memory fallback",
                exc,
            )
            return None

    async def _ensure_lua_script(self) -> str | None:
        """Load the Lua script into Redis (cached via EVALSHA)."""
        if self._lua_sha is not None:
            return self._lua_sha
        if self._redis is None:
            return None
        try:
            self._lua_sha = await self._redis.script_load(_SLIDING_WINDOW_LUA)
            return self._lua_sha
        except Exception:  # noqa: BLE001
            return None

    # -- IP extraction -----------------------------------------------------

    @staticmethod
    def _client_ip(request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    # -- In-memory backend -------------------------------------------------

    def _reap_stale_memory_keys(self, now: float) -> None:
        """Remove IPs whose timestamps are entirely outside the window."""
        cutoff = now - self._window
        for ip_key in list(self._requests.keys()):
            ts = self._requests[ip_key]
            while ts and ts[0] < cutoff:
                ts.pop(0)
            if not ts:
                self._requests.pop(ip_key, None)

    def _tick_memory_maintenance(self, now: float) -> None:
        """Bounded-cost sweep keyed off a modest request counter."""
        self._memory_tick += 1
        if self._memory_tick >= 4096:
            self._memory_tick = 0
            self._reap_stale_memory_keys(now)

    def _check_memory(self, ip: str, now: float) -> tuple[bool, int]:
        """Return ``(allowed, retry_after_seconds)``."""
        self._tick_memory_maintenance(now)

        timestamps = self._requests.get(ip)
        if timestamps is None:
            timestamps = []

        cutoff = now - self._window
        while timestamps and timestamps[0] < cutoff:
            timestamps.pop(0)

        if len(timestamps) >= self._max_rpm:
            retry_after = int(self._window - (now - timestamps[0])) + 1
            self._requests[ip] = timestamps
            return False, retry_after

        timestamps.append(now)
        self._requests[ip] = timestamps
        return True, 0

    # -- Redis backend -----------------------------------------------------

    async def _check_redis(self, ip: str, now: float) -> tuple[bool, int] | None:
        """Return ``(allowed, retry_after)`` or ``None`` on Redis failure."""
        sha = await self._ensure_lua_script()
        if sha is None:
            return None
        key = f"{self._key_prefix}{ip}"
        member = f"{now}"
        try:
            result = await self._redis.evalsha(
                sha, 1, key, str(now), str(self._window), str(self._max_rpm), member,
            )
            allowed = int(result[0])
            if allowed:
                return True, 0
            oldest_ts = float(result[1])
            retry_after = int(self._window - (now - oldest_ts)) + 1
            return False, max(retry_after, 1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RateLimitMiddleware: Redis error ({}), falling back to memory", exc)
            self._lua_sha = None
            return None

    # -- Dispatch ----------------------------------------------------------

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in _AUTH_EXEMPT_PATHS:
            return await call_next(request)

        ip = self._client_ip(request)
        now = time.time()

        result = None
        if self._redis is not None:
            result = await self._check_redis(ip, now)

        if result is None:
            result = self._check_memory(ip, now)

        allowed, retry_after = result
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded ({self._max_rpm} req/min). Try again later.",
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)
