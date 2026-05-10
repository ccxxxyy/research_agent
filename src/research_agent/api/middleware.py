"""API authentication and rate-limiting middleware.

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
In-memory sliding-window counter keyed by client IP. No external
dependency (Redis not required). Sufficient for single-instance
deployments; for horizontal scaling, swap to a Redis-backed counter
or an API gateway (nginx, Cloudflare, etc.).

The window is 60 seconds; the cap is ``RATE_LIMIT_RPM`` (default 30
requests/minute). Exceeding the limit returns 429 with a
``Retry-After`` header.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

_AUTH_EXEMPT_PATHS = frozenset({
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
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


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory sliding-window rate limiter keyed by client IP.

    Returns 429 when the per-IP request count exceeds ``max_rpm``
    within a 60-second window.
    """

    def __init__(self, app: ASGIApp, *, max_rpm: int = 30) -> None:
        super().__init__(app)
        self._max_rpm = max_rpm
        self._window = 60.0
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _cleanup(self, timestamps: list[float], now: float) -> list[float]:
        cutoff = now - self._window
        while timestamps and timestamps[0] < cutoff:
            timestamps.pop(0)
        return timestamps

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in _AUTH_EXEMPT_PATHS:
            return await call_next(request)

        ip = self._client_ip(request)
        now = time.time()
        timestamps = self._cleanup(self._requests[ip], now)

        if len(timestamps) >= self._max_rpm:
            retry_after = int(self._window - (now - timestamps[0])) + 1
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded ({self._max_rpm} req/min). Try again later."},
                headers={"Retry-After": str(retry_after)},
            )

        timestamps.append(now)
        self._requests[ip] = timestamps
        return await call_next(request)
