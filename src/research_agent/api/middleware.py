"""API 认证、超时与限流中间件。

认证
----
基于 ``API_SECRET_KEY`` 环境变量的 Bearer-token 校验。当该值为空（开发环境默认），认证禁用 —— 所有请求直接放行。
这样在保留生产环境仅需翻转一个环境变量即可锁定访问的同时，不会破坏本地``curl`` / UI 开发流程。

免认证端点：
- ``GET /health`` —— 必须对编排器无条件可达。
- ``GET /docs``、``GET /openapi.json`` —— 便于开发的 Swagger UI。

限流
----
基于客户端 IP 的滑动窗口计数器，提供两种后端：

1. Redis（分布式）—— 当提供 ``redis_url`` 且服务器可达时启用。使用有序集合 + Lua 脚本实现原子化的按 Key 计数，支持负载均衡器后多实例协同。
2. 内存（兜底）—— 以客户端 IP 为键的 ``dict``，每个键最多存储数十个浮点时间戳。周期性清扫会移除窗口外的过期键，防止客户端IP 扫描导致内存膨胀。

后端选择在启动时一次性完成；若 Redis 在启动之后变得不可达，每个失败的请求会透明降级到内存计数并记录警告（不返回 500）。

窗口大小为 60 秒，上限为 ``RATE_LIMIT_RPM``（默认 30 次/分钟）。
超限返回 429 并附带 ``Retry-After`` 头。

请求 ID 追踪
-------------
``RequestIdMiddleware`` 为每个请求生成（或传播）唯一的追踪 ID。
该 ID 被注入 loguru 上下文，使请求期间产生的日志行可关联，并通过``X-Request-ID`` 响应头回传给客户端。

HTTP 请求超时（可选）
---------------------
当配置 ``http_request_timeout_seconds > 0`` 时，``RequestTimeoutMiddleware`` 使用 ``asyncio.wait_for`` 包装下游处理器。
长 SSE 流（``/api/supervisor/research/stream``）被排除在外。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import Request, Response
    from starlette.types import ASGIApp

_AUTH_EXEMPT_PATHS = frozenset({
    "/",
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
    """拒绝未携带有效 Bearer token 的请求。

    当 ``secret_key`` 为空时禁用（直接放行）。
    """

    def __init__(self, app: ASGIApp, *, secret_key: str = "") -> None:
        super().__init__(app)
        self._secret_key = secret_key

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self._secret_key:
            return await call_next(request)

        path = request.url.path
        if path in _AUTH_EXEMPT_PATHS or path.startswith("/static"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if auth_header == f"Bearer {self._secret_key}":
            return await call_next(request)

        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key. Provide 'Authorization: Bearer <key>'."},
        )


class RequestIdMiddleware(BaseHTTPMiddleware):
    """生成或传播唯一的请求追踪 ID。

    若传入请求携带 ``X-Request-ID``（由反向代理或客户端设置），则复用该值；否则生成一个 UUID-4。解析后的 ID 将：

    * 存储在 ``request.state.request_id`` 供下游处理器使用；
    * 绑定到 loguru 上下文（``extra["request_id"]``），使请求期间产生的每条日志行均携带该追踪 ID；
    * 通过 ``X-Request-ID`` 响应头回传。
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        with logger.contextualize(request_id=request_id):
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """可选的 ASGI 处理器最大执行时长限制。"""

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
        except TimeoutError:
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
    """基于客户端 IP 的滑动窗口限流器。

    当提供 ``redis_url`` 时，使用 Redis 有序集合配合原子 Lua 脚本实现跨实例分布式计数。Redis 不可用时降级为内存计数。
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

        # 内存兜底（始终可用）
        self._requests: dict[str, list[float]] = {}
        self._memory_tick: int = 0

        # Redis 后端（尽力而为）
        self._redis = None
        self._lua_sha: str | None = None
        if redis_url:
            self._redis = self._try_connect_redis(redis_url)

    # -- Redis 引导 --------------------------------------------------------

    @staticmethod
    def _try_connect_redis(redis_url: str):
        """返回一个 ``redis.asyncio.Redis`` 实例，失败时返回 ``None``。"""
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
        """将 Lua 脚本加载到 Redis 中（通过 EVALSHA 缓存）。"""
        if self._lua_sha is not None:
            return self._lua_sha
        if self._redis is None:
            return None
        try:
            self._lua_sha = await self._redis.script_load(_SLIDING_WINDOW_LUA)
            return self._lua_sha
        except Exception:  # noqa: BLE001
            return None

    # -- IP 提取 ------------------------------------------------------------

    @staticmethod
    def _client_ip(request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    # -- 内存后端 -----------------------------------------------------------

    def _reap_stale_memory_keys(self, now: float) -> None:
        """移除时间戳已完全滑出窗口的 IP 记录。"""
        cutoff = now - self._window
        for ip_key in list(self._requests.keys()):
            ts = self._requests[ip_key]
            while ts and ts[0] < cutoff:
                ts.pop(0)
            if not ts:
                self._requests.pop(ip_key, None)

    def _tick_memory_maintenance(self, now: float) -> None:
        """基于请求计数器的有限代价周期性清扫。"""
        self._memory_tick += 1
        if self._memory_tick >= 4096:
            self._memory_tick = 0
            self._reap_stale_memory_keys(now)

    def _check_memory(self, ip: str, now: float) -> tuple[bool, int]:
        """返回 ``(是否放行, retry_after 秒数)``。"""
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

    # -- Redis 后端 ---------------------------------------------------------

    async def _check_redis(self, ip: str, now: float) -> tuple[bool, int] | None:
        """返回 ``(是否放行, retry_after)``，Redis 失败时返回 ``None``。"""
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

    # -- 请求分发 -----------------------------------------------------------

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
