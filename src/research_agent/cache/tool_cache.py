"""MCP 工具原始返回值的分层 TTL 缓存。

为什么需要
----------
金融研究系统不应缓存 LLM 最终回答——行情 / 新闻 / 资金流持续变化。
可以缓存的是上游工具的原始负载：同一 MCP 子进程里，一次 supervisor回合（以及同进程内并发回合）经常会重复调用同一个``get_index_quotes()``。

设计
----
* **默认进程内内存** — MCP server 以 stdio 子进程运行；进程本地 dict是唯一无需额外接线就能工作的后端。
可选 Redis （``TOOL_CACHE_BACKEND=redis``）在 Redis 可用时跨 worker 共享命中。
* **TTL 分档** — realtime / short / medium / daily / long，对应各类数据实际变化频率。
* **错误永不入缓存** — ``{"error": ...}`` 始终重新打上游，避免短暂故障污染缓存。
* **装饰器顺序** — ``@mcp.tool()`` 在外、``@cached_tool(...)`` 在内，以便 FastMCP 注册的是带缓存的协程。

环境变量（首次使用时读取一次）
------------------------------
* ``TOOL_CACHE_ENABLED`` — ``1``/``true``（默认）或 ``0``/``false``。
* ``TOOL_CACHE_BACKEND`` — ``memory``（默认）或 ``redis``。
* ``REDIS_URL`` — backend 为 ``redis`` 时使用；连接失败则静默回退内存。
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import time
from copy import deepcopy
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

P = ParamSpec("P")
R = TypeVar("R")

# ---------------------------------------------------------------------
# TTL 分档（秒）
# ---------------------------------------------------------------------
TTL_REALTIME = 20
"""行情 / 分时 / 涨跌榜 — 秒级新鲜度。"""

TTL_SHORT = 120
"""新闻 / 快讯 / 热搜词 — 1～2 分钟。"""

TTL_MEDIUM = 300
"""资金流 / 龙虎榜 / 板块列表 — 数分钟。"""

TTL_DAILY = 3600
"""日 K / 财务 / 基金净值 — 同一会话内小时级。"""

TTL_LONG = 21_600
"""公司概况 / 搜索 / 评级 / 持仓 — 数小时。"""


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _make_key(namespace: str, name: str, bound_args: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(bound_args), sort_keys=True, default=str, ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"toolcache:{namespace}:{name}:{digest}"


def _is_error_payload(result: Any) -> bool:
    return isinstance(result, dict) and "error" in result


# ---------------------------------------------------------------------
# 后端
# ---------------------------------------------------------------------
class _MemoryBackend:
    """进程本地 dict，按 key 过期；异步锁保护。"""

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at <= time.monotonic():
                self._store.pop(key, None)
                return None
            return deepcopy(value)

    async def set(self, key: str, value: Any, ttl: float) -> None:
        async with self._lock:
            self._store[key] = (time.monotonic() + ttl, deepcopy(value))

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    def size(self) -> int:
        return len(self._store)


class _RedisBackend:
    """轻量异步 Redis 封装；调用方在失败时回退到内存。"""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def get(self, key: str) -> Any | None:
        raw = await self._client.get(key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    async def set(self, key: str, value: Any, ttl: float) -> None:
        payload = json.dumps(value, ensure_ascii=False, default=str)
        await self._client.set(key, payload, ex=max(1, int(ttl)))

    async def clear(self) -> None:
        # 只清理本命名空间，绝不 FLUSHDB。
        cursor = 0
        while True:
            cursor, keys = await self._client.scan(cursor, match="toolcache:*", count=200)
            if keys:
                await self._client.delete(*keys)
            if cursor == 0:
                break


# ---------------------------------------------------------------------
# 门面
# ---------------------------------------------------------------------
class ToolResultCache:
    """供 :func:`cached_tool` 使用的共享缓存门面。"""

    def __init__(self, backend: _MemoryBackend | _RedisBackend, *, enabled: bool) -> None:
        self._backend = backend
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self.skips = 0  # 错误载荷 / 已禁用时跳过写入

    async def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        try:
            value = await self._backend.get(key)
        except Exception as exc:  # noqa: BLE001 — 绝不能打断工具主路径
            logger.warning("tool_cache 读取失败 ({}): {}", key, exc)
            return None
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        return value

    async def set(self, key: str, value: Any, ttl: float) -> None:
        if not self.enabled or ttl <= 0 or _is_error_payload(value):
            self.skips += 1
            return
        try:
            await self._backend.set(key, value, ttl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("tool_cache 写入失败 ({}): {}", key, exc)

    async def clear(self) -> None:
        await self._backend.clear()
        self.hits = self.misses = self.skips = 0

    def stats(self) -> dict[str, int | bool]:
        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "skips": self.skips,
        }


_cache: ToolResultCache | None = None


def _build_backend() -> _MemoryBackend | _RedisBackend:
    backend_name = os.getenv("TOOL_CACHE_BACKEND", "memory").strip().lower()
    if backend_name == "redis":
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            from redis.asyncio import Redis

            client = Redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=1.0)
            # 此处只构造客户端；真正连通性在首次 get/set 时验证，
            # 失败由门面捕获并记日志，不阻断工具调用。
            logger.info("tool_cache: 已配置 Redis 后端 ({})", redis_url)
            return _RedisBackend(client)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "tool_cache: Redis 不可用 ({}), 回退到内存后端",
                exc,
            )
    return _MemoryBackend()


def get_tool_cache() -> ToolResultCache:
    """惰性单例 — 可从 MCP 子进程安全调用。"""
    global _cache
    if _cache is None:
        enabled = _env_flag("TOOL_CACHE_ENABLED", default=True)
        _cache = ToolResultCache(_build_backend(), enabled=enabled)
        logger.debug(
            "tool_cache 已就绪 (enabled={}, backend={})",
            enabled,
            type(_cache._backend).__name__,
        )
    return _cache


def reset_tool_cache_for_tests() -> None:
    """丢弃单例，便于测试重建干净后端。"""
    global _cache
    _cache = None


# ---------------------------------------------------------------------
# 装饰器
# ---------------------------------------------------------------------
def cached_tool(
    ttl: int,
    *,
    namespace: str,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """将工具成功返回的 dict 缓存 ``ttl`` 秒。

    用法::

        @mcp.tool()
        @cached_tool(ttl=TTL_REALTIME, namespace="fin")
        async def get_index_quotes() -> dict:
            ...

    Args:
        ttl: 存活秒数。请使用 ``TTL_*`` 常量。
        namespace: 短服务器前缀（``fin`` / ``fund`` / ``news``），避免不同 MCP server 的 key 互相碰撞。
    """

    def decorator(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            cache = get_tool_cache()
            # 将位置参数与关键字参数绑定成稳定映射，用于生成缓存键。
            # MCP/LangChain 通常以 kwargs 调用；单元测试可能传位置参数——两者都合并。
            bound = inspect.signature(fn).bind_partial(*args, **kwargs)
            bound.apply_defaults()
            key = _make_key(namespace, fn.__name__, bound.arguments)

            cached = await cache.get(key)
            if cached is not None:
                logger.debug("tool_cache 命中 {}.{}", namespace, fn.__name__)
                return cached  # type: ignore[return-value]

            result = await fn(*args, **kwargs)
            await cache.set(key, result, float(ttl))
            return result

        # 保留原函数，便于测试 / 内省。
        wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
        return wrapper

    return decorator
