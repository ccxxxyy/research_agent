"""Per-user Token 配额管理。

为每个用户维护滑动窗口内的 Token 消耗计数器，防止单一用户耗尽共享 LLM 预算。

两种后端：
1. Redis（分布式）—— 利用 INCRBY + TTL 实现原子化累加，支持多实例协同。
2. 内存（兜底）—— 单进程 dict，适合开发环境。

配额窗口默认为 24 小时（86400 秒），每用户上限由 ``USER_TOKEN_QUOTA_DAILY`` 环境变量控制（默认 500,000 token）。
超额时返回 429 并附带 ``Retry-After`` 头。

匿名用户（``user_id="anonymous"``）共享同一配额池。若要区分匿名用户，上层可结合 IP + fingerprint 生成唯一标识。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class _UserBucket:
    """单用户的配额桶。"""

    tokens_used: int = 0
    window_start: float = field(default_factory=time.time)


class TokenQuotaManager:
    """Per-user Token 配额检查器。

    Usage::

        quota = TokenQuotaManager(daily_limit=500_000)
        ok, remaining = quota.check_and_consume("alice", 1200)
        if not ok:
            return HTTP_429(...)
    """

    def __init__(
        self,
        *,
        daily_limit: int = 500_000,
        window_seconds: float = 86400.0,
        redis_client=None,
    ) -> None:
        self._daily_limit = daily_limit
        self._window = window_seconds
        self._redis = redis_client
        self._redis_prefix = "tq:daily:"

        self._lock = threading.Lock()
        self._buckets: dict[str, _UserBucket] = {}

    @property
    def daily_limit(self) -> int:
        return self._daily_limit

    # -- 公共 API -----------------------------------------------------------

    def check_and_consume(
        self,
        user_id: str,
        token_count: int,
    ) -> tuple[bool, int]:
        """检查配额并消耗 token。

        返回 ``(是否放行, 剩余配额)``。
        若 token_count 超出剩余额度则不消耗，返回 ``(False, remaining)``。
        """
        if self._daily_limit <= 0:
            return True, self._daily_limit

        if self._redis is not None:
            result = self._check_redis(user_id, token_count)
            if result is not None:
                return result

        return self._check_memory(user_id, token_count)

    def get_usage(self, user_id: str) -> tuple[int, int]:
        """返回 ``(已用 token, 配额上限)``。"""
        if self._redis is not None:
            usage = self._get_redis_usage(user_id)
            if usage is not None:
                return usage, self._daily_limit

        with self._lock:
            bucket = self._buckets.get(user_id)
            if bucket is None:
                return 0, self._daily_limit
            now = time.time()
            if now - bucket.window_start >= self._window:
                return 0, self._daily_limit
            return bucket.tokens_used, self._daily_limit

    # -- 内存后端 -----------------------------------------------------------

    def _check_memory(self, user_id: str, token_count: int) -> tuple[bool, int]:
        now = time.time()
        with self._lock:
            bucket = self._buckets.get(user_id)
            if bucket is None or now - bucket.window_start >= self._window:
                bucket = _UserBucket(tokens_used=0, window_start=now)
                self._buckets[user_id] = bucket

            remaining = self._daily_limit - bucket.tokens_used
            if token_count > remaining:
                return False, remaining

            bucket.tokens_used += token_count
            return True, remaining - token_count

    # -- Redis 后端 ---------------------------------------------------------

    def _check_redis(self, user_id: str, token_count: int) -> tuple[bool, int] | None:
        key = f"{self._redis_prefix}{user_id}"
        try:
            pipe = self._redis.pipeline(transaction=True)
            pipe.get(key)
            pipe.ttl(key)
            current_raw, ttl = pipe.execute()

            current = int(current_raw or 0)
            remaining = self._daily_limit - current

            if token_count > remaining:
                return False, remaining

            pipe2 = self._redis.pipeline(transaction=True)
            pipe2.incrby(key, token_count)
            if ttl is None or ttl < 0:
                pipe2.expire(key, int(self._window))
            pipe2.execute()

            return True, remaining - token_count
        except Exception as exc:  # noqa: BLE001
            logger.warning("TokenQuota: Redis error ({}), falling back to memory", exc)
            return None

    def _get_redis_usage(self, user_id: str) -> int | None:
        key = f"{self._redis_prefix}{user_id}"
        try:
            val = self._redis.get(key)
            return int(val or 0)
        except Exception:  # noqa: BLE001
            return None
