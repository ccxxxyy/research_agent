"""多模型路由器 — 自动降级、熔断器与用量追踪。"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from research_agent.config import LLMConfig
from research_agent.llm.tier import (
    AGENT_TIER_MAP,
    FALLBACK_CHAIN,
    AgentName,
    ModelTier,
)
from research_agent.llm.usage_tracker import UsageCallbackHandler, UsageTracker


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitState(str, Enum):
    """熔断器三态。"""

    CLOSED = "closed"      # 正常：请求正常通过
    OPEN = "open"          # 熔断：直接拒绝，走 fallback
    HALF_OPEN = "half_open"  # 半开：允许一次试探


class CircuitBreaker:
    """简单的熔断器实现。

    - failure_threshold: 连续失败多少次触发熔断（默认 3）
    - recovery_timeout: 熔断后多少秒进入半开状态（默认 30）
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def allow_request(self) -> bool:
        """判断当前是否允许请求通过。"""
        current = self.state
        return current in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self) -> None:
        """记录一次成功调用，重置计数。"""
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """记录一次失败调用。"""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self._failure_threshold:
            self._state = CircuitState.OPEN


class CircuitBreakerRunnable(Runnable):
    """为 LLM Runnable 包装熔断器逻辑。

    当主模型连续失败达到阈值后，直接跳过主模型调用，立即抛出异常让 LangChain 的 with_fallbacks 机制接管。

    透明代理：所有未在本类定义的属性/方法（如 ``bind_tools``、 ``with_structured_output`` 等）自动委托给内部的 ChatOpenAI 实例，
    确保与 LangGraph 的 ``create_react_agent`` 等高层 API 兼容。
    """

    def __init__(
        self,
        wrapped: ChatOpenAI,
        breaker: CircuitBreaker,
    ) -> None:
        self._wrapped = wrapped
        self._breaker = breaker

    def __getattr__(self, name: str) -> Any:
        """透明代理：将未在本类定义的属性访问委托给底层模型。"""
        return getattr(self._wrapped, name)

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    @property
    def InputType(self) -> type[Input]:  # noqa: N802
        return self._wrapped.InputType

    @property
    def OutputType(self) -> type[Output]:  # noqa: N802
        return self._wrapped.OutputType

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        if not self._breaker.allow_request():
            raise RuntimeError(
                f"Circuit breaker OPEN for model: requests blocked until recovery"
            )
        try:
            result = self._wrapped.invoke(input, config=config, **kwargs)
            self._breaker.record_success()
            return result
        except Exception:
            self._breaker.record_failure()
            raise

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        if not self._breaker.allow_request():
            raise RuntimeError(
                f"Circuit breaker OPEN for model: requests blocked until recovery"
            )
        try:
            result = await self._wrapped.ainvoke(input, config=config, **kwargs)
            self._breaker.record_success()
            return result
        except Exception:
            self._breaker.record_failure()
            raise

    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        return self._wrapped.get_name(suffix, name=name)


# ---------------------------------------------------------------------------
# Model Router 模型路由器
# ---------------------------------------------------------------------------


class ModelRouter:
    """根据任务复杂度将 LLM 调用路由到合适的模型。

    支持三级路由（light/medium/heavy），每层可独立配置提供商，
    模型不可用时自动降级，并按 Agent/层级追踪用量。
    每个模型层级配备独立的熔断器：连续失败 N 次后自动跳过，直接走 fallback 模型，避免不可用的提供商拖累整体延迟。
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._usage = UsageTracker()
        self._breakers: dict[ModelTier, CircuitBreaker] = {
            tier: CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
            for tier in ModelTier
        }
        self._registry: dict[ModelTier, ChatOpenAI] = self._build_registry()

    def _resolve_credentials(self, tier: ModelTier) -> tuple[str, str]:
        """解析指定层级的 API key 和 base URL。

        优先级（按层级）：
          1. 层级专属覆盖（如 LIGHT_API_KEY / LIGHT_API_BASE）
          2. DashScope 凭据（Qwen 模型）
          3. DeepSeek 凭据
          4. OpenAI 凭据
        """
        cfg = self._config

        tier_key = getattr(cfg, f"{tier.value}_api_key", "").strip()
        tier_base = getattr(cfg, f"{tier.value}_api_base", "").strip()

        if tier_key and tier_base:
            return tier_key, tier_base

        fallback_key = (
            cfg.dashscope_api_key or cfg.deepseek_api_key or cfg.openai_api_key
        )
        fallback_base = (
            cfg.dashscope_api_base or cfg.deepseek_api_base or cfg.openai_api_base
        )

        return tier_key or fallback_key, tier_base or fallback_base

    def _build_registry(self) -> dict[ModelTier, ChatOpenAI]:
        registry: dict[ModelTier, ChatOpenAI] = {}

        tier_temps = {
            ModelTier.LIGHT: 0.1,
            ModelTier.MEDIUM: 0.3,
            ModelTier.HEAVY: 0.7,
        }

        for tier in ModelTier:
            model_name = getattr(self._config, f"{tier.value}_model")
            api_key, base_url = self._resolve_credentials(tier)
            handler = UsageCallbackHandler(self._usage, tier_label=tier.value)
            rt = getattr(self._config, "request_timeout_seconds", None)
            chat_kw: dict = {
                "model": model_name,
                "api_key": SecretStr(api_key),
                "base_url": base_url,
                "temperature": tier_temps[tier],
                "max_retries": 2,
                "callbacks": [handler],
            }
            if rt is not None and float(rt) > 0:
                chat_kw["request_timeout"] = float(rt)
            registry[tier] = ChatOpenAI(**chat_kw)

        return registry

    def get_model(self, tier: ModelTier) -> Runnable:
        """按层级获取模型，附带熔断器 + 自动降级链。"""
        primary = CircuitBreakerRunnable(
            self._registry[tier], self._breakers[tier]
        )
        if tier in FALLBACK_CHAIN:
            fallback_tier = FALLBACK_CHAIN[tier]
            fallback = CircuitBreakerRunnable(
                self._registry[fallback_tier], self._breakers[fallback_tier]
            )
            return primary.with_fallbacks([fallback])
        return primary

    def for_agent(self, agent: AgentName | str) -> Runnable:
        """获取特定 Agent 角色推荐使用的模型。"""
        resolved: AgentName = agent if isinstance(agent, AgentName) else AgentName(agent)
        tier = AGENT_TIER_MAP.get(resolved, ModelTier.MEDIUM)
        return self.get_model(tier)

    def get_breaker(self, tier: ModelTier) -> CircuitBreaker:
        """获取指定层级的熔断器（用于监控/测试）。"""
        return self._breakers[tier]

    @property
    def usage(self) -> UsageTracker:
        return self._usage
