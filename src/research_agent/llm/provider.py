"""多模型路由器 — 自动降级与用量追踪。"""

from __future__ import annotations

from langchain_core.runnables import Runnable
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


class ModelRouter:
    """根据任务复杂度将 LLM 调用路由到合适的模型。

    支持三级路由（light/medium/heavy），每层可独立配置提供商，
    模型不可用时自动降级，并按 Agent/层级追踪用量。
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._usage = UsageTracker()
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
        """按层级获取模型，附带自动降级链。"""
        primary = self._registry[tier]
        if tier in FALLBACK_CHAIN:
            fallback_tier = FALLBACK_CHAIN[tier]
            return primary.with_fallbacks([self._registry[fallback_tier]])
        return primary

    def for_agent(self, agent: AgentName | str) -> Runnable:
        """获取特定 Agent 角色推荐使用的模型。"""
        resolved: AgentName = agent if isinstance(agent, AgentName) else AgentName(agent)
        tier = AGENT_TIER_MAP.get(resolved, ModelTier.MEDIUM)
        return self.get_model(tier)

    @property
    def usage(self) -> UsageTracker:
        return self._usage
