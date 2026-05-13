"""Multi-model router with automatic fallback and usage tracking."""

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
    """Routes LLM calls to appropriate models based on task complexity.

    Supports three-tier routing (light/medium/heavy) with per-tier
    provider configuration, automatic fallback when a model is
    unavailable, and per-agent/per-tier usage tracking.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._usage = UsageTracker()
        self._registry: dict[ModelTier, ChatOpenAI] = self._build_registry()

    def _resolve_credentials(self, tier: ModelTier) -> tuple[str, str]:
        """Resolve API key and base URL for a tier.

        Priority (per tier):
          1. Tier-specific override (e.g. LIGHT_API_KEY / LIGHT_API_BASE)
          2. DashScope credentials (for Qwen models)
          3. DeepSeek credentials
          4. OpenAI credentials
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
            registry[tier] = ChatOpenAI(
                model=model_name,
                api_key=SecretStr(api_key),
                base_url=base_url,
                temperature=tier_temps[tier],
                max_retries=2,
                callbacks=[handler],
            )

        return registry

    def get_model(self, tier: ModelTier) -> Runnable:
        """Get a model by tier, with automatic fallback chain attached."""
        primary = self._registry[tier]
        if tier in FALLBACK_CHAIN:
            fallback_tier = FALLBACK_CHAIN[tier]
            return primary.with_fallbacks([self._registry[fallback_tier]])
        return primary

    def for_agent(self, agent: AgentName | str) -> Runnable:
        """Get the recommended model for a specific agent role."""
        resolved: AgentName = agent if isinstance(agent, AgentName) else AgentName(agent)
        tier = AGENT_TIER_MAP.get(resolved, ModelTier.MEDIUM)
        return self.get_model(tier)

    @property
    def usage(self) -> UsageTracker:
        return self._usage
