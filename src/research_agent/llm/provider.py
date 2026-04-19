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
from research_agent.llm.usage_tracker import UsageTracker


class ModelRouter:
    """Routes LLM calls to appropriate models based on task complexity.

    Supports three-tier routing (light/medium/heavy), automatic fallback
    when a model is unavailable, and per-agent/per-tier usage tracking.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._usage = UsageTracker()
        self._registry: dict[ModelTier, ChatOpenAI] = self._build_registry()

    def _build_registry(self) -> dict[ModelTier, ChatOpenAI]:
        # Use the primary available key for all tiers.
        # DashScope / DeepSeek / OpenAI all use OpenAI-compatible format.
        api_key = self._config.deepseek_api_key or self._config.openai_api_key
        base_url = self._config.deepseek_api_base or self._config.openai_api_base

        return {
            ModelTier.LIGHT: ChatOpenAI(
                model=self._config.light_model,
                api_key=SecretStr(api_key),
                base_url=base_url,
                temperature=0.1,
                max_retries=2,
            ),
            ModelTier.MEDIUM: ChatOpenAI(
                model=self._config.medium_model,
                api_key=SecretStr(api_key),
                base_url=base_url,
                temperature=0.3,
                max_retries=2,
            ),
            ModelTier.HEAVY: ChatOpenAI(
                model=self._config.heavy_model,
                api_key=SecretStr(api_key),
                base_url=base_url,
                temperature=0.7,
                max_retries=2,
            ),
        }

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
