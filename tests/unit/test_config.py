"""Unit tests — config.py defaults and model names."""

from __future__ import annotations

from research_agent.config import LLMConfig, Settings


class TestLLMConfigDefaults:
    def test_light_model_default(self) -> None:
        cfg = LLMConfig(openai_api_key="x", deepseek_api_key="x")
        assert cfg.light_model == "qwen3.6-plus"

    def test_medium_model_default(self) -> None:
        cfg = LLMConfig(openai_api_key="x", deepseek_api_key="x")
        assert cfg.medium_model == "deepseek-v4-pro"

    def test_heavy_model_default(self) -> None:
        cfg = LLMConfig(openai_api_key="x", deepseek_api_key="x")
        assert cfg.heavy_model == "deepseek-v4-pro"

    def test_deepseek_api_base_default(self) -> None:
        cfg = LLMConfig(openai_api_key="x", deepseek_api_key="x")
        assert "deepseek.com" in cfg.deepseek_api_base

    def test_three_distinct_tiers(self) -> None:
        cfg = LLMConfig(openai_api_key="x", deepseek_api_key="x")
        assert cfg.light_model != cfg.medium_model


class TestSettingsRoot:
    def test_default_env_is_development(self) -> None:
        s = Settings()
        assert s.is_dev
