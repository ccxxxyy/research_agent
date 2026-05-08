"""Unit tests — config.py model defaults and LLMConfig shape."""

from __future__ import annotations

from research_agent.config import LLMConfig, get_settings


class TestLLMConfigDefaults:
    def test_default_light_model(self) -> None:
        cfg = LLMConfig(openai_api_key="x", deepseek_api_key="x")
        assert cfg.light_model == "qwen3.6-plus"

    def test_default_medium_model(self) -> None:
        cfg = LLMConfig(openai_api_key="x", deepseek_api_key="x")
        assert cfg.medium_model == "deepseek-v4-pro"

    def test_default_heavy_model(self) -> None:
        cfg = LLMConfig(openai_api_key="x", deepseek_api_key="x")
        assert cfg.heavy_model == "deepseek-v4-pro"

    def test_three_tier_models_all_present(self) -> None:
        cfg = LLMConfig(openai_api_key="x", deepseek_api_key="x")
        for attr in ("light_model", "medium_model", "heavy_model"):
            val = getattr(cfg, attr)
            assert isinstance(val, str) and len(val) > 0, f"{attr} is empty"

    def test_dashscope_api_base_defaults(self) -> None:
        cfg = LLMConfig(openai_api_key="x", deepseek_api_key="x")
        assert "deepseek" in cfg.deepseek_api_base or "dashscope" in cfg.deepseek_api_base
