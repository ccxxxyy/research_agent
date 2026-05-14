"""Unit tests — config.py model defaults and LLMConfig shape."""

from __future__ import annotations

from research_agent.config import LLMConfig


def _make_llm_cfg() -> LLMConfig:
    """Hermetic LLMConfig so the developer's `.env` does not leak in."""
    return LLMConfig(
        _env_file=None,  # type: ignore[arg-type]
        openai_api_key="x",
        deepseek_api_key="x",
    )


class TestLLMConfigDefaults:
    def test_default_light_model(self) -> None:
        cfg = _make_llm_cfg()
        assert cfg.light_model == "qwen3-max-2026-01-23"

    def test_default_medium_model(self) -> None:
        cfg = _make_llm_cfg()
        assert cfg.medium_model == "qwen3.6-plus"

    def test_default_heavy_model(self) -> None:
        cfg = _make_llm_cfg()
        assert cfg.heavy_model == "deepseek-v4-pro"

    def test_three_tier_models_all_present(self) -> None:
        cfg = _make_llm_cfg()
        for attr in ("light_model", "medium_model", "heavy_model"):
            val = getattr(cfg, attr)
            assert isinstance(val, str) and len(val) > 0, f"{attr} is empty"

    def test_dashscope_api_base_defaults(self) -> None:
        cfg = _make_llm_cfg()
        assert "deepseek" in cfg.deepseek_api_base or "dashscope" in cfg.deepseek_api_base
