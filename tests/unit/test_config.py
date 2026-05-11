"""Unit tests — config.py defaults and model names.

These tests assert the *code-level defaults* in ``LLMConfig``. A
developer's ``.env`` file at the repo root would otherwise leak in
through pydantic-settings and silently invalidate the assertions
(e.g. an operator pointing ``DEEPSEEK_API_BASE`` at a Dashscope
OpenAI-compatible endpoint). We therefore construct each config with
``_env_file=None`` to make the test hermetic and assert exactly what
the source declares.
"""

from __future__ import annotations

from research_agent.config import LLMConfig, Settings


def _make_cfg() -> LLMConfig:
    """Construct an LLMConfig that does NOT read from the local .env.

    pydantic-settings reads its ``env_file`` setting at instantiation
    time; passing ``_env_file=None`` short-circuits that. We still
    inject the two API keys so the value validators (if any) pass.
    """
    return LLMConfig(
        _env_file=None,  # type: ignore[arg-type]
        openai_api_key="x",
        deepseek_api_key="x",
    )


class TestLLMConfigDefaults:
    def test_light_model_default(self) -> None:
        # Source-of-truth default in config.py — operator overrides
        # via env vars are tested separately.
        assert _make_cfg().light_model == "qwen3-max-2026-01-23"

    def test_medium_model_default(self) -> None:
        assert _make_cfg().medium_model == "qwen3.6-plus"

    def test_heavy_model_default(self) -> None:
        assert _make_cfg().heavy_model == "deepseek-v4-pro"

    def test_deepseek_api_base_default(self) -> None:
        assert "deepseek.com" in _make_cfg().deepseek_api_base

    def test_three_distinct_tiers(self) -> None:
        """LIGHT / MEDIUM / HEAVY must all resolve to different models —
        a misconfiguration that flattens them would defeat the whole
        point of tiered routing."""
        cfg = _make_cfg()
        tiers = {cfg.light_model, cfg.medium_model, cfg.heavy_model}
        assert len(tiers) == 3


class TestSettingsRoot:
    def test_default_env_is_development(self) -> None:
        s = Settings(_env_file=None)  # type: ignore[arg-type]
        assert s.is_dev

    def test_sse_research_heartbeat_seconds_default(self) -> None:
        s = Settings(_env_file=None)  # type: ignore[arg-type]
        assert s.sse_research_heartbeat_seconds == 15.0
