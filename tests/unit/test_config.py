"""单元测试 — config.py 默认值与模型名称。

这些测试断言 ``LLMConfig`` 中的 代码级默认值。如果仓库根目录下存在开发者的 ``.env`` 文件，
pydantic-settings 会在实例化时读取它，从而悄悄覆盖断言所期望的值（例如运维人员将 ``DEEPSEEK_API_BASE`` 指向Dashscope 的 OpenAI 兼容端点）。
因此在构造每个配置时传入``_env_file=None``，使测试保持隔离，精确断言源码中声明的默认值。
"""

from __future__ import annotations

from research_agent.config import LLMConfig, Settings


def _make_cfg() -> LLMConfig:
    """构造一个不读取本地 .env 的 LLMConfig。

    pydantic-settings 在实例化时读取 ``env_file`` 设置；传入``_env_file=None`` 可跳过该步骤。仍注入两个 API 密钥， 以便值校验器（如果存在）能正常通过。
    """
    return LLMConfig(
        _env_file=None,  # type: ignore[arg-type]
        openai_api_key="x",
        deepseek_api_key="x",
    )


class TestLLMConfigDefaults:
    def test_light_model_default(self) -> None:
        # config.py 中的权威默认值 — 通过环境变量的运维覆盖在其他测试中单独验证。
        assert _make_cfg().light_model == "deepseek-v4-flash"

    def test_medium_model_default(self) -> None:
        assert _make_cfg().medium_model == "qwen3.6-plus"

    def test_heavy_model_default(self) -> None:
        assert _make_cfg().heavy_model == "deepseek-v4-pro"

    def test_deepseek_api_base_default(self) -> None:
        assert "deepseek.com" in _make_cfg().deepseek_api_base

    def test_three_distinct_tiers(self) -> None:
        """LIGHT / MEDIUM / HEAVY 必须解析为不同的模型 —— 如果错误配置导致三者相同，分层路由就失去了意义。"""
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

    def test_checkpoint_sqlite_path_default(self) -> None:
        s = Settings(_env_file=None)  # type: ignore[arg-type]
        assert s.checkpoint_sqlite_path == "data/langgraph_checkpoint.db"
