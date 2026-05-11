"""Centralized configuration via pydantic-settings."""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"
    deepseek_api_key: str = ""
    deepseek_api_base: str = "https://api.deepseek.com/v1"
    dashscope_api_key: str = ""
    dashscope_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    light_model: str = "qwen3-max-2026-01-23"
    light_api_key: str = ""
    light_api_base: str = ""

    medium_model: str = "qwen3.6-plus"
    medium_api_key: str = ""
    medium_api_base: str = ""

    heavy_model: str = "deepseek-v4-pro"
    heavy_api_key: str = ""
    heavy_api_base: str = ""


class DatabaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    postgres_uri: str = "postgresql+asyncpg://research:research@localhost:5432/research_agent"
    postgres_sync_uri: str = "postgresql://research:research@localhost:5432/research_agent"
    redis_url: str = "redis://localhost:6379/0"


class ObservabilityConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    langsmith_api_key: str = ""
    langsmith_project: str = "research-agent"
    langchain_tracing_v2: bool = False
    log_level: str = "INFO"


class Settings(BaseSettings):
    """Root settings aggregating all sub-configs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Environment = Environment.DEVELOPMENT
    app_host: str = "0.0.0.0"
    app_port: int = 8080
    cors_origins: str = "*"
    api_secret_key: str = ""
    rate_limit_rpm: int = 30

    reflection_enabled: bool = False
    """When True, the research supervisor wraps its final synthesis in a
    critic+writer reflection loop. Default OFF because reflection adds
    1–3 extra LLM calls per request and most demo flows don't need it;
    flip to True in production for higher answer quality."""

    reflection_pass_threshold: float = 0.85
    """Critic score at or above which the reflection loop terminates
    early. 0.85 maps to the critic prompt's "ship after a light
    rewrite" band."""

    reflection_max_iterations: int = 2
    """Hard cap on writer rewrites. Worst-case LLM budget per request
    is ``max_iterations + 1`` critic calls plus ``max_iterations``
    writer calls."""

    llm: LLMConfig = LLMConfig()
    database: DatabaseConfig = DatabaseConfig()
    observability: ObservabilityConfig = ObservabilityConfig()

    @property
    def is_dev(self) -> bool:
        return self.app_env == Environment.DEVELOPMENT


@lru_cache
def get_settings() -> Settings:
    return Settings()
