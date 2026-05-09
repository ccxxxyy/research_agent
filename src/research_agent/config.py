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

    light_model: str = "qwen3.6-plus"
    medium_model: str = "deepseek-v4-pro"
    heavy_model: str = "deepseek-v4-pro"


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

    llm: LLMConfig = LLMConfig()
    database: DatabaseConfig = DatabaseConfig()
    observability: ObservabilityConfig = ObservabilityConfig()

    @property
    def is_dev(self) -> bool:
        return self.app_env == Environment.DEVELOPMENT


@lru_cache
def get_settings() -> Settings:
    return Settings()
