"""Centralized configuration via pydantic-settings."""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import Field
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

    default_recursion_limit: int = Field(
        default=50,
        ge=10,
        le=150,
        description=(
            "Default LangGraph recursion limit applied when the client "
            "does not specify one. The 6-specialist research supervisor "
            "needs ~4 graph steps per specialist hand-off (transfer + "
            "tool-call + tool-result + transfer-back), plus supervisor "
            "planning and optional reflection (up to 5 extra steps). "
            "25 (LangGraph's built-in default) is too low for complex "
            "multi-specialist queries; 50 covers 6 specialists + "
            "reflection with headroom."
        ),
    )

    sse_research_heartbeat_seconds: float = Field(
        default=15.0,
        ge=0,
        le=86400,
        description=(
            "Interval between SSE comment-free keep-alive DATA frames "
            "on ``/api/supervisor/research/stream`` while the graph "
            "is idle — keeps reverse proxies / CDNs from closing long "
            "requests. Zero disables heartbeat."
        ),
    )

    checkpoint_sqlite_path: str = Field(
        default="data/langgraph_checkpoint.db",
        description=(
            "When Postgres is unreachable at startup, LangGraph writes "
            "checkpoints to this SQLite file (parent dirs are created). "
            "Set to empty string to skip SQLite and fall back to "
            "in-memory checkpoints only."
        ),
    )

    mcp_tool_discovery_timeout: float = Field(
        default=30.0,
        ge=5,
        le=300,
        description=(
            "Timeout in seconds for each MCP tool-discovery call at "
            "startup. If a subprocess takes longer than this to "
            "enumerate its tools, that specialist is skipped."
        ),
    )

    memory_store_sqlite_path: str = Field(
        default="data/langgraph_memory_store.db",
        description=(
            "When Postgres is unreachable at startup, long-term memory "
            "(user preferences, research history) is persisted to this "
            "SQLite file via AsyncSqliteStore. Set to empty string to "
            "skip SQLite and fall back to InMemoryStore (non-persistent)."
        ),
    )

    llm: LLMConfig = LLMConfig()
    database: DatabaseConfig = DatabaseConfig()
    observability: ObservabilityConfig = ObservabilityConfig()

    @property
    def is_dev(self) -> bool:
        return self.app_env == Environment.DEVELOPMENT


@lru_cache
def get_settings() -> Settings:
    return Settings()
