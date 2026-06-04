"""基于 pydantic-settings 的集中式配置。"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    request_timeout_seconds: float = Field(
        default=180.0,
        ge=10,
        le=3600,
        description=(
            "每次调用 OpenAI 兼容聊天补全接口的 HTTP 超时时间（秒）。"
            "作为 ``request_timeout`` 传递给 LangChain ``ChatOpenAI``，防止停滞的提供商连接无限期占用工作线程。"
        ),
    )

    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"
    deepseek_api_key: str = ""
    deepseek_api_base: str = "https://api.deepseek.com/v1"
    dashscope_api_key: str = ""
    dashscope_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    light_model: str = "deepseek-v4-flash"
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
    log_file_path: str = Field(
        default="logs/research_agent.log",
        description=("滚动应用日志文件的路径。设为空字符串则仅输出日志到 stderr （不写入文件）。"),
    )


class Settings(BaseSettings):
    """聚合所有子配置的根设置类。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Environment = Environment.DEVELOPMENT
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    cors_origins: str = "*"
    api_secret_key: str = ""
    rate_limit_rpm: int = 30
    user_token_quota_daily: int = Field(
        default=500_000,
        ge=0,
        description=(
            "每用户 24 小时内可消耗的最大 Token 数。设为 0 则禁用配额检查。匿名用户共享同一配额池。"
        ),
    )

    http_request_timeout_seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=7200.0,
        description=(
            "可选的 ASGI 层每请求挂钟超时时间（秒）。"
            "零值表示禁用。豁免路径：``/health``、文档/OpenAPI 以及 ``/api/supervisor/research/stream``（SSE 可能合理地超出任何短时限制）。"
        ),
    )

    reflection_enabled: bool = False
    """当设为 True 时，研究主管会将最终综合结果包裹在批评者+写作者的反思循环中。
    默认关闭，因为反思每次请求会增加 1–3 次额外的 LLM 调用，且大多数流程不需要；在生产环境中设为 True 可提高回答质量。"""

    hitl_enabled: bool = False
    """当设为 True 时，研究主管在生成综合草稿后暂停，等待人工审批后再继续进入反思（如已启用）或最终定稿。
    前端会收到包含草稿的 ``review_requested`` SSE 事件；
    审阅者调用``POST /api/supervisor/research/{thread_id}/approve`` 或 ``/resume`` 以继续。
    需要持久化检查点存储（SQLite 或 Postgres）—— 内存检查点会在请求间丢失状态。"""

    reflection_pass_threshold: float = 0.85
    """批评者评分达到或超过此阈值时，反思循环提前终止。
    0.85 对应批评者提示词中"轻微修改后即可发布"的区间。"""

    reflection_max_iterations: int = 2
    """写作者重写次数的硬上限。每次请求最坏情况的 LLM 预算为 ``max_iterations + 1`` 次批评者调用加上 ``max_iterations``次写作者调用。"""

    default_recursion_limit: int = Field(
        default=40,
        ge=10,
        le=150,
        description=(
            "客户端未指定时应用的默认 LangGraph 递归限制。"
            "4 次专家移交 × 每次约 4 步 + supervisor 规划/合成 ≈ 20-25 步；40 留有余量且不会过深。"
        ),
    )

    sse_research_heartbeat_seconds: float = Field(
        default=15.0,
        ge=0,
        le=86400,
        description=(
            "图空闲期间 ``/api/supervisor/research/stream`` 上 SSE 保活 DATA 帧的发送间隔 — 防止反向代理/CDN 关闭长连接请求。零值禁用心跳。"
        ),
    )

    checkpoint_sqlite_path: str = Field(
        default="data/langgraph_checkpoint.db",
        description=(
            "启动时 Postgres 不可达时，LangGraph 将检查点写入此 SQLite 文件"
            "（父目录会自动创建）。设为空字符串则跳过 SQLite，仅回退到内存检查点。"
        ),
    )

    mcp_tool_discovery_timeout: float = Field(
        default=30.0,
        ge=5,
        le=300,
        description=(
            "启动时每个 MCP 工具发现调用的超时时间（秒）。若子进程枚举工具耗时超过此值，则跳过该 specialist。"
        ),
    )

    memory_store_sqlite_path: str = Field(
        default="data/langgraph_memory_store.db",
        description=(
            "启动时 Postgres 不可达时，长期记忆（用户偏好、研究历史）通过 AsyncSqliteStore 持久化到此 SQLite 文件。"
            "设为空字符串则跳过 SQLite，回退到 InMemoryStore（非持久化）。"
        ),
    )

    conversation_sqlite_path: str = Field(
        default="data/conversations.db",
        description="会话历史持久化 SQLite 路径，存储用户对话记录和消息。",
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
