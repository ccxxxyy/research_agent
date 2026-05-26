"""LangSmith 链路追踪集成，追踪每次 Agent 调用链路，用于 Agent 执行可观测性。每次 LLM 调用花了多长时间、用了多少 token"""

from __future__ import annotations

import os

from loguru import logger

from research_agent.config import ObservabilityConfig


def setup_tracing(config: ObservabilityConfig) -> None:
    """若已配置则启用 LangSmith 链路追踪。

    LangSmith 提供：
    - 每次 LLM 调用、工具调用和链执行的完整追踪
    - LangGraph 中各节点的延迟分解
    - Token 用量与费用跟踪
    - 用于评估的反馈收集
    """
    if not config.langchain_tracing_v2 or not config.langsmith_api_key:
        logger.info("LangSmith tracing disabled")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = config.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = config.langsmith_project

    logger.info("LangSmith tracing enabled for project: {}", config.langsmith_project)
