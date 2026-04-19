"""LangSmith tracing integration for agent execution observability."""

from __future__ import annotations

import os

from loguru import logger

from research_agent.config import ObservabilityConfig


def setup_tracing(config: ObservabilityConfig) -> None:
    """Enable LangSmith tracing if configured.

    LangSmith provides:
    - Full trace of every LLM call, tool call, and chain execution
    - Latency breakdown per node in the LangGraph
    - Token usage and cost tracking
    - Feedback collection for evaluation
    """
    if not config.langchain_tracing_v2 or not config.langsmith_api_key:
        logger.info("LangSmith tracing disabled")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = config.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = config.langsmith_project

    logger.info("LangSmith tracing enabled for project: {}", config.langsmith_project)
