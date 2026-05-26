"""Agent 定义 — system prompt、工具、模型层级与构建器。

导出两层内容：

1. 通用的 ``AgentConfig`` / ``build_agent`` 组合，供原始的基于节点的研究图（``graph/supervisor.py``）使用。
2. 专家构建器，供 supervisor 图（``graph/minimal_supervisor.py`` 和``graph/research_supervisor.py``）消费。
    在此处重新导出意味着下游代码无需拼写完整路径``research_agent.agents.specialists`` — 一个小的易用性改进，使 supervisor 的接线代码更具可读性。
"""

from research_agent.agents.base import AgentConfig, build_agent
from research_agent.agents.specialists import (
    SPECIALIST_BUILDERS,
    build_coder_expert,
    build_data_expert,
    build_knowledge_expert,
    build_math_expert,
    build_news_expert,
    build_report_expert,
    build_sentiment_expert,
    build_text_analyst,
    build_time_expert,
)

__all__ = [
    "AgentConfig",
    "SPECIALIST_BUILDERS",
    "build_agent",
    "build_coder_expert",
    "build_data_expert",
    "build_knowledge_expert",
    "build_math_expert",
    "build_news_expert",
    "build_report_expert",
    "build_sentiment_expert",
    "build_text_analyst",
    "build_time_expert",
]
