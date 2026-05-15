"""Agent definitions — system prompts, tools, model tiers, and builders.

Two layers are exported:

1. The Phase-1 generic ``AgentConfig`` / ``build_agent`` pair, used by
   the original node-based research graph (``graph/supervisor.py``).
2. The Phase-3/4 specialist builders consumed by the supervisor
   graphs (``graph/minimal_supervisor.py`` and
   ``graph/research_supervisor.py``). Re-exporting them here means
   downstream code never has to spell out
   ``research_agent.agents.specialists`` — a small ergonomic win that
   keeps the supervisor wiring readable.
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
