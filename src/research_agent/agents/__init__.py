"""Agent definitions — each agent has a system prompt, tools, and model tier."""

from research_agent.agents.base import AgentConfig, build_agent

__all__ = ["AgentConfig", "build_agent"]
