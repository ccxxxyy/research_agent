"""Base agent configuration and factory for creating LangGraph react agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from research_agent.llm.provider import ModelRouter
from research_agent.llm.tier import AgentName


@dataclass(frozen=True)
class AgentConfig:
    """Declarative agent specification — decouples definition from execution."""

    name: AgentName
    system_prompt: str
    tools: list[BaseTool] = field(default_factory=list)
    description: str = ""


def build_agent(
    config: AgentConfig,
    model_router: ModelRouter,
    **kwargs: Any,
) -> Any:
    """Create a LangGraph react agent from an AgentConfig.

    Uses create_react_agent which wraps tool-calling in a ReAct loop:
    Reasoning → Action → Observation → Reasoning ...
    """
    model = model_router.for_agent(config.name)

    return create_react_agent(
        model=model,
        tools=config.tools,
        name=config.name.value,
        prompt=config.system_prompt,
        **kwargs,
    )
