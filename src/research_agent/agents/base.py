"""Agent 基础配置与工厂函数，用于创建 LangGraph ReAct Agent。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langgraph.prebuilt import create_react_agent

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from research_agent.llm.provider import ModelRouter
    from research_agent.llm.tier import AgentName


@dataclass(frozen=True)
class AgentConfig:
    """声明式 Agent 规格 — 将定义与执行解耦。"""

    name: AgentName
    system_prompt: str
    tools: list[BaseTool] = field(default_factory=list)
    description: str = ""


def build_agent(
    config: AgentConfig,
    model_router: ModelRouter,
    **kwargs: Any,
) -> Any:
    """根据 AgentConfig 创建一个 LangGraph ReAct Agent。

    使用 create_react_agent，它将工具调用封装在 ReAct 循环中：
    推理 → 行动 → 观察 → 推理 ...
    """
    model = model_router.for_agent(config.name)

    return create_react_agent(
        model=model,
        tools=config.tools,
        name=config.name.value,
        prompt=config.system_prompt,
        **kwargs,
    )
