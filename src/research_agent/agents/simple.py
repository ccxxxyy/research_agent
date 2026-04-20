"""Single-agent factory using LangGraph's create_react_agent.

This module provides the simplest possible agent setup to verify that
the LLM → Function Calling → Tool execution loop works end-to-end.

The ReAct pattern runs the following loop:

    Reasoning: "I need to know the current time."
    ↓
    Acting:    LLM emits a tool_call for `get_current_time(timezone_name="Asia/Shanghai")`
    ↓
    Observing: Tool returns "2026-04-19T21:30:00+08:00"
    ↓
    Reasoning: "Now I can answer the user."
    ↓
    Final answer emitted to the user.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from research_agent.llm.provider import ModelRouter
from research_agent.llm.tier import AgentName
from research_agent.tools.native import DEFAULT_TOOLS


SIMPLE_AGENT_PROMPT = """\
You are a helpful assistant with access to a small toolbox.

Available tools:
- get_current_time: Returns the current date/time for an IANA timezone.
- calculate: Evaluates mathematical expressions.
- get_word_count: Counts words in a text.

Guidelines:
1. Use tools whenever they can give a more accurate, up-to-date, or
   deterministic answer than your own reasoning.
2. For arithmetic, ALWAYS use the ``calculate`` tool — do not do mental math.
3. For time-sensitive questions, ALWAYS use ``get_current_time``.
4. After getting tool results, synthesize a clear, concise final answer.
"""


def build_simple_agent(
    model_router: ModelRouter,
    tools: list[BaseTool] | None = None,
    prompt: str = SIMPLE_AGENT_PROMPT,
):
    """Build a single ReAct-style agent with Function Calling enabled.

    The returned object is a compiled LangGraph app with the structure:

        START → agent_node ─┬─→ tools_node → agent_node (loop)
                             └─→ END (when the LLM stops requesting tools)

    Args:
        model_router: Resolves the model for the "retriever" tier (light, fast).
        tools: Tools to expose. Defaults to :data:`DEFAULT_TOOLS`.
        prompt: System prompt that explains the toolbox to the LLM.

    Returns:
        A compiled LangGraph app that can be invoked with ``ainvoke``.
    """
    tools = tools if tools is not None else DEFAULT_TOOLS
    model = model_router.for_agent(AgentName.RETRIEVER)

    return create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
    )
