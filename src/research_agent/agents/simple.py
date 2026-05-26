"""使用 LangGraph create_react_agent 的单 Agent 工厂。

本模块提供最简单的 Agent 设置，用于验证
LLM → Function Calling → 工具执行 的端到端流程是否正常工作。

ReAct 模式运行以下循环：

    推理: "我需要知道当前时间。"
    ↓
    行动:    LLM 发出对 `get_current_time(timezone_name="Asia/Shanghai")` 的 tool_call
    ↓
    观察: 工具返回 "2026-04-19T21:30:00+08:00"
    ↓
    推理: "现在我可以回答用户了。"
    ↓
    向用户输出最终答案。
"""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt import create_react_agent

from research_agent.llm.provider import ModelRouter
from research_agent.llm.tier import AgentName
from research_agent.tools.native import DEFAULT_TOOLS


SIMPLE_AGENT_PROMPT = """\
你是一个拥有小型工具箱的智能助手。

可用工具：
- get_current_time：返回指定 IANA 时区的当前日期/时间。
- calculate：计算数学表达式。
- get_word_count：统计文本中的单词数。

使用指南：
1. 当工具能给出比你自身推理更准确、更及时或更确定的答案时，始终使用工具。
2. 对于算术运算，始终使用 ``calculate`` 工具 — 不要心算。
3. 对于时间敏感的问题，始终使用 ``get_current_time``。
4. 获取工具结果后，综合出清晰、简洁的最终答案。
"""


def build_simple_agent(
    model_router: ModelRouter,
    tools: list[BaseTool] | None = None,
    prompt: str = SIMPLE_AGENT_PROMPT,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """构建一个启用 Function Calling 的单 ReAct 风格 Agent。

    返回的对象是一个已编译的 LangGraph 应用，结构如下：

        START → agent_node ─┬─→ tools_node → agent_node（循环）
                             └─→ END（当 LLM 停止请求工具时）

    Args:
        model_router: 为 "retriever" 层级（轻量、快速）解析模型。
        tools: 要暴露的工具。默认为 :data:`DEFAULT_TOOLS`。
        prompt: 向 LLM 说明工具箱的 system prompt。
        checkpointer: 可选的 LangGraph checkpointer。提供时，Agent 支持以 ``thread_id`` 为键的多轮对话，并可在进程重启后从最后持久化的步骤恢复。不提供时每次调用都从头开始。

    Returns:
        一个可通过 ``ainvoke`` 调用的已编译 LangGraph 应用。

    Usage:
        在 runnable config 中提供 ``thread_id`` 以持久化状态::

            agent = build_simple_agent(router, checkpointer=saver)
            cfg = {"configurable": {"thread_id": "session-42"}}
            await agent.ainvoke({"messages": [HumanMessage("hi")]}, config=cfg)
            # 使用相同 thread_id 的后续调用可看到之前的消息。
    """
    tools = tools if tools is not None else DEFAULT_TOOLS
    model = model_router.for_agent(AgentName.RETRIEVER)

    return create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        checkpointer=checkpointer,
    )
