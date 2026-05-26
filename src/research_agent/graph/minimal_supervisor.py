"""最小化 supervisor 模式图

为什么要与 ``supervisor.py`` 分离成独立图？
本包中的 ``supervisor.py（ research_supervisor.py)`` 模块连接了一条重量级的
Corrective-RAG + Reflection 流水线，是完整研究智能体的长期归属。
因为目标是在不引入检索、评分或反思噪音的前提下，隔离并演示supervisor-of-specialists 模式本身。提供了一个干净、快速、完全可测试的最小示例，便于在此基础上迭代。

拓扑结构：

              ┌─────────────────────────┐
              │       supervisor        │   <-- LLM 决定下一步由谁处理
              └────┬───────┬───────┬────┘
                   │       │       │
                   ▼       ▼       ▼
              math_expert  time_expert  text_analyst
                   │       │       │
                   └───────┴───────┘
                           │
                           ▼
                        END（当 supervisor 输出最终回答时结束）

切换（Handoff）通过 ``langgraph_supervisor`` 自动注入到 supervisor 中的``transfer_to_<name>`` 工具完成。
每一步中 supervisor 要么调用转移工具（路由），要么生成最终的助手消息（终止）。
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph_supervisor import create_supervisor
from loguru import logger

from research_agent.agents.specialists import (
    build_coder_expert,
    build_math_expert,
    build_text_analyst,
    build_time_expert,
)
from research_agent.llm.provider import ModelRouter
from research_agent.llm.tier import ModelTier


SUPERVISOR_PROMPT_BASE = """\
你是一个小型专家团队的 Supervisor（主管）：

  - math_expert   ：使用计算器工具计算数学表达式。
  - time_expert   ：返回指定时区的当前日期/时间。
  - text_analyst  ：统计给定字符串中的单词数。
"""

SUPERVISOR_PROMPT_CODER_EXTENSION = """\
  - coder_expert  ：通过 MCP 子进程运行任意沙箱化 Python。
                    适用于需要真实代码执行的任务（统计、排序、集合运算、正则表达式、JSON 变换）— 凡是超出 ``math_expert``单表达式能力范围的，都可以交给它。
"""

SUPERVISOR_PROMPT_RULES = """\
你的职责：
1. 仔细阅读用户的请求。
2. 如果请求需要某位专家处理，请调用对应的 ``transfer_to_<name>`` 工具进行移交。每次只委派一个子任务；等待结果后再路由下一个子任务。
3. 对于复合请求（例如"统计 X 的单词数然后乘以 Y"），将其拆分为子任务，按顺序委派给各专家。
4. 收集完所有专家结果后，为用户撰写最终的简洁回答。不要把你自己能轻松完成的子任务也委派出去（例如格式调整）。
5. 永远不要自己做算术、时间查询、词数统计或代码执行 — 一律委派给专家。
"""


def _build_supervisor_prompt(*, include_coder: bool) -> str:
    """组装 supervisor 系统提示词，根据团队成员动态调整。

    仅当 ``coder_expert`` 确实存在于编译后的图中时，才在名单中列出它。
    如果提示词中列出了 supervisor 无法移交的专家，运行时调用``transfer_to_coder_expert`` 将会失败。
    """
    parts = [SUPERVISOR_PROMPT_BASE]
    if include_coder:
        parts.append(SUPERVISOR_PROMPT_CODER_EXTENSION)
    parts.append("\n" + SUPERVISOR_PROMPT_RULES)
    return "".join(parts)


def build_minimal_supervisor(
    *,
    model_router: ModelRouter,
    checkpointer: BaseCheckpointSaver | None = None,
    supervisor_tier: ModelTier = ModelTier.MEDIUM,
    coder_tools: Sequence[BaseTool] | None = None,
) -> CompiledStateGraph:
    """构建并编译 supervisor 图。

    图中始终包含三个 Python 本地专家（``math_expert``、``time_expert``、``text_analyst``）。
    当提供``coder_tools`` 时，会添加第四个 MCP 支持的专家（``coder_expert``），并扩展 supervisor 提示词以声明该专家。
    这种拆分使 ``build_minimal_supervisor`` 可被纯 Python 单元测试使用（无需启动 MCP 子进程），同时允许完整的冒烟测试 / 生产环境
    FastAPI 生命周期接入真实的 MCP 链接。

    Args:
        model_router: supervisor 和专家共用的模型路由器。supervisor 使用更强的模型层级（默认 :attr:`ModelTier.MEDIUM`），
        因为路由决策需要更好的推理能力；专家运行在 LIGHT 上。

        checkpointer: 可选的 LangGraph checkpointer。提供后，整个supervisor 对话（包括专家移交）将按 ``thread_id`` 持久化，多轮对话 / 恢复保证也扩展到此图。

        supervisor_tier: 覆盖 supervisor 模型层级。如果路由错误在实际使用中明显，可设为 HEAVY。

        coder_tools: 可选的预加载 MCP 工具列表（通常是 :func:`research_agent.mcp_servers.client_factory.load_code_server_tools`的返回值）。
        当为 ``None`` 或空时，``coder_expert`` 专家不会被添加到团队中。

    Returns:
        通过 ``ainvoke`` / ``astream`` 调用的已编译 LangGraph 应用。
    """
    math = build_math_expert(model_router)
    time_ = build_time_expert(model_router)
    text = build_text_analyst(model_router)

    agents: list = [math, time_, text]
    specialist_names = ["math_expert", "time_expert", "text_analyst"]

    include_coder = bool(coder_tools)
    if include_coder:
        coder = build_coder_expert(model_router, coder_tools or [])
        agents.append(coder)
        specialist_names.append("coder_expert")

    supervisor_model = model_router.get_model(supervisor_tier)
    prompt = _build_supervisor_prompt(include_coder=include_coder)

    workflow = create_supervisor(
        agents=agents,
        model=supervisor_model,
        prompt=prompt,
        # ``last_message`` 使共享状态保持紧凑：只有最后一条专家回复会追加回 supervisor 上下文。
        # 当需要 supervisor 看到专家的思维链时，切换到 ``full_history``。
        output_mode="last_message",
    )

    compiled = workflow.compile(checkpointer=checkpointer)
    logger.info(
        "Minimal supervisor compiled: tier={} specialists={}",
        supervisor_tier.value,
        specialist_names,
    )
    return compiled
