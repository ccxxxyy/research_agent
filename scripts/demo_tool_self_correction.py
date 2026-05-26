"""演示 Agent 在无人干预下从工具错误中自动恢复。

运行::

    uv run python scripts/demo_tool_self_correction.py

ReAct 循环有一个简单却非常有用的特性：工具的错误消息会作为 ToolMessage 反馈给 LLM，LLM 可以读取错误信息并使用不同参数重试。

通过两个确定性探针来展示这一点。每个探针使用一个自定义工具，
该工具保证在第一次调用时以特定、有教育意义的方式失败，然后接受修正后的调用。这避免了依赖随机的 LLM 推理来制造失败。

探针 1 — 临时服务故障（使用相同参数重试）注入 ``flaky_calculator``，首次调用返回错误，第二次成功。 LLM 必须读取错误并使用相同表达式再次调用。

探针 2 — 严格验证失败（使用修正后的参数重试）注入 ``length_in_cm``，拒绝枚举集合 {m, mm, cm, inch} 之外的任何单位。
    用户的问题故意在文本中使用了"meters"这个口语化词汇；LLM 必须读取工具的错误消息并用 unit="m" 重新调用。

每次调用都受严格的 ``recursion_limit`` 约束，防止进程在病态重试循环中卡死。
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from loguru import logger

from research_agent.agents.simple import build_simple_agent
from research_agent.config import get_settings
from research_agent.llm.provider import ModelRouter


# ---------------------------------------------------------------------------
# 演示用的可观测工具
# ---------------------------------------------------------------------------

FLAKY_CALLS = 0
STRICT_CALLS: list[dict[str, Any]] = []


@tool
def flaky_calculator(expression: str) -> str:
    """计算简单的算术表达式。此计算器不稳定。

    该服务已知不可靠：可能出现临时性错误。如果返回内容以 "Error:" 开头，请使用相同参数重试一次。

    Args:
        expression: 算术表达式，如 "12 * 11"。
    """
    global FLAKY_CALLS
    FLAKY_CALLS += 1
    logger.debug("flaky_calculator call #{} expression={!r}", FLAKY_CALLS, expression)
    if FLAKY_CALLS == 1:
        return "Error: ServiceTemporarilyUnavailable. Please retry the same call once."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@tool
def length_in_cm(value: float, unit: str) -> str:
    """将长度转换为厘米。

    ``unit`` 参数期望接收内部单位代码字符串，而非人类可读的名称。 此处不列出所有可接受的代码；如果传入的代码未被识别，工具的错误消息会报告哪些代码是有效的。

    Args:
        value: 数值长度（非负数）。
        unit: 内部单位代码字符串。若不确定有效值，参考工具的错误消息。
    """
    STRICT_CALLS.append({"value": value, "unit": unit})
    logger.debug("length_in_cm call #{} value={} unit={!r}", len(STRICT_CALLS), value, unit)
    # 故意使用不透明的代码（UNIT_* 命名空间）。首次遇到的模型几乎肯定会猜 "m"、"meter" 或 "meters"，从而触发纠正反馈路径。
    factors = {
        "UNIT_METRE": 100.0,
        "UNIT_MM": 0.1,
        "UNIT_CM": 1.0,
        "UNIT_INCH": 2.54,
    }
    if unit not in factors:
        return (
            f"Error: unit code '{unit}' is not recognised. "
            f"Valid codes are exactly: {sorted(factors)}. "
            f"Retry with one of these codes."
        )
    return f"{value * factors[unit]:.4f} cm"


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _trace(messages: list[Any]) -> dict[str, Any]:
    tool_calls: list[str] = []
    tool_results: list[str] = []
    final = ""

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(f"{tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            preview = str(msg.content).splitlines()[0][:110]
            tool_results.append(f"{msg.name} -> {preview}")
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            final = str(msg.content)

    return {
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "final": final,
        "n_messages": len(messages),
    }


async def run_probe(agent, label: str, question: str, success_predicate) -> None:
    _banner(label)
    print(f"  用户问题 : {question}")

    # 严格的递归限制，使失控重试循环快速失败而非卡死。
    # 一个 ReAct 步骤约等于一次 LLM 调用；10 步 足以舒适覆盖 "一次重试"场景。
    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": 10},
        )
    except Exception as e:
        print(f"  [FAIL] Agent 抛出 {type(e).__name__}: {e}")
        return

    trace = _trace(result["messages"])

    print("\n  已发出的工具调用:")
    for tc in trace["tool_calls"]:
        print(f"    - {tc}")
    print("\n  观测到的工具结果:")
    for tr in trace["tool_results"]:
        print(f"    - {tr}")
    print(f"\n  工具调用总数  : {len(trace['tool_calls'])}")
    print(f"  最终回答      : {trace['final'][:160]}")

    ok = success_predicate(trace)
    verdict = "PASS" if ok else "WARN"
    print(f"  [{verdict}] 自纠正断言")


def probe1_predicate(trace: dict[str, Any]) -> bool:
    """成功条件：flaky_calculator 调用 >=2 次 且 最终回答包含 132。"""
    flaky_count = sum(1 for tc in trace["tool_calls"] if tc.startswith("flaky_calculator"))
    had_error = any("Error:" in r for r in trace["tool_results"])
    return flaky_count >= 2 and had_error and "132" in trace["final"]


def probe2_predicate(trace: dict[str, Any]) -> bool:
    """成功条件：length_in_cm 调用 >=2 次且使用了不同的单位代码；结果包含 100。"""
    units_tried = {c["unit"] for c in STRICT_CALLS}
    had_error = any("Error:" in r for r in trace["tool_results"])
    return (
        len(STRICT_CALLS) >= 2
        and len(units_tried) >= 2
        and had_error
        and "100" in trace["final"]
    )


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

SELF_CORRECTION_PROMPT = """\
You are a careful assistant with access to several tools.

Self-correction protocol — follow it strictly:
- If any tool returns content starting with "Error:", READ the error
  message, decide whether to retry with the same arguments (for transient
  errors) or adjust the arguments (for validation errors).
- Retry AT MOST ONCE per tool. After one retry, if the error persists,
  answer the user truthfully about the failure.
- Always produce a concise final user-visible answer when you stop.
"""


async def main() -> None:
    global FLAKY_CALLS
    settings = get_settings()
    logger.info("Light model: {}", settings.llm.light_model)

    router = ModelRouter(settings.llm)
    agent = build_simple_agent(
        router,
        tools=[flaky_calculator, length_in_cm],
        prompt=SELF_CORRECTION_PROMPT,
    )

    # ------ 探针 1 — 临时错误，使用相同参数重试 ------
    FLAKY_CALLS = 0
    await run_probe(
        agent,
        "探针 1 — 临时错误，使用相同参数重试",
        "Use flaky_calculator to compute 12 * 11. Report the final number.",
        probe1_predicate,
    )

    # ------ 探针 2 — 验证错误，使用修正后的参数重试 ------
    STRICT_CALLS.clear()
    await run_probe(
        agent,
        "探针 2 — 验证错误，使用修正后的参数重试",
        "Convert 1 metre into centimetres using length_in_cm. "
        "You don't know the tool's expected unit code up front — "
        "make an initial reasonable guess, read the tool's error if any, "
        "and correct your unit argument accordingly.",
        probe2_predicate,
    )

    _banner("要点总结")
    print(
        """
  ReAct 循环在应用侧无需任何特殊错误处理代码的情况下就具有 出色的容错能力。关键要素是：

    1. 工具返回结构化的、描述性的错误（从不抛异常）。
    2. 错误变成图状态中的 ToolMessage。
    3. LLM 在下一轮看到这些 ToolMessage 并决定是否重试、 调整参数、还是放弃。
    4. recursion_limit 封顶失控循环，使不诚实的工具或混乱的模型无法无限消耗 token。

  在生产中，搭配一个 supervisor 节点在连续 N 次失败后升级到更强的模型，以及结构化遥测使失败的工具在仪表盘中显现 —在它们降低用户体验之前。
        """.rstrip()
    )


if __name__ == "__main__":
    asyncio.run(main())
