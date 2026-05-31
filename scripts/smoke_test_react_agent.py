"""单 Agent Function Calling 循环的冒烟测试。

运行::

    uv run python scripts/smoke_test_react_agent.py

本脚本验证的端到端流程：
1. LLM 抽象层从 ``.env`` 加载 API 凭证。
2. ``create_agent`` 编译出一个已绑定工具的可工作图。
3. LLM 根据用户查询正确决定何时调用工具。
4. 工具结果被回传给 LLM。
5. LLM 基于工具输出生成最终回答。

运行三个探针，分别测试不同的工具：
    - 时间查询      → 强制调用 ``get_current_time``。
    - 数学查询      → 强制调用 ``calculate``。
    - 词数统计查询  → 强制调用 ``get_word_count``。
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from loguru import logger

from research_agent.agents.simple import build_simple_agent
from research_agent.config import get_settings
from research_agent.llm.provider import ModelRouter

PROBES: list[dict[str, str]] = [
    {
        "label": "时间查询",
        "question": "What's the current date and time in Shanghai? Please use your tools.",
        "expected_tool": "get_current_time",
    },
    {
        "label": "数学查询",
        "question": "Calculate (1234 * 567) + (89 ** 2) / 7. Show the exact number.",
        "expected_tool": "calculate",
    },
    {
        "label": "词数统计查询",
        "question": (
            "Count the words in this sentence: 'LangGraph makes multi-agent "
            "systems observable and recoverable.'"
        ),
        "expected_tool": "get_word_count",
    },
]


def _summarize_trace(messages: list[Any]) -> dict[str, Any]:
    """从 ReAct 消息跟踪中提取工具调用和最终回答。"""
    tool_calls: list[str] = []
    tool_results: list[str] = []
    final_answer: str = ""

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(f"{tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            tool_results.append(f"{msg.name} → {msg.content}")
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            final_answer = str(msg.content)

    return {
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "final_answer": final_answer,
        "message_count": len(messages),
    }


async def run_probe(agent: Any, probe: dict[str, str]) -> None:
    print("\n" + "=" * 70)
    print(f"  {probe['label']}: {probe['question']}")
    print(f"  预期工具调用: {probe['expected_tool']}")
    print("=" * 70)

    result = await agent.ainvoke({"messages": [HumanMessage(content=probe["question"])]})

    trace = _summarize_trace(result["messages"])

    print(f"\n  已发出的工具调用  : {trace['tool_calls']}")
    print(f"  工具返回结果      : {trace['tool_results']}")
    print(f"  最终回答          : {trace['final_answer']}")
    print(f"  消息总数          : {trace['message_count']}")

    expected = probe["expected_tool"]
    called_tools = [tc.split("(")[0] for tc in trace["tool_calls"]]
    if expected in called_tools:
        print(f"  [PASS] '{expected}' 已被调用。")
    else:
        print(f"  [WARN] 预期 '{expected}' 但实际调用了 {called_tools}")


async def main() -> None:
    settings = get_settings()
    logger.info("Loaded API base: {}", settings.llm.deepseek_api_base)
    logger.info("Light model   : {}", settings.llm.light_model)

    router = ModelRouter(settings.llm)
    agent = build_simple_agent(router)
    logger.info("ReAct agent compiled with default toolset.")

    for probe in PROBES:
        try:
            await run_probe(agent, probe)
        except Exception as e:
            print(f"\n  [FAIL] 探针抛出 {type(e).__name__}: {e}")

    print("\n" + "=" * 70)
    print("  冒烟测试完成。")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
