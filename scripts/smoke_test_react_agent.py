"""Smoke test for the single-agent Function Calling loop.

Run with:
    uv run python scripts/smoke_test_react_agent.py

What this script verifies (end-to-end):
1. LLM abstraction loads API credentials from ``.env``.
2. ``create_react_agent`` compiles a working graph with tools attached.
3. The LLM correctly decides WHEN to invoke tools based on the user query.
4. Tool results are fed back to the LLM.
5. The LLM produces a final answer grounded in tool outputs.

Three probes are run, each stressing a different tool:
    - Time query      → forces ``get_current_time`` call.
    - Math query      → forces ``calculate`` call.
    - Word-count query → forces ``get_word_count`` call.
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
        "label": "TIME QUERY",
        "question": "What's the current date and time in Shanghai? Please use your tools.",
        "expected_tool": "get_current_time",
    },
    {
        "label": "MATH QUERY",
        "question": "Calculate (1234 * 567) + (89 ** 2) / 7. Show the exact number.",
        "expected_tool": "calculate",
    },
    {
        "label": "WORD-COUNT QUERY",
        "question": (
            "Count the words in this sentence: 'LangGraph makes multi-agent "
            "systems observable and recoverable.'"
        ),
        "expected_tool": "get_word_count",
    },
]


def _summarize_trace(messages: list[Any]) -> dict[str, Any]:
    """Extract tool calls and final answer from a ReAct message trace."""
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
    print(f"  Expected tool call: {probe['expected_tool']}")
    print("=" * 70)

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=probe["question"])]}
    )

    trace = _summarize_trace(result["messages"])

    print(f"\n  Tool calls emitted  : {trace['tool_calls']}")
    print(f"  Tool results        : {trace['tool_results']}")
    print(f"  Final answer        : {trace['final_answer']}")
    print(f"  Total messages      : {trace['message_count']}")

    expected = probe["expected_tool"]
    called_tools = [tc.split("(")[0] for tc in trace["tool_calls"]]
    if expected in called_tools:
        print(f"  [PASS] '{expected}' was invoked.")
    else:
        print(f"  [WARN] Expected '{expected}' but got {called_tools}")


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
            print(f"\n  [FAIL] Probe raised {type(e).__name__}: {e}")

    print("\n" + "=" * 70)
    print("  Smoke test complete.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
