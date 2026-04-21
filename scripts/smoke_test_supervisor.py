"""Smoke test for Phase-3 minimal supervisor (multi-agent handoffs).

Run:
    uv run python scripts/smoke_test_supervisor.py

Requires a working LLM configuration in ``.env`` (same as Phase-1 smoke).

This script sends a COMPOSITE question that forces the supervisor to
delegate to at least two different specialists in sequence, then
synthesise a single final answer.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage
from loguru import logger

from research_agent.config import get_settings
from research_agent.graph.minimal_supervisor import build_minimal_supervisor
from research_agent.llm.provider import ModelRouter


def _last_plain_assistant(messages: list) -> str:
    from langchain_core.messages import AIMessage

    for m in reversed(messages):
        if isinstance(m, AIMessage):
            tc = getattr(m, "tool_calls", None) or []
            if not tc and m.content:
                return str(m.content)
    return ""


async def main() -> None:
    settings = get_settings()
    router = ModelRouter(settings.llm)
    graph = build_minimal_supervisor(model_router=router)

    question = (
        "I need two things in one answer: "
        "(1) What is the current time in Asia/Shanghai? "
        "(2) What is 17 * 23? "
        "After you have both facts, combine them into one short paragraph."
    )
    logger.info("Composite question:\n{}", question)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 25},
    )

    final = _last_plain_assistant(result["messages"])
    print("\n=== Final supervisor answer ===\n")
    print(final)
    print("\n=== Message count ===", len(result["messages"]))

    # Soft checks — we don't hard-assert exact wording from the LLM.
    lower = final.lower()
    ok_time = "shanghai" in lower or "+08" in final or "202" in final
    ok_math = "391" in final.replace(",", "").replace(" ", "")
    print("\n=== Heuristic verification ===")
    print(f"  time-ish content present : {ok_time}")
    print(f"  17*23=391 present        : {ok_math}")
    if ok_time and ok_math:
        print("\n  [PASS] Supervisor appears to have delegated correctly.")
    else:
        print("\n  [WARN] Heuristic checks failed — inspect the trace above.")


if __name__ == "__main__":
    asyncio.run(main())
