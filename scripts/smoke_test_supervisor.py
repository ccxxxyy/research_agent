"""最小 supervisor（多 Agent 移交）冒烟测试。

运行::

    uv run python scripts/smoke_test_supervisor.py

需要 ``.env`` 中有可用的 LLM 配置（与 Phase-1 冒烟测试相同）。

本脚本发送一个复合问题，迫使 supervisor 依次委派给至少两个不同的专家，然后综合出一个最终回答。
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
    print("\n=== Supervisor 最终回答 ===\n")
    print(final)
    print("\n=== 消息总数 ===", len(result["messages"]))

    # 宽松校验 — 不硬性断言 LLM 的精确措辞。
    lower = final.lower()
    ok_time = "shanghai" in lower or "+08" in final or "202" in final
    ok_math = "391" in final.replace(",", "").replace(" ", "")
    print("\n=== 启发式校验 ===")
    print(f"  包含时间相关内容   : {ok_time}")
    print(f"  17*23=391 存在     : {ok_math}")
    if ok_time and ok_math:
        print("\n  [PASS] Supervisor 似乎正确完成了委派。")
    else:
        print("\n  [WARN] 启发式校验失败 — 请检查上方跟踪信息。")


if __name__ == "__main__":
    asyncio.run(main())
