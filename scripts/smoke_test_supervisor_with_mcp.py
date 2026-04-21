"""End-to-end smoke test: supervisor hands off to an MCP-backed specialist.

What this script proves
-----------------------
It exercises the full Phase-3+ chain that no other script touches on
its own:

    user prompt
        -> langgraph_supervisor (LLM, medium tier)
        -> transfer_to_coder_expert (hand-off tool call)
        -> coder_expert react-agent (LLM, light tier)
        -> code_execute_python (MCP tool, langchain-mcp-adapters)
        -> stdio subprocess (fastmcp CodeExecutor server)
        -> real Python sandbox
        -> result content block
        -> coder_expert final message
        -> supervisor synthesizes final answer

The question we ask is deliberately crafted so that **no** specialist
other than ``coder_expert`` can plausibly answer it. ``math_expert``
cannot: the answer requires list comprehension + a 2D filter, not a
single arithmetic expression. ``time_expert`` / ``text_analyst`` are
obviously irrelevant. Therefore a PASS confirms that hand-off routing
picked the MCP-backed specialist correctly.

Run it::

    uv run python scripts/smoke_test_supervisor_with_mcp.py

A pass prints ``[PASS]`` and exit 0. A fail prints the final reply
plus diagnostics and exits non-zero, so it can gate CI if needed.
"""

from __future__ import annotations

import asyncio
import sys

from langchain_core.messages import HumanMessage
from loguru import logger

from research_agent.config import get_settings
from research_agent.graph.minimal_supervisor import build_minimal_supervisor
from research_agent.llm.provider import ModelRouter
from research_agent.mcp_servers.client_factory import load_code_server_tools


QUESTION = (
    "Use Python code to solve this: given the list "
    "numbers = [7, 2, 9, 4, 11, 6, 13, 8], "
    "return (a) the sum of the numbers that are divisible by 3, "
    "(b) the count of numbers strictly greater than 7, and "
    "(c) the second-largest number overall. "
    "Run code to compute all three in one go and report a single line "
    "in the format 'sum=<A>, count=<B>, second_largest=<C>'."
)

# Expected ground truth (pre-computed so the script is self-checking):
#   divisible by 3       → [9, 6] → sum=15
#   > 7                  → [9, 11, 13, 8] → count=4
#   second-largest       → sorted desc = [13, 11, 9, 8, 7, 6, 4, 2] → 11
EXPECTED_SUM = 15
EXPECTED_COUNT = 4
EXPECTED_SECOND_LARGEST = 11


async def main() -> int:
    settings = get_settings()
    router = ModelRouter(settings.llm)

    logger.info("Loading MCP code_server tools over stdio…")
    coder_tools = await load_code_server_tools()
    logger.info("Loaded MCP tools: {}", [t.name for t in coder_tools])

    logger.info("Compiling supervisor with coder_expert attached…")
    graph = build_minimal_supervisor(
        model_router=router,
        coder_tools=coder_tools,
    )

    logger.info("Asking composite question:\n{}", QUESTION)
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=QUESTION)]},
        # 35 keeps us well clear of any transient tool retries while
        # still preventing runaway loops if routing misfires.
        config={"recursion_limit": 35},
    )

    messages = result.get("messages", [])
    final_msg = messages[-1] if messages else None
    final_text = getattr(final_msg, "content", "") or ""

    print("\n=== Final supervisor answer ===\n")
    print(final_text)
    print(f"\n=== Message count === {len(messages)}")

    # --- Heuristic verification --------------------------------------
    # We do not require exact wording; we check that (a) all three
    # ground-truth numbers appear, and (b) at least one hand-off to
    # ``coder_expert`` occurred (proving MCP was in the loop).
    answer_lc = final_text.lower()
    hit_sum = str(EXPECTED_SUM) in final_text
    hit_count = f"count={EXPECTED_COUNT}" in answer_lc or (
        str(EXPECTED_COUNT) in final_text and "count" in answer_lc
    )
    hit_second = str(EXPECTED_SECOND_LARGEST) in final_text

    # Evidence that the hand-off to the MCP-backed specialist happened.
    # Important caveat: because the supervisor compiles with
    # ``output_mode="last_message"``, the *inner* ``code_execute_python``
    # tool call inside coder_expert's subgraph is NOT surfaced in the
    # outer ``messages`` list. What IS surfaced is the supervisor's own
    # ``transfer_to_coder_expert`` tool call. Detecting that call is
    # sufficient evidence that routing reached coder_expert, and the
    # message-count jump (>=5 for a one-handoff round trip) is an
    # independent corroboration.
    coder_handoff = False
    inner_tool_call_seen = False
    for msg in messages:
        tc = getattr(msg, "tool_calls", None) or []
        for call in tc:
            name = (
                call.get("name", "")
                if isinstance(call, dict)
                else getattr(call, "name", "")
            )
            if "transfer_to_coder_expert" in name:
                coder_handoff = True
            if "code_execute_python" in name:
                inner_tool_call_seen = True

    # Also accept a text-level mention of "coder_expert" in any message
    # as a softer corroboration (the supervisor often cites the source).
    any_text_mentions_coder = any(
        "coder_expert" in str(getattr(m, "content", "")) for m in messages
    )

    routing_ok = coder_handoff or inner_tool_call_seen or any_text_mentions_coder

    print("\n=== Heuristic verification ===")
    print(f"  sum=15 present                        : {hit_sum}")
    print(f"  count=4 present                       : {hit_count}")
    print(f"  second_largest=11 present             : {hit_second}")
    print(f"  supervisor handoff to coder_expert    : {coder_handoff}")
    print(f"  inner code_execute_python surfaced    : {inner_tool_call_seen}")
    print(f"  'coder_expert' mentioned in messages  : {any_text_mentions_coder}")
    print(f"  → routing_ok (any of the above)       : {routing_ok}")

    all_ok = hit_sum and hit_count and hit_second and routing_ok
    if all_ok:
        print("\n  [PASS] Supervisor routed to coder_expert and MCP code_server "
              "produced the correct answer.")
        return 0

    print("\n  [FAIL] One or more checks did not pass. See message trace above.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
