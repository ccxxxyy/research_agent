"""Phase-4.4 end-to-end demo — research supervisor over real MCP tools.

This script is the **production flow-of-the-day**: it spawns the two
MCP subprocesses that matter for A-share research (``fin_data_server``
and ``pdf_report_server``) plus the sandboxed ``code_server``, wires
their tools into the three Phase-4 specialists, compiles the
research supervisor, and then sends the graph a composite research
question that is deliberately crafted to exercise multi-specialist
routing.

Run::

    uv run python scripts/demo_financial_research.py

Requirements:
  - Network access (akshare → 东财/新浪, and cninfo for disclosures).
  - A working LLM config in ``.env`` (same file used by Phase-1 / 3 smoke).

Exit codes:
    0 → supervisor produced a final answer that passes soft sanity
        checks.
    1 → any configuration / MCP / LLM error, or the supervisor failed
        to route to the expected specialists.

Soft checks (intentionally lenient — we do NOT pin exact LLM wording):
  * final answer is non-empty
  * final answer references the company (name or ticker)
  * supervisor issued ``transfer_to_data_expert`` AND
    ``transfer_to_report_expert`` at least once each (verifying that
    routing reached both MCP-backed specialists, not just one).

Why we DON'T check for ``fin_*`` / ``pdf_*`` calls directly
----------------------------------------------------------
``build_research_supervisor`` compiles with ``output_mode="last_message"``,
which is the right production setting (compact state, cheap token
usage). A consequence is that each specialist's *internal*
``ToolMessage``/``AIMessage`` traffic stays inside its own subgraph
and never appears in the parent ``result["messages"]`` — only the
specialist's final summary message bubbles up, along with the
supervisor's ``transfer_to_*`` hand-off records. So we verify
routing via the hand-off records and verify correctness via the
content of the final answer.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from loguru import logger

from research_agent.config import get_settings
from research_agent.graph.research_supervisor import build_research_supervisor
from research_agent.llm.provider import ModelRouter
from research_agent.mcp_servers.client_factory import (
    load_code_server_tools,
    load_fin_data_server_tools,
    load_pdf_report_server_tools,
)


COMPANY_NAME = "宁德时代"
COMPANY_TICKER = "300750"

# A composite prompt that forces multi-hop routing:
#   data_expert   → basic info + recent price
#   report_expert → most recent annual/quarterly disclosure & key excerpt
#   coder_expert  → (optional) simple derived stat
# We DO NOT tell the supervisor which specialist to use; the whole
# point of the demo is that it plans the hand-offs itself.
QUESTION = (
    f"我想快速了解 {COMPANY_NAME}（股票代码 {COMPANY_TICKER}）最新的基本面与披露信息。\n"
    "请完成以下三件事，并在最终答复中用小标题区分：\n"
    "  1) 给出公司的基本资料（所属行业、最新收盘价、市值），"
    "     并附最近 5 个交易日的收盘价；\n"
    "  2) 从巨潮资讯检索该公司最近一份公开披露（最近 60 天内），"
    "     给出标题、披露日期、直接 PDF 链接，并用 1-2 段话摘录其中"
    "     任意关键章节的原文（<=200 字）；\n"
    "  3) 基于步骤 1 返回的 5 日收盘价，调用代码专家计算均值与标准差"
    "     （两位小数即可）。\n"
    "最后写一段 3-5 行的总结。"
)


def _last_plain_assistant(messages: list) -> str:
    """Return the last assistant message that is NOT a tool call."""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            tc = getattr(m, "tool_calls", None) or []
            if not tc and m.content:
                return str(m.content)
    return ""


def _trace_tool_calls(messages: list) -> list[str]:
    """Collect every tool name observable in the outer supervisor state.

    With ``output_mode="last_message"``, only the supervisor's own
    ``transfer_to_*`` calls appear here — specialists' internal MCP
    tool calls stay inside their subgraphs.

    Each ``transfer_to_*`` hand-off typically produces TWO entries:
      * an ``AIMessage.tool_calls`` record from the supervisor,
      * a ``ToolMessage`` response echoing the same name.
    We collect both without deduping; callers can count or ``set()``.
    """
    names: list[str] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            nm = getattr(m, "name", None) or ""
            if nm:
                names.append(nm)
        elif isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                nm = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if nm:
                    names.append(str(nm))
    return names


def _transfers_reached(calls: list[str]) -> set[str]:
    """Return the set of distinct specialists the supervisor routed to.

    ``transfer_back_to_supervisor`` (automatically injected by
    ``langgraph_supervisor``) is intentionally stripped — we care
    about which specialists were *invoked*, not how many times the
    supervisor regained control.
    """
    reached: set[str] = set()
    for n in calls:
        if n.startswith("transfer_to_") and n != "transfer_to_supervisor":
            reached.add(n[len("transfer_to_") :])
    return reached


async def _load_all_tools() -> dict[str, Any]:
    """Spawn the three MCP servers in parallel and collect their tools.

    Running the three loads concurrently shaves roughly 1 second off
    cold-start. Each call to ``load_*_tools`` spawns its own
    subprocess; the subprocesses are independent.
    """
    data_tools, report_tools, coder_tools = await asyncio.gather(
        load_fin_data_server_tools(),
        load_pdf_report_server_tools(),
        load_code_server_tools(),
    )
    logger.info(
        "MCP tools loaded: data={} report={} coder={}",
        len(data_tools),
        len(report_tools),
        len(coder_tools),
    )
    return {
        "data_tools": data_tools,
        "report_tools": report_tools,
        "coder_tools": coder_tools,
    }


async def main() -> int:
    settings = get_settings()
    router = ModelRouter(settings.llm)

    try:
        loaded = await _load_all_tools()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load MCP tools: {}", exc)
        return 1

    graph = build_research_supervisor(model_router=router, **loaded)

    logger.info("Sending composite research question:\n{}", QUESTION)

    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=QUESTION)]},
            # Recursion-limit budget:
            #   supervisor plans (≥3 hand-offs) + each specialist's
            #   ReAct loop (≤4 tool calls) + supervisor synthesis.
            # 50 is comfortable headroom.
            config={"recursion_limit": 50},
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Graph invocation crashed: {}", exc)
        return 1

    messages = result["messages"]
    final = _last_plain_assistant(messages)
    calls = _trace_tool_calls(messages)
    reached = _transfers_reached(calls)

    print("\n=== Final supervisor answer ===\n")
    print(final if final else "<empty>")
    print("\n=== Trace summary ===")
    print(f"  total messages        : {len(messages)}")
    print(f"  specialists reached   : {sorted(reached) or ['<none>']}")
    print(f"  total tool-name events: {len(calls)}")

    print("\n=== Heuristic verification ===")
    ok_answer = bool(final.strip())
    ok_subject = (COMPANY_NAME in final) or (COMPANY_TICKER in final)
    ok_data = "data_expert" in reached
    ok_report = "report_expert" in reached
    print(f"  non-empty final answer      : {ok_answer}")
    print(f"  company mentioned in answer : {ok_subject}")
    print(f"  data_expert was routed to   : {ok_data}")
    print(f"  report_expert was routed to : {ok_report}")

    if ok_answer and ok_subject and ok_data and ok_report:
        print("\n  [PASS] Supervisor routed to data_expert + report_expert.")
        return 0
    print("\n  [WARN] Heuristic checks failed — inspect trace above.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
