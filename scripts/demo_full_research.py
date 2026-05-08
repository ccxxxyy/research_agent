"""Phase-4.6 end-to-end demo — research supervisor over ALL four specialists.

What this script demonstrates
-----------------------------
``demo_financial_research.py`` proved that the supervisor could route
across ``data_expert`` + ``report_expert`` + ``coder_expert`` (the
three MCP-stdio specialists). What it could NOT exercise was the
**knowledge_expert** — that specialist needs a populated FAISS
collection, and at the time the only collection was the synthetic
2-page PDF the smoke-test ingested.

Now that ``scripts/seed_real_research_reports.py`` has populated the
``prod_reports`` collection with three real A-share annual reports
spanning the AI / semiconductor value chain (寒武纪 — AI 算力芯片,
中际旭创 — CPO/光模块, 兆易创新 — 存储芯片), we can put the FULL team
under one research question that forces a four-way fan-out:

    ┌─────────────────────────────────────────────────────────┐
    │            research_supervisor   (HEAVY tier)           │
    └─┬─────────────┬───────────────┬───────────────┬─────────┘
      ▼             ▼               ▼               ▼
  data_expert  report_expert  coder_expert  knowledge_expert
   (akshare)    (cninfo)       (sandbox py)    (FAISS RAG)

The single user turn asks for:
    1) 最新基本面 (data_expert)
    2) 最近 30 天的新公开披露 (report_expert)
    3) 已灌入的知识库里 prod_reports 中年报原文摘录 (knowledge_expert)
    4) 在步骤 1 的最近 5 日收盘价上算均值 / 标准差 (coder_expert)

then a 3-5 行的总结 written by the supervisor itself.

Run::

    .venv/Scripts/python.exe scripts/demo_full_research.py
    # optional: --company 宁德时代 --ticker 300750

Prerequisites:
    * ``scripts/seed_real_research_reports.py`` has been run at least
      once (the demo will refuse to start if ``prod_reports`` is
      missing or empty for the requested ticker).
    * Network access (akshare / cninfo).
    * A working LLM config in ``.env``.

Exit codes:
    0 → supervisor produced a final answer that passes the four soft
        routing checks below.
    1 → any configuration / MCP / LLM error, or one of the four
        specialists was NOT routed to.

Soft heuristic checks (intentionally lenient — we never pin LLM wording):
    * non-empty final answer
    * answer mentions the target company (name OR 6-digit ticker)
    * supervisor issued ``transfer_to_data_expert`` AND
      ``transfer_to_report_expert`` AND ``transfer_to_coder_expert``
      AND ``transfer_to_knowledge_expert`` at least once each.

Why we don't assert specific tool names
---------------------------------------
``output_mode="last_message"`` (chosen in
``build_research_supervisor`` to keep token-cost realistic) means the
specialists' inner ``ToolMessage`` traffic stays in their subgraphs.
The outer state therefore only carries the supervisor's
``transfer_to_*`` hand-offs and each specialist's final summary —
that's exactly what we verify.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------
# Force UTF-8 on stdout/stderr.
#
# The supervisor's final answer routinely contains emoji and CJK
# characters (✅, ❌, 表格 frame chars). On Windows the default
# console code page is cp936 / GBK, and ``print(answer_with_emoji)``
# raises ``UnicodeEncodeError`` mid-script — we lose the trace
# summary even though the ainvoke succeeded. Reconfiguring the
# wrappers to UTF-8 at import time avoids that without forcing the
# operator to set ``PYTHONIOENCODING`` in their shell. ``errors=
# "replace"`` is a safety net for the rare case where the wrapper
# can't be reconfigured (e.g. when stdout is captured by a non-text
# stream); we still keep going and just print '?' for unmappable
# chars.
# ---------------------------------------------------------------------
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from loguru import logger  # noqa: E402

from research_agent.config import get_settings
from research_agent.graph.research_supervisor import build_research_supervisor
from research_agent.llm.provider import ModelRouter
from research_agent.mcp_servers.client_factory import (
    load_code_server_tools,
    load_fin_data_server_tools,
    load_knowledge_tools_inproc,
    load_pdf_report_server_tools,
)


# Default target intentionally aligns with one of the seeded tickers.
# 中际旭创 (300308) is the CPO/光模块 龙头, riding the AI data-centre
# interconnect wave — its 2025 年报 is the most narrative-rich entry
# in the seeded corpus (业绩爆发型故事), which gives knowledge_expert
# something quotable to retrieve and gives the supervisor a strong
# multi-specialist demo on the AI infra theme.
DEFAULT_COMPANY = "中际旭创"
DEFAULT_TICKER = "300308"
DEFAULT_COLLECTION = "prod_reports"


def _question(company: str, ticker: str, collection: str) -> str:
    """Compose the composite four-way research question.

    Crafted to make each specialist the OBVIOUS choice for one
    sub-task — the supervisor should never need to guess between two
    candidates, which keeps the demo deterministic enough for an
    interview walk-through.
    """
    return (
        f"我想对 {company}（股票代码 {ticker}）做一份小型研究简报。\n"
        f"请按照以下四个步骤完成，并在最终答复用四级小标题分隔：\n"
        f"  1) 【最新基本面】给出公司的所属行业、最新收盘价、市值，"
        f"     并附最近 5 个交易日的收盘价；\n"
        f"  2) 【最近披露】从 巨潮资讯 检索该公司最近 30 天内的 1 份"
        f"     新公开披露（标题、披露日期、PDF 直链即可，不必下载）；\n"
        f"  3) 【已入库年报】我已经把该公司最近一份年报灌进了知识库 "
        f"     `{collection}`，请用 1-2 段从中摘录关于 经营情况讨论 "
        f"     或 主要财务指标 的原文（每段 <= 200 字，标注 source 与 page）；\n"
        f"  4) 【数据计算】把步骤 1 返回的 5 日收盘价交给代码专家，"
        f"     计算均值与样本标准差（保留两位小数）。\n"
        f"最后写一段 3-5 行的整体总结。"
    )


def _last_plain_assistant(messages: list) -> str:
    """Return the last AIMessage that is NOT itself a tool call.

    With ``output_mode="last_message"`` the supervisor's final
    synthesis is always such a message — anything before it is
    either a hand-off ``AIMessage`` (carries ``tool_calls``) or a
    specialist's interim summary.
    """
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            tc = getattr(m, "tool_calls", None) or []
            if not tc and m.content:
                return str(m.content)
    return ""


def _trace_tool_calls(messages: list) -> list[str]:
    """Collect every tool name observable in the outer supervisor state.

    Each ``transfer_to_*`` typically produces TWO entries:
    one ``AIMessage.tool_calls`` from the supervisor and one
    ``ToolMessage`` echo. We keep both — counting is the caller's job.
    """
    names: list[str] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            nm = getattr(m, "name", None) or ""
            if nm:
                names.append(nm)
        elif isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                nm = (
                    tc.get("name")
                    if isinstance(tc, dict)
                    else getattr(tc, "name", None)
                )
                if nm:
                    names.append(str(nm))
    return names


def _transfers_reached(calls: list[str]) -> set[str]:
    """Return the set of distinct specialists the supervisor routed to.

    ``transfer_back_to_supervisor`` (auto-injected by
    ``langgraph_supervisor``) is intentionally stripped; we care about
    which specialists were *invoked*, not how many supervisor turns
    happened.
    """
    reached: set[str] = set()
    for n in calls:
        if n.startswith("transfer_to_") and n != "transfer_to_supervisor":
            reached.add(n[len("transfer_to_") :])
    return reached


async def _load_all_tools() -> dict[str, Any]:
    """Spawn the three MCP servers + load in-process knowledge tools.

    The three MCP loaders run concurrently; the knowledge loader is
    in-process (just imports + StructuredTool wrapping) so it
    contributes negligible wall-time. We still ``gather`` it for
    symmetry — if it ever grows expensive (e.g. eager bge load) the
    parallelism is already in place.
    """
    data_tools, report_tools, coder_tools, knowledge_tools = await asyncio.gather(
        load_fin_data_server_tools(),
        load_pdf_report_server_tools(),
        load_code_server_tools(),
        load_knowledge_tools_inproc(),
    )
    logger.info(
        "Tools loaded: data={} report={} coder={} knowledge={}",
        len(data_tools),
        len(report_tools),
        len(coder_tools),
        len(knowledge_tools),
    )
    return {
        "data_tools": data_tools,
        "report_tools": report_tools,
        "coder_tools": coder_tools,
        "knowledge_tools": knowledge_tools,
    }


async def _verify_collection_seeded(collection: str) -> bool:
    """Refuse to start the demo if ``collection`` is unseeded.

    Running the demo against an empty collection would make the
    knowledge_expert always return ``quality='low'`` and exhaust its
    three retries on every turn — wasting LLM budget for an obvious
    operator error. Far better to surface "run the seed script
    first" up front.
    """
    from research_agent.mcp_servers.knowledge_server import list_collections

    listing = await list_collections()
    for c in listing.get("collections", []):
        if c.get("name") == collection and c.get("chunk_count", 0) > 0:
            logger.info(
                "Collection {!r} present with {} chunks.",
                collection,
                c["chunk_count"],
            )
            return True
    logger.error(
        "Collection {!r} is missing or empty. Run "
        "`scripts/seed_real_research_reports.py` first.",
        collection,
    )
    return False


async def amain(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end demo of the four-specialist research supervisor."
    )
    parser.add_argument(
        "--company", default=DEFAULT_COMPANY, help="Company name in Chinese."
    )
    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help="6-digit A-share ticker (must match --company).",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="FAISS collection the knowledge_expert should target.",
    )
    parser.add_argument(
        "--no-verify-seed",
        action="store_true",
        help=(
            "Skip the up-front check that the collection has been "
            "seeded. Use only when intentionally exercising the "
            "agent's empty-library handling."
        ),
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    router = ModelRouter(settings.llm)

    if not args.no_verify_seed:
        ok = await _verify_collection_seeded(args.collection)
        if not ok:
            return 1

    try:
        loaded = await _load_all_tools()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load tools: {}", exc)
        return 1

    graph = build_research_supervisor(model_router=router, **loaded)

    question = _question(args.company, args.ticker, args.collection)
    logger.info("Sending composite four-way research question:\n{}", question)

    try:
        # Recursion budget needs to comfortably cover:
        #   * 4 supervisor hand-offs (one per specialist)
        #   * each specialist's ReAct loop (3-6 tool calls,
        #     knowledge_expert can take up to 3 search retries)
        #   * supervisor final synthesis
        # Empirically the run needs ~25 supervisor steps with
        # ``deepseek-v3.2`` (PASSes well below 75). Stronger /
        # newer models (e.g. ``deepseek-v4-pro``) occasionally
        # ReAct-loop on the coder hand-off, consuming the inner
        # subgraph's slice of the recursion budget. 150 buys
        # enough headroom that even a misbehaving specialist
        # will be visibly diagnosed (we surface the exception
        # text) rather than silently truncated by the limit.
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": 150},
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Graph invocation crashed: {}", exc)
        return 1

    messages = result["messages"]
    final = _last_plain_assistant(messages)
    calls = _trace_tool_calls(messages)
    reached = _transfers_reached(calls)

    # Persist a JSON artifact alongside the script. The console print
    # below is "best effort" — on a misconfigured Windows console
    # any single un-mappable character (e.g. ✅ in the answer) would
    # otherwise eat the whole trace summary. Saving to disk means
    # the interview material survives even when the print fails.
    transcript_path = (
        Path(__file__).resolve().parent
        / f"demo_full_research.last.{datetime.now():%Y%m%d-%H%M%S}.json"
    )
    transcript_path.write_text(
        json.dumps(
            {
                "company": args.company,
                "ticker": args.ticker,
                "collection": args.collection,
                "question": question,
                "final_answer": final,
                "tool_call_events": calls,
                "specialists_reached": sorted(reached),
                "total_messages": len(messages),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Transcript JSON written to {}", transcript_path)

    print("\n=== Final supervisor answer ===\n")
    print(final if final else "<empty>")
    print("\n=== Trace summary ===")
    print(f"  total messages        : {len(messages)}")
    print(f"  specialists reached   : {sorted(reached) or ['<none>']}")
    print(f"  total tool-name events: {len(calls)}")
    print(f"  transcript saved to   : {transcript_path.name}")

    print("\n=== Heuristic verification ===")
    ok_answer = bool(final.strip())
    ok_subject = (args.company in final) or (args.ticker in final)
    ok_data = "data_expert" in reached
    ok_report = "report_expert" in reached
    ok_coder = "coder_expert" in reached
    ok_knowledge = "knowledge_expert" in reached
    print(f"  non-empty final answer        : {ok_answer}")
    print(f"  company mentioned in answer   : {ok_subject}")
    print(f"  data_expert routed to         : {ok_data}")
    print(f"  report_expert routed to       : {ok_report}")
    print(f"  coder_expert routed to        : {ok_coder}")
    print(f"  knowledge_expert routed to    : {ok_knowledge}")

    if all((ok_answer, ok_subject, ok_data, ok_report, ok_coder, ok_knowledge)):
        print(
            "\n  [PASS] Supervisor routed to all four specialists "
            "and produced a non-empty, on-topic answer."
        )
        return 0
    print(
        "\n  [WARN] Heuristic checks failed — inspect trace above. "
        "Most common cause: the supervisor decided one of the "
        "sub-tasks was already 'covered' by another specialist's "
        "summary and skipped the hand-off."
    )
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(amain(sys.argv[1:])))
