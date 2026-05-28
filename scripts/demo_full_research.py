"""端到端演示 — 研究 supervisor 调度全部四个专家。

本脚本展示的内容
----------------
``demo_financial_research.py`` 证明了 supervisor 能在 ``data_expert`` + ``report_expert`` + ``coder_expert``（三个MCP-stdio 专家）之间路由。
但它无法测试knowledge_expert — 该专家需要一个已填充的 FAISS collection，
而当时唯一可用的collection 只有冒烟测试灌入的合成 2 页 PDF。

现在 ``scripts/seed_real_research_reports.py`` 已向 ``prod_reports`` collection 灌入了三份真实 A 股年报，
覆盖 AI /半导体价值链（寒武纪 — AI 算力芯片、中际旭创 — CPO/光模块、兆易创新 — 存储芯片），可以用一个研究问题驱动 完整团队的四路扇出：

    ┌─────────────────────────────────────────────────────────┐
    │            research_supervisor   (HEAVY 层)              │
    └─┬─────────────┬───────────────┬───────────────┬─────────┘
      ▼             ▼               ▼               ▼
  data_expert  report_expert  coder_expert  knowledge_expert
   (akshare)    (cninfo)       (sandbox py)    (FAISS RAG)

单轮用户提问要求：
    1) 最新基本面 (data_expert)
    2) 最近 30 天的新公开披露 (report_expert)
    3) 已灌入的知识库 prod_reports 中年报原文摘录 (knowledge_expert)
    4) 在步骤 1 的最近 5 日收盘价上算均值 / 标准差 (coder_expert)

然后由 supervisor 自行撰写 3-5 行的总结。

运行::

    .venv/Scripts/python.exe scripts/demo_full_research.py
    # 可选: --company 宁德时代 --ticker 300750

前置条件:
    * ``scripts/seed_real_research_reports.py`` 至少运行过一次（若请求的标的在 ``prod_reports`` 中不存在或为空，演示会拒绝启动）。
    * 网络连接（akshare / cninfo）。
    * ``.env`` 中已配置可用的 LLM。

退出码:
    0 → supervisor 的最终回答通过了以下四项宽松路由校验。
    1 → 任何配置 / MCP / LLM 错误，或四个专家中有任何一个未被路由到。

宽松启发式校验（故意不固定 LLM 措辞）：
    * 最终回答非空
    * 回答中提及目标公司（名称或 6 位代码）
    * supervisor 至少各发出一次 ``transfer_to_data_expert``、``transfer_to_report_expert``、``transfer_to_coder_expert``和 ``transfer_to_knowledge_expert``。

为何不断言具体工具名
--------------------
``output_mode="last_message"``（在 ``build_research_supervisor``中选用，以保持 token 开销合理）意味着各专家内部的``ToolMessage`` 通信保留在其子图中。
因此外层状态只携带supervisor 的 ``transfer_to_*`` 移交记录和各专家的最终摘要 — 这正是我们所验证的内容。
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
# 强制 stdout/stderr 使用 UTF-8。
#
# supervisor 的最终回答经常包含 emoji 和中日韩字符（✅、❌、表格框线字符）。
# 在 Windows 上默认控制台代码页为 cp936 / GBK，``print(answer_with_emoji)`` 会在脚本运行中途抛出``UnicodeEncodeError`` — 即使 ainvoke 成功也会丢失跟踪摘要。
# 在导入时将包装器重配为 UTF-8 可避免此问题，无需在 shell 中设置 ``PYTHONIOENCODING``。``errors="replace"`` 是安全网，
# 用于包装器无法重配的少数情况（如 stdout 被非文本流捕获）；
# 对无法映射的字符仅打印 '?'。
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

# 默认目标故意对齐已灌入的某个标的。
# 中际旭创 (300308) 是 CPO/光模块龙头，乘 AI 数据中心互联浪潮 — 其 2025 年报在灌入语料库中叙事最丰富（业绩爆发型故事），
# 可为 knowledge_expert 提供可引用的内容，也为 supervisor 在 AI 基础设施主题上提供强有力的多专家演示。
DEFAULT_COMPANY = "中际旭创"
DEFAULT_TICKER = "300308"
DEFAULT_COLLECTION = "prod_reports"


def _question(company: str, ticker: str, collection: str) -> str:
    """构造四路复合研究问题。

    精心设计使每个专家成为某个子任务的显而易见的选择 — supervisor 永远不需要在两个候选之间猜测，
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
    """返回最后一条 **非工具调用** 的 AIMessage。

    使用 ``output_mode="last_message"`` 时，supervisor 的最终综合始终是这样的消息 — 之前的要么是携带 ``tool_calls`` 的移交 ``AIMessage``，要么是专家的中间摘要。
    """
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            tc = getattr(m, "tool_calls", None) or []
            if not tc and m.content:
                return str(m.content)
    return ""


def _trace_tool_calls(messages: list) -> list[str]:
    """收集外层 supervisor 状态中可观测到的所有工具名称。

    每个 ``transfer_to_*`` 通常产生两条记录：一条来自 supervisor 的 ``AIMessage.tool_calls``，一条 ``ToolMessage`` 回声。两者均保留 — 计数由调用方负责。
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
    """返回 supervisor 路由到的不同专家集合。

    ``transfer_back_to_supervisor``（由 ``langgraph_supervisor``自动注入）被故意排除；我们关心的是哪些专家被调用了，而非 supervisor 回收了多少次控制权。
    """
    reached: set[str] = set()
    for n in calls:
        if n.startswith("transfer_to_") and n != "transfer_to_supervisor":
            reached.add(n[len("transfer_to_") :])
    return reached


async def _load_all_tools() -> dict[str, Any]:
    """启动三个 MCP 服务器 + 加载进程内知识库工具。

    三个 MCP 加载器并发运行；知识库加载器在进程内执行（仅导入 + StructuredTool 包装），耗时可忽略。
    仍使用 ``gather`` 以保持对称性 — 如果将来变得昂贵（如提前加载 bge），并发机制已就位。
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
    """如果 ``collection`` 未灌入则拒绝启动演示。

    在空 collection 上运行演示会使 knowledge_expert 始终返回 ``quality='low'`` 并耗尽其三次重试 — 为一个显而易见的操作 失误浪费 LLM 预算。不如提前提示"先运行灌入脚本"。
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
        "Collection {!r} is missing or empty. Run `scripts/seed_real_research_reports.py` first.",
        collection,
    )
    return False


async def amain(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="四专家研究 supervisor 端到端演示。")
    parser.add_argument("--company", default=DEFAULT_COMPANY, help="公司中文名称。")
    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help="6 位 A 股代码（需与 --company 匹配）。",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="knowledge_expert 应搜索的 FAISS collection。",
    )
    parser.add_argument(
        "--no-verify-seed",
        action="store_true",
        help=("跳过 collection 已灌入的前置检查。仅在故意测试 Agent 空知识库处理时使用。"),
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
        # 递归预算需要舒适地覆盖：
        #   * 4 次 supervisor 移交（每个专家一次）
        #   * 每个专家的 ReAct 循环（3-6 次工具调用， knowledge_expert 可能最多 3 次搜索重试）
        #   * supervisor 最终综合
        # 实测 ``deepseek-v3.2`` 约需 25 步即可通过（远低于 75）。
        # 更强/更新的模型（如 ``deepseek-v4-pro``）偶尔会在 coder移交时 ReAct 循环，消耗内部子图的递归配额。
        # 150 提供足够余量，即使专家行为异常也能给出可诊断的异常信息,而非被限制静默截断。
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

    # 将 JSON 产物保存在脚本旁边。下方的控制台打印是"尽力而为"
    # — 在配置不当的 Windows 控制台上，任何单个无法映射的字符（如回答中的 ✅）都会吃掉整个跟踪摘要。保存到磁盘意味着至少可以保留完整的 JSON 记录供诊断。
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

    print("\n=== Supervisor 最终回答 ===\n")
    print(final if final else "<空>")
    print("\n=== 跟踪摘要 ===")
    print(f"  消息总数          : {len(messages)}")
    print(f"  已到达的专家      : {sorted(reached) or ['<无>']}")
    print(f"  工具调用事件总数  : {len(calls)}")
    print(f"  记录已保存至      : {transcript_path.name}")

    print("\n=== 启发式校验 ===")
    ok_answer = bool(final.strip())
    ok_subject = (args.company in final) or (args.ticker in final)
    ok_data = "data_expert" in reached
    ok_report = "report_expert" in reached
    ok_coder = "coder_expert" in reached
    ok_knowledge = "knowledge_expert" in reached
    print(f"  最终回答非空              : {ok_answer}")
    print(f"  回答中提及目标公司        : {ok_subject}")
    print(f"  已路由到 data_expert      : {ok_data}")
    print(f"  已路由到 report_expert    : {ok_report}")
    print(f"  已路由到 coder_expert     : {ok_coder}")
    print(f"  已路由到 knowledge_expert : {ok_knowledge}")

    if all((ok_answer, ok_subject, ok_data, ok_report, ok_coder, ok_knowledge)):
        print("\n  [PASS] Supervisor 路由到全部四个专家，并产生了非空且切题的回答。")
        return 0
    print(
        "\n  [WARN] 启发式校验失败 — 请检查上方跟踪信息。"
        "最常见原因：supervisor 认为某个子任务已被另一专家的摘要'覆盖'而跳过了移交。"
    )
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(amain(sys.argv[1:])))
