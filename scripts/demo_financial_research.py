"""Phase-4.4 端到端演示 — 研究 supervisor 调度真实 MCP 工具。

本脚本是 **日常生产流程**：它启动两个与 A 股研究相关的 MCP 子进程（``fin_data_server`` 和 ``pdf_report_server``）以及沙箱化的``code_server``，
将它们的工具接入三个专家，编译研究supervisor，然后向图发送一个复合研究问题，该问题经过精心设计以测试多专家路由。

运行::

    uv run python scripts/demo_financial_research.py

前置条件:
  - 网络连接（akshare → 东财/新浪，cninfo 获取披露）。
  - ``.env`` 中已配置可用的 LLM（与 Phase-1 / 3 冒烟测试相同）。

退出码:
    0 → supervisor 的最终回答通过了宽松合理性校验。
    1 → 任何配置 / MCP / LLM 错误，或 supervisor 未路由到预期的专家。

宽松校验（故意不固定 LLM 措辞）：
  * 最终回答非空
  * 最终回答引用了目标公司（名称或代码）
  * supervisor 至少各发出一次 ``transfer_to_data_expert`` 和 ``transfer_to_report_expert``（验证路由到达了两个 MCP 支持的专家，而非只到达一个）。

为何 **不** 直接检查 ``fin_*`` / ``pdf_*`` 调用
------------------------------------------------
``build_research_supervisor`` 使用``output_mode="last_message"`` 编译，这是正确的生产设置（紧凑状态、低 token 开销）。
其结果是每个专家的 *内部*``ToolMessage``/``AIMessage`` 通信保留在其子图中，
不会出现在父级 ``result["messages"]`` 里 — 只有专家的最终摘要消息和supervisor 的 ``transfer_to_*`` 移交记录会冒泡上来。因此通过移交记录验证路由，通过最终回答内容验证正确性。
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

# 复合提示词，强制多跳路由：
#   data_expert   → 基本信息 + 近期价格
#   report_expert → 最近年报/季报披露及关键摘录
#   coder_expert  → （可选）简单衍生统计
# 不告诉 supervisor 使用哪个专家；要点正是让它自行规划移交。
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
    """返回最后一条非工具调用的 assistant 消息。"""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            tc = getattr(m, "tool_calls", None) or []
            if not tc and m.content:
                return str(m.content)
    return ""


def _trace_tool_calls(messages: list) -> list[str]:
    """收集外层 supervisor 状态中可观测到的所有工具名称。

    使用 ``output_mode="last_message"`` 时，此处只会出现supervisor 自身的 ``transfer_to_*`` 调用 — 专家内部的 MCP 工具调用保留在其子图中。

    每个 ``transfer_to_*`` 移交通常产生 **两条** 记录：
      * supervisor 的 ``AIMessage.tool_calls`` 记录，
      * 回声的 ``ToolMessage`` 响应。两者均保留不去重；调用方可自行计数或 ``set()``。
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

    ``transfer_back_to_supervisor``（由 ``langgraph_supervisor``自动注入）被故意排除 — 我们关心的是哪些专家被 *调用* 了，而非 supervisor 重获控制了多少次。
    """
    reached: set[str] = set()
    for n in calls:
        if n.startswith("transfer_to_") and n != "transfer_to_supervisor":
            reached.add(n[len("transfer_to_") :])
    return reached


async def _load_all_tools() -> dict[str, Any]:
    """并行启动三个 MCP 服务器并收集其工具。

    并发运行三个加载器可在冷启动时节省约 1 秒。每次调用``load_*_tools`` 都会启动独立的子进程。
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
            # 递归预算：
            #   supervisor 规划（≥3 次移交）+ 每个专家的
            #   ReAct 循环（≤4 次工具调用）+ supervisor 综合。
            # 50 是舒适的余量。
            config={"recursion_limit": 50},
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Graph invocation crashed: {}", exc)
        return 1

    messages = result["messages"]
    final = _last_plain_assistant(messages)
    calls = _trace_tool_calls(messages)
    reached = _transfers_reached(calls)

    print("\n=== Supervisor 最终回答 ===\n")
    print(final if final else "<空>")
    print("\n=== 跟踪摘要 ===")
    print(f"  消息总数          : {len(messages)}")
    print(f"  已到达的专家      : {sorted(reached) or ['<无>']}")
    print(f"  工具调用事件总数  : {len(calls)}")

    print("\n=== 启发式校验 ===")
    ok_answer = bool(final.strip())
    ok_subject = (COMPANY_NAME in final) or (COMPANY_TICKER in final)
    ok_data = "data_expert" in reached
    ok_report = "report_expert" in reached
    print(f"  最终回答非空              : {ok_answer}")
    print(f"  回答中提及目标公司        : {ok_subject}")
    print(f"  已路由到 data_expert      : {ok_data}")
    print(f"  已路由到 report_expert    : {ok_report}")

    if ok_answer and ok_subject and ok_data and ok_report:
        print("\n  [PASS] Supervisor 路由到了 data_expert + report_expert。")
        return 0
    print("\n  [WARN] 启发式校验失败 — 请检查上方跟踪信息。")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
