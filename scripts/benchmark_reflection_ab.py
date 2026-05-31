"""Reflection A/B 量化对比实验。

对同一组查询分别以 reflection=OFF 和 reflection=ON 跑研究 supervisor，
用批评者模型对两组输出打分，输出 JSON 报告展示质量差异。

用法::

    # 完整测试（需 LLM API Key + MCP 工具可用）
    python scripts/benchmark_reflection_ab.py

    # 只跑前 N 条
    python scripts/benchmark_reflection_ab.py --limit 5

    # 自定义输出目录
    python scripts/benchmark_reflection_ab.py --output-dir eval_results
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from loguru import logger  # noqa: E402

from research_agent.config import get_settings  # noqa: E402
from research_agent.graph.reflection import (  # noqa: E402
    _extract_json,
    _normalise_critique,
)
from research_agent.graph.research_supervisor import build_research_supervisor  # noqa: E402
from research_agent.llm.provider import ModelRouter  # noqa: E402
from research_agent.llm.tier import ModelTier  # noqa: E402
from research_agent.memory.checkpointer import init_checkpointer  # noqa: E402

SAMPLE_QUERIES = [
    "分析宁德时代 2024 年的营收和利润表现",
    "对比隆基绿能和通威股份的 ROE 和毛利率",
    "贵州茅台最近的新闻舆情和市场情绪如何",
    "查一下比亚迪最近的公告披露和股价走势",
    "帮我分析中芯国际的基本面和行业地位",
    "宁德时代最近的舆情量化评分如何",
    "对比宁德时代和比亚迪的营收增速",
    "隆基绿能最新年报里的经营情况讨论",
]


async def _run_single_query(
    graph: Any,
    query: str,
    thread_suffix: str,
) -> dict[str, Any]:
    """通过编译后的图运行单条查询，返回回复和元数据。"""
    import uuid

    thread_id = f"ab-{thread_suffix}-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config=config,
        )
        messages = result.get("messages", [])
        reply = ""
        for msg in reversed(messages):
            if (
                isinstance(msg, AIMessage)
                and not getattr(msg, "tool_calls", None)
                and msg.content
                and isinstance(msg.content, str)
                and msg.content.strip()
            ):
                reply = msg.content
                break

        reflection_meta = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                rm = (msg.additional_kwargs or {}).get("reflection")
                if rm:
                    reflection_meta = rm
                    break

        return {
            "query": query,
            "reply": reply,
            "reply_len": len(reply),
            "message_count": len(messages),
            "thread_id": thread_id,
            "reflection_meta": reflection_meta,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "query": query,
            "reply": "",
            "reply_len": 0,
            "message_count": 0,
            "thread_id": thread_id,
            "reflection_meta": None,
            "error": str(exc),
        }


async def _score_reply(model_router: ModelRouter, query: str, reply: str) -> dict[str, Any]:
    """用 LIGHT 模型作为评审员打分（使用与反思评审相同的提示）。"""
    if not reply.strip():
        return {
            "quality_score": 0.0,
            "reasoning": "empty reply",
            "feedback": "",
            "issues": ["empty"],
        }

    critic_model = model_router.get_model(ModelTier.LIGHT)
    prompt = (
        "你是金融研究回答的质量评审员。对以下回答在五个维度打分：\n"
        "忠实度、引用、完整性、结构、清晰度。\n\n"
        f"## 用户问题\n{query}\n\n"
        f"## 回答\n{reply[:4000]}\n\n"
        '输出一个 JSON 对象：{"quality_score": float 0-1, "reasoning": str, '
        '"feedback": str, "issues": [str]}\n'
        "只输出 JSON，不加其他文字。"
    )
    from langchain_core.messages import HumanMessage as HumanMsg
    from langchain_core.messages import SystemMessage

    try:
        response = await critic_model.ainvoke(
            [
                SystemMessage(content="You are a research quality evaluator."),
                HumanMsg(content=prompt),
            ]
        )
        raw = response.content if isinstance(response.content, str) else str(response.content)
        return _normalise_critique(_extract_json(raw))
    except Exception as exc:  # noqa: BLE001
        return {
            "quality_score": 0.5,
            "reasoning": f"judge error: {exc}",
            "feedback": "",
            "issues": [],
        }


async def run_ab_experiment(
    queries: list[str],
    output_dir: Path,
) -> Path:
    """运行完整的 A/B 实验流程。"""
    settings = get_settings()
    model_router = ModelRouter(settings.llm)
    checkpointer = await init_checkpointer(settings.database.postgres_sync_uri)

    from research_agent.main import _try_build_research_supervisor

    logger.info("Building graph with reflection=OFF...")
    graph_off, roster_off = await _try_build_research_supervisor(
        model_router=model_router,
        checkpointer=checkpointer,
        settings=settings,
    )
    if graph_off is None:
        logger.error("Failed to build graph (OFF). Check MCP/LLM availability.")
        sys.exit(1)

    logger.info("Building graph with reflection=ON...")

    from research_agent.mcp_servers.client_factory import (
        load_code_server_tools,
        load_fin_data_server_tools,
        load_knowledge_tools_inproc,
        load_news_sentiment_server_tools,
        load_news_server_tools,
        load_pdf_report_server_tools,
    )

    timeout = float(getattr(settings, "mcp_tool_discovery_timeout", 30.0))
    results = await asyncio.gather(
        asyncio.wait_for(load_fin_data_server_tools(), timeout=timeout),
        asyncio.wait_for(load_pdf_report_server_tools(), timeout=timeout),
        asyncio.wait_for(load_code_server_tools(), timeout=timeout),
        asyncio.wait_for(load_knowledge_tools_inproc(), timeout=timeout),
        asyncio.wait_for(load_news_server_tools(), timeout=timeout),
        asyncio.wait_for(load_news_sentiment_server_tools(), timeout=timeout),
        return_exceptions=True,
    )
    tool_names = [
        "fin_data_server",
        "pdf_report_server",
        "code_server",
        "knowledge_tools_inproc",
        "news_server",
        "news_sentiment_server",
    ]
    tools: dict[str, list] = {}
    for name, r in zip(tool_names, results, strict=False):
        tools[name] = [] if isinstance(r, Exception) else list(r)

    graph_on = build_research_supervisor(
        model_router=model_router,
        data_tools=tools["fin_data_server"] or None,
        report_tools=tools["pdf_report_server"] or None,
        coder_tools=tools["code_server"] or None,
        knowledge_tools=tools["knowledge_tools_inproc"] or None,
        news_tools=tools["news_server"] or None,
        sentiment_tools=tools["news_sentiment_server"] or None,
        checkpointer=checkpointer,
        enable_reflection=True,
        reflection_pass_threshold=settings.reflection_pass_threshold,
        reflection_max_iterations=settings.reflection_max_iterations,
    )

    results_off: list[dict] = []
    results_on: list[dict] = []

    for i, query in enumerate(queries):
        logger.info("[{}/{}] Query: {}", i + 1, len(queries), query[:50])

        logger.info("  Running OFF...")
        r_off = await _run_single_query(graph_off, query, "off")
        results_off.append(r_off)

        logger.info("  Running ON...")
        r_on = await _run_single_query(graph_on, query, "on")
        results_on.append(r_on)

    logger.info("Scoring replies with LLM judge...")
    comparisons: list[dict] = []
    for r_off, r_on in zip(results_off, results_on, strict=False):
        score_off = await _score_reply(model_router, r_off["query"], r_off["reply"])
        score_on = await _score_reply(model_router, r_on["query"], r_on["reply"])
        comparisons.append(
            {
                "query": r_off["query"],
                "off": {
                    "score": score_off["quality_score"],
                    "reply_len": r_off["reply_len"],
                    "error": r_off["error"],
                    "reasoning": score_off.get("reasoning", ""),
                },
                "on": {
                    "score": score_on["quality_score"],
                    "reply_len": r_on["reply_len"],
                    "error": r_on["error"],
                    "reflection_iterations": (r_on.get("reflection_meta") or {}).get(
                        "iterations_run", 0
                    ),
                    "reflection_final_score": (r_on.get("reflection_meta") or {}).get(
                        "final_score", 0
                    ),
                    "reasoning": score_on.get("reasoning", ""),
                },
                "delta": round(score_on["quality_score"] - score_off["quality_score"], 4),
            }
        )

    off_scores = [c["off"]["score"] for c in comparisons if c["off"]["error"] is None]
    on_scores = [c["on"]["score"] for c in comparisons if c["on"]["error"] is None]

    def _stats(scores: list[float]) -> dict[str, float]:
        if not scores:
            return {"mean": 0, "median": 0, "std": 0, "count": 0}
        return {
            "mean": round(statistics.mean(scores), 4),
            "median": round(statistics.median(scores), 4),
            "std": round(statistics.stdev(scores) if len(scores) > 1 else 0, 4),
            "count": len(scores),
        }

    report = {
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "num_queries": len(queries),
            "model_config": {
                "heavy": settings.llm.heavy_model,
                "medium": settings.llm.medium_model,
                "light": settings.llm.light_model,
            },
            "reflection_config": {
                "pass_threshold": settings.reflection_pass_threshold,
                "max_iterations": settings.reflection_max_iterations,
            },
        },
        "summary": {
            "off": _stats(off_scores),
            "on": _stats(on_scores),
            "mean_delta": round(statistics.mean(on_scores) - statistics.mean(off_scores), 4)
            if off_scores and on_scores
            else 0,
            "improved_count": sum(1 for c in comparisons if c["delta"] > 0.05),
            "degraded_count": sum(1 for c in comparisons if c["delta"] < -0.05),
            "neutral_count": sum(1 for c in comparisons if -0.05 <= c["delta"] <= 0.05),
        },
        "comparisons": comparisons,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"reflection_ab_{ts}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("  Reflection A/B 实验结果")
    print("=" * 60)
    print(f"  查询数: {len(queries)}")
    print(f"  OFF 均分: {report['summary']['off']['mean']:.3f}")
    print(f"  ON  均分: {report['summary']['on']['mean']:.3f}")
    print(f"  平均提升: {report['summary']['mean_delta']:+.3f}")
    print(
        f"  提升/持平/退步: {report['summary']['improved_count']}/{report['summary']['neutral_count']}/{report['summary']['degraded_count']}"
    )
    print(f"  报告: {report_path}")
    print("=" * 60)

    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reflection A/B 量化对比实验")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条查询")
    parser.add_argument("--output-dir", default="eval_results", help="报告输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = SAMPLE_QUERIES[: args.limit] if args.limit else SAMPLE_QUERIES
    output_dir = Path(args.output_dir)
    asyncio.run(run_ab_experiment(queries, output_dir))


if __name__ == "__main__":
    main()
