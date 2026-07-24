"""本地离线评估运行器 — 不依赖 LangSmith，输出 JSON 报告。

用法::

    # 完整评估（需要 LLM API key + MCP 工具）
    python -m evals.run_local

    # 指定输出目录
    python -m evals.run_local --output-dir eval_results

    # 只跑前 N 条（快速验证）
    python -m evals.run_local --limit 5

    # 跑完后对比两次结果
    python -m evals.compare eval_results/report_A.json eval_results/report_B.json

前置条件:
  - LLM API 密钥（DASHSCOPE_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY）
  - MCP 工具可用（或接受部分 specialist 缺失的降级评估）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from loguru import logger

from evals.datasets import CN_ROUTING_PATH, load_json_dataset, load_merged_routing_dataset
from evals.evaluators import (
    keyword_coverage,
    market_isolation,
    market_routing_accuracy,
    memory_persistence,
    routing_accuracy,
    tool_selection_precision,
)
from evals.targets import build_eval_environment, supervisor_target

_DATASET_PATH = CN_ROUTING_PATH  # 兼容旧 CLI；默认实际加载合并集
_DEFAULT_OUTPUT_DIR = Path("eval_results")

_DETERMINISTIC_EVALUATORS = [
    routing_accuracy,
    keyword_coverage,
    memory_persistence,
    tool_selection_precision,
    market_routing_accuracy,
    market_isolation,
]


def _aggregate(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "count": 0}
    return {
        "mean": round(statistics.mean(scores), 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "std": round(statistics.stdev(scores) if len(scores) > 1 else 0.0, 4),
        "count": len(scores),
    }


async def _run_single(
    idx: int,
    total: int,
    example: dict[str, Any],
) -> dict[str, Any]:
    """Run one example through the supervisor and all deterministic evaluators."""
    query = example["query"]
    logger.info("[{}/{}] Running: {}", idx + 1, total, query[:60])

    record: dict[str, Any] = {
        "query": query,
        "category": example.get("category", ""),
        "user_id": example.get("user_id", "anonymous"),
        "expected_specialists": example.get("expected_specialists", []),
        "expected_reply_keywords": example.get("expected_reply_keywords", []),
        "expected_market": example.get("expected_market"),
    }

    try:
        outputs = await supervisor_target(example)
        record["outputs"] = outputs
        record["error"] = None
    except Exception as exc:  # noqa: BLE001
        logger.error("[{}/{}] Failed: {}", idx + 1, total, exc)
        record["outputs"] = {
            "reply": "",
            "specialists_reached": [],
            "memory_saved": False,
            "thread_id": "",
            "market": "",
            "market_source": "",
        }
        record["error"] = str(exc)

    run_ns = SimpleNamespace(outputs=record["outputs"])
    example_ns = SimpleNamespace(inputs=example)

    scores: dict[str, float] = {}
    comments: dict[str, str] = {}
    for evaluator in _DETERMINISTIC_EVALUATORS:
        result = evaluator(run_ns, example_ns)
        scores[result["key"]] = result["score"]
        comments[result["key"]] = result.get("comment", "")

    record["scores"] = scores
    record["comments"] = comments
    return record


async def run_eval(
    output_dir: Path,
    limit: int | None = None,
    *,
    dataset_path: Path | None = None,
    include_us: bool = True,
) -> Path:
    """Run full local evaluation and write JSON report."""
    if dataset_path is not None:
        examples = load_json_dataset(dataset_path)
        if limit:
            examples = examples[:limit]
        dataset_label = dataset_path.name
    else:
        examples = load_merged_routing_dataset(include_us=include_us, limit=limit)
        dataset_label = "merged_cn_us_mixed_routing" if include_us else CN_ROUTING_PATH.name

    logger.info("Loading dataset: {} examples ({})", len(examples), dataset_label)
    logger.info("Building evaluation environment...")
    await build_eval_environment()
    logger.info("Environment ready. Starting evaluation...")

    details: list[dict] = []
    for idx, example in enumerate(examples):
        record = await _run_single(idx, len(examples), example)
        details.append(record)

    all_scores: dict[str, list[float]] = {}
    by_category: dict[str, dict[str, list[float]]] = {}

    for record in details:
        cat = record["category"]
        for key, score in record["scores"].items():
            all_scores.setdefault(key, []).append(score)
            by_category.setdefault(cat, {}).setdefault(key, []).append(score)

    aggregate = {k: _aggregate(v) for k, v in all_scores.items()}
    category_aggregate = {
        cat: {k: _aggregate(v) for k, v in metrics.items()} for cat, metrics in by_category.items()
    }

    from research_agent.config import get_settings

    settings = get_settings()

    report = {
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "dataset": dataset_label,
            "num_examples": len(examples),
            "num_errors": sum(1 for d in details if d["error"]),
            "model_config": {
                "heavy": settings.llm.heavy_model,
                "medium": settings.llm.medium_model,
                "light": settings.llm.light_model,
            },
        },
        "aggregate": aggregate,
        "by_category": category_aggregate,
        "details": details,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"eval_report_{ts}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_summary(aggregate, category_aggregate, report["metadata"])
    logger.info("Report saved to: {}", report_path)
    return report_path


def _print_summary(
    aggregate: dict,
    by_category: dict,
    metadata: dict,
) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 70)
    print(f"  EVALUATION REPORT — {metadata['timestamp'][:19]}")
    print(f"  Dataset: {metadata['dataset']} ({metadata['num_examples']} examples)")
    print(f"  Errors: {metadata['num_errors']}")
    print(
        f"  Models: heavy={metadata['model_config']['heavy']}, "
        f"medium={metadata['model_config']['medium']}"
    )
    print("=" * 70)

    print(f"\n{'Metric':<28} {'Mean':>8} {'Min':>8} {'Max':>8} {'Std':>8} {'N':>5}")
    print("-" * 65)
    for key, stats in sorted(aggregate.items()):
        print(
            f"  {key:<26} {stats['mean']:>7.3f} {stats['min']:>7.3f} "
            f"{stats['max']:>7.3f} {stats['std']:>7.3f} {stats['count']:>5}"
        )

    header = (
        f"\n{'Category':<20} {'routing_acc':>12} {'keyword_cov':>12} "
        f"{'tool_prec':>10} {'mkt_acc':>8} {'isolate':>8}"
    )
    print(header)
    print("-" * 78)
    for cat in sorted(by_category):
        metrics = by_category[cat]
        ra = metrics.get("routing_accuracy", {}).get("mean", 0)
        kc = metrics.get("keyword_coverage", {}).get("mean", 0)
        tp = metrics.get("tool_selection_precision", {}).get("mean", 0)
        ma = metrics.get("market_routing_accuracy", {}).get("mean", 0)
        mi = metrics.get("market_isolation", {}).get("mean", 0)
        print(f"  {cat:<18} {ra:>11.3f} {kc:>11.3f} {tp:>9.3f} {ma:>7.3f} {mi:>7.3f}")

    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local evaluation suite (no LangSmith required).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to a single dataset JSON; default merges CN + US routing sets",
    )
    parser.add_argument(
        "--cn-only",
        action="store_true",
        help="Only run the legacy A-share supervisor_routing.json set",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Directory for report output (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run first N examples (for quick validation)",
    )
    args = parser.parse_args()

    report_path = asyncio.run(
        run_eval(
            args.output_dir,
            args.limit,
            dataset_path=args.dataset,
            include_us=not args.cn_only,
        )
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
