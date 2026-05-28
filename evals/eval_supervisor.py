"""CLI 入口：运行研究 supervisor 的 LangSmith 评估套件。

用法::

    python -m evals.eval_supervisor
    python -m evals.eval_supervisor --dataset-name my-custom-name
    python -m evals.eval_supervisor --experiment-prefix nightly

前置条件:
  - 已设置 ``LANGSMITH_API_KEY``（或 ``LANGCHAIN_API_KEY``）
  - ``LANGCHAIN_TRACING_V2=true``
  - supervisor（HEAVY 层）和评判（LIGHT 层）的 LLM API 密钥
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from langsmith import Client, evaluate
from loguru import logger

from evals.evaluators import (
    create_reply_quality_evaluator,
    keyword_coverage,
    memory_persistence,
    routing_accuracy,
    tool_selection_precision,
)
from evals.targets import build_eval_environment, supervisor_target
from research_agent.config import get_settings

_DEFAULT_DATASET_NAME = "supervisor-routing-eval"
_DATASET_PATH = Path(__file__).parent / "datasets" / "supervisor_routing.json"


def _ensure_dataset(client: Client, dataset_name: str) -> str:
    """从本地 JSON 文件创建或更新 LangSmith 数据集。

    返回数据集名称（用于 ``evaluate(data=...)``）。
    """
    examples = json.loads(_DATASET_PATH.read_text(encoding="utf-8"))

    try:
        ds = client.read_dataset(dataset_name=dataset_name)
        logger.info("Dataset '{}' already exists (id={})", dataset_name, ds.id)
    except Exception:  # noqa: BLE001
        ds = client.create_dataset(
            dataset_name=dataset_name,
            description=(
                "Research supervisor routing accuracy, reply quality, "
                "and memory persistence evaluation set."
            ),
        )
        logger.info("Created dataset '{}' (id={})", dataset_name, ds.id)

    existing = list(client.list_examples(dataset_id=ds.id))
    if len(existing) >= len(examples):
        logger.info(
            "Dataset already has {} examples (local file has {}); skipping upload.",
            len(existing),
            len(examples),
        )
        return dataset_name

    for ex in examples:
        client.create_example(
            inputs={
                "query": ex["query"],
                "user_id": ex.get("user_id", "anonymous"),
                "expected_specialists": ex.get("expected_specialists", []),
                "expected_reply_keywords": ex.get("expected_reply_keywords", []),
                "category": ex.get("category", ""),
            },
            outputs={},
            dataset_id=ds.id,
        )
    logger.info("Uploaded {} examples to '{}'", len(examples), dataset_name)
    return dataset_name


async def _run_eval(dataset_name: str, experiment_prefix: str) -> None:
    """构建环境，然后启动 ``langsmith.evaluate``。"""
    logger.info("Building evaluation environment (graph + memory)...")
    await build_eval_environment()
    logger.info("Environment ready. Starting evaluation...")

    settings = get_settings()
    reply_quality_evaluator = create_reply_quality_evaluator(
        model=settings.llm.light_model,
        api_key=settings.llm.light_api_key or settings.llm.dashscope_api_key,
        api_base=settings.llm.light_api_base or settings.llm.dashscope_api_base,
    )

    results = evaluate(
        supervisor_target,
        data=dataset_name,
        evaluators=[
            routing_accuracy,
            reply_quality_evaluator,
            keyword_coverage,
            memory_persistence,
            tool_selection_precision,
        ],
        experiment_prefix=experiment_prefix,
        max_concurrency=2,
    )

    logger.info("Evaluation complete. Results URL: {}", getattr(results, "experiment_url", "N/A"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LangSmith evaluation suite for the research supervisor.",
    )
    parser.add_argument(
        "--dataset-name",
        default=_DEFAULT_DATASET_NAME,
        help=f"LangSmith dataset name (default: {_DEFAULT_DATASET_NAME})",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="supervisor-eval",
        help="Experiment prefix shown in LangSmith dashboard (default: supervisor-eval)",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip dataset upload (assume it already exists in LangSmith)",
    )
    args = parser.parse_args()

    client = Client()

    if not args.skip_upload:
        _ensure_dataset(client, args.dataset_name)

    asyncio.run(_run_eval(args.dataset_name, args.experiment_prefix))


if __name__ == "__main__":
    main()
