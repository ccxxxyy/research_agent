"""MIXED 评估集契约测试 — 无 LLM / 无网络。"""

from __future__ import annotations

import pytest

from evals.datasets import MIXED_ROUTING_PATH, load_json_dataset, load_merged_routing_dataset
from evals.evaluators import _normalize_market_label


def test_mixed_dataset_nonempty() -> None:
    examples = load_json_dataset(MIXED_ROUTING_PATH)
    assert len(examples) >= 5


def test_merged_includes_mixed() -> None:
    merged = load_merged_routing_dataset()
    assert any(ex.get("category", "").startswith("mixed_") for ex in merged)


@pytest.mark.parametrize("example", load_json_dataset(MIXED_ROUTING_PATH))
def test_mixed_example_schema(example: dict) -> None:
    assert example.get("query")
    assert example.get("expected_market") == "MIXED"
    assert isinstance(example.get("expected_specialists"), list)
    specs = set(example["expected_specialists"])
    # 至少一侧 CN、一侧 US（平行隔离）
    cn = specs & {
        "data_expert",
        "news_expert",
        "report_expert",
        "fund_expert",
        "sentiment_expert",
    }
    us = specs & {
        "us_data_expert",
        "us_filing_expert",
        "us_news_expert",
        "us_sentiment_expert",
    }
    assert cn, f"缺少 A 股专家: {example['query']}"
    assert us, f"缺少美股专家: {example['query']}"


@pytest.mark.asyncio
@pytest.mark.parametrize("example", load_json_dataset(MIXED_ROUTING_PATH))
async def test_mixed_expected_market_matches_resolver(example: dict) -> None:
    from research_agent.market import resolve_market

    resolution = await resolve_market(example["query"], memory=None, user_id="anonymous")
    expected = _normalize_market_label(example["expected_market"])
    actual = _normalize_market_label(resolution.market.value)
    assert actual == expected
