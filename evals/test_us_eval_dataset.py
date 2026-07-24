"""美股评估集契约测试 — 无 LLM / 无网络。

验证：
1. JSON schema 字段完整；
2. ``expected_market`` 与 ``resolve_market`` 判定一致（误标样本会在 CI 失败）。
"""

from __future__ import annotations

import pytest

from evals.datasets import (
    MIXED_ROUTING_PATH,
    US_ROUTING_PATH,
    load_json_dataset,
    load_merged_routing_dataset,
)
from evals.evaluators import _normalize_market_label


def test_us_dataset_exists_and_nonempty() -> None:
    examples = load_json_dataset(US_ROUTING_PATH)
    assert len(examples) >= 20


def test_mixed_dataset_exists_and_nonempty() -> None:
    examples = load_json_dataset(MIXED_ROUTING_PATH)
    assert len(examples) >= 5


def test_merged_dataset_includes_us_and_mixed() -> None:
    merged = load_merged_routing_dataset()
    us_only = load_json_dataset(US_ROUTING_PATH)
    mixed_only = load_json_dataset(MIXED_ROUTING_PATH)
    assert len(merged) >= len(us_only) + len(mixed_only) + 100
    assert any(ex.get("category", "").startswith("us_") for ex in merged)
    assert any(str(ex.get("category", "")).startswith("mixed_") for ex in merged)


@pytest.mark.parametrize("example", load_json_dataset(US_ROUTING_PATH))
def test_us_example_schema(example: dict) -> None:
    assert example.get("query")
    assert isinstance(example.get("expected_specialists"), list)
    assert example["expected_specialists"]
    assert example.get("expected_market") in {"US", "CN_A"}
    assert example.get("category")
    assert isinstance(example.get("expected_reply_keywords"), list)


@pytest.mark.asyncio
@pytest.mark.parametrize("example", load_json_dataset(US_ROUTING_PATH))
async def test_us_expected_market_matches_resolver(example: dict) -> None:
    from research_agent.market import resolve_market

    resolution = await resolve_market(example["query"], memory=None, user_id="anonymous")
    expected = _normalize_market_label(example["expected_market"])
    actual = _normalize_market_label(resolution.market.value)
    assert actual == expected, (
        f"query={example['query']!r} expected_market={expected} resolved={actual} "
        f"source={resolution.source} reasons={resolution.reasons}"
    )


@pytest.mark.parametrize("example", load_json_dataset(MIXED_ROUTING_PATH))
def test_mixed_example_schema(example: dict) -> None:
    assert example.get("query")
    assert isinstance(example.get("expected_specialists"), list)
    assert example["expected_specialists"]
    assert example.get("expected_market") == "MIXED"
    assert str(example.get("category", "")).startswith("mixed_")


@pytest.mark.asyncio
@pytest.mark.parametrize("example", load_json_dataset(MIXED_ROUTING_PATH))
async def test_mixed_expected_market_matches_resolver(example: dict) -> None:
    from research_agent.market import resolve_market

    resolution = await resolve_market(example["query"], memory=None, user_id="anonymous")
    expected = _normalize_market_label(example["expected_market"])
    actual = _normalize_market_label(resolution.market.value)
    assert actual == expected, (
        f"query={example['query']!r} expected_market={expected} resolved={actual} "
        f"source={resolution.source} reasons={resolution.reasons}"
    )
