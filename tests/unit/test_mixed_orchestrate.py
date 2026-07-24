"""MIXED 编排计划单元测试（P5）。"""

from __future__ import annotations

from research_agent.market import (
    Market,
    build_mixed_orchestration_plan,
    detect_market_from_query,
    parse_market_override,
)


def test_parse_market_override_accepts_mixed() -> None:
    assert parse_market_override("MIXED") is Market.MIXED
    assert parse_market_override("auto") is None
    assert parse_market_override("US") is Market.US


def test_build_plan_none_for_single_market() -> None:
    r = detect_market_from_query("特斯拉股价")
    assert r.market == Market.US
    assert build_mixed_orchestration_plan(r, "特斯拉股价") is None


def test_build_plan_compare_quote() -> None:
    q = "对比宁德时代和特斯拉最近的股价表现"
    r = detect_market_from_query(q)
    plan = build_mixed_orchestration_plan(r, q)
    assert plan is not None
    assert plan.is_comparison is True
    sides = {t.side for t in plan.subtasks}
    assert Market.CN_A in sides and Market.US in sides
    assert any("data_expert" in t.preferred_experts for t in plan.subtasks)
    assert any("us_data_expert" in t.preferred_experts for t in plan.subtasks)
    text = plan.format_for_prompt()
    assert text.startswith("[MixedOrchestration]")
    assert "synthesis=" in text


def test_build_plan_filing_intent() -> None:
    q = "贵州茅台年报和 Apple 的 10-K"
    r = detect_market_from_query(q)
    plan = build_mixed_orchestration_plan(r, q)
    assert plan is not None
    assert any(t.intent == "filing" for t in plan.subtasks)
