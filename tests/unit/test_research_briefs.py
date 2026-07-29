"""研究模板目录常量完整性。"""

from __future__ import annotations

from research_agent.graph.research_briefs import (
    CN_BRIEF,
    CN_DEEP,
    CN_MACRO,
    RESEARCH_BRIEF_ROUTER,
    RESEARCH_BRIEF_TEMPLATE,
    US_BRIEF,
    US_DEEP,
    US_MACRO,
)


def test_catalog_covers_cn_us_three_genres() -> None:
    for part in (CN_BRIEF, US_BRIEF, CN_DEEP, US_DEEP, CN_MACRO, US_MACRO, RESEARCH_BRIEF_ROUTER):
        assert part.strip()
        assert part in RESEARCH_BRIEF_TEMPLATE


def test_router_mentions_market_and_priority() -> None:
    assert "CN_A" in RESEARCH_BRIEF_ROUTER
    assert "MIXED" in RESEARCH_BRIEF_ROUTER
    assert "深度 > 大盘板块 > 晨报" in RESEARCH_BRIEF_ROUTER
