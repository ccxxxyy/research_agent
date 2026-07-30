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


def test_macro_gap_forbids_unrequested_sentiment() -> None:
    assert "禁止" in RESEARCH_BRIEF_ROUTER and "sentiment_expert" in RESEARCH_BRIEF_ROUTER
    assert "万得" in RESEARCH_BRIEF_ROUTER or "同花顺" in RESEARCH_BRIEF_ROUTER
    assert "自选股" in CN_MACRO or "蓝筹" in CN_MACRO
    assert "目标价" in RESEARCH_BRIEF_ROUTER
    assert "eastmoney_flash_fallback" in RESEARCH_BRIEF_ROUTER
    assert "应收账款" in RESEARCH_BRIEF_ROUTER or "中报 PDF" in RESEARCH_BRIEF_ROUTER
    assert "北向资金个股流向" in RESEARCH_BRIEF_ROUTER or "个股北向" in RESEARCH_BRIEF_ROUTER
    assert (
        "巨潮公告 PDF 未提取" in RESEARCH_BRIEF_ROUTER or "巨潮 PDF 未提取" in RESEARCH_BRIEF_ROUTER
    )
