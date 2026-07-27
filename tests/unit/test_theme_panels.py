"""A 股看板主题面板规则（离线）。"""

from research_agent.market.theme_panels import (
    build_mainline_themes,
    build_sentiment_benchmark,
    build_speculative_pool,
)


def test_mainline_themes_scores_zt_match():
    concepts = [
        {"code": "BK01", "name": "人工智能", "change_pct": 3.0, "up_count": 20},
        {"code": "BK02", "name": "银行", "change_pct": 5.0, "up_count": 10},
    ]
    zt = [
        {"code": "600001", "name": "甲", "industry": "人工智能", "streak": 2},
        {"code": "600002", "name": "乙", "industry": "人工智能", "streak": 1},
    ]
    themes = build_mainline_themes(concepts, zt)
    assert themes[0]["name"] == "人工智能"
    assert themes[0]["zt_count"] == 2
    assert themes[0]["leaders"]


def test_sentiment_benchmark_prefers_streak_then_seal():
    zt = [
        {"code": "1", "name": "首板大封", "streak": 1, "seal_amount": 9e8, "industry": "电子"},
        {"code": "2", "name": "三板", "streak": 3, "seal_amount": 1e7, "industry": "传媒"},
        {"code": "3", "name": "二板", "streak": 2, "seal_amount": 2e8, "industry": "软件"},
    ]
    items = build_sentiment_benchmark(zt)
    assert items[0]["name"] == "三板"
    assert items[0]["tag"] == "3连板"
    assert items[1]["name"] == "二板"


def test_speculative_pool_rules():
    zt = [
        {"code": "100", "name": "高标", "streak": 4, "open_count": 0, "industry": "传媒"},
        {"code": "101", "name": "分歧", "streak": 2, "open_count": 2, "industry": "软件"},
    ]
    lhb = [
        {
            "code": "200",
            "name": "机构票",
            "comment": "2家机构买入，成功率20%",
            "net_buy": 80_000_000,
            "industry": "电子",
        }
    ]
    changes = [{"code": "300", "name": "拉升", "type": "急速拉升", "industry": "通信"}]
    pool = build_speculative_pool(zt, lhb, changes)
    labels = {x["code"]: x["label"] for x in pool}
    assert labels["100"] == "妖股候选"
    assert labels["101"] == "妖股候选"
    assert labels["200"] == "庄股/机构候选"
    assert labels["300"] == "情绪发散"
