"""美股看板主题面板聚合（离线）。"""

from research_agent.market.us_theme_panels import (
    build_us_intraday_moves,
    build_us_mainline_themes,
    build_us_sentiment,
    build_us_speculative,
)


def test_us_mainline_scores_member_hits():
    sectors = [{"symbol": "XLK", "name": "科技 (XLK)", "change_pct": 1.0}]
    themes = [{"symbol": "SMH", "name": "半导体 (SMH)", "change_pct": 2.0}]
    gainers = [
        {"symbol": "NVDA", "name": "英伟达", "change_pct": 4.0},
        {"symbol": "AMD", "name": "超威", "change_pct": 3.0},
    ]
    out = build_us_mainline_themes(sectors, themes, gainers, [], [])
    assert out[0]["symbol"] == "SMH"
    assert out[0]["hit_count"] == 2


def test_us_intraday_moves_labels():
    gainers = [{"symbol": "AAA", "name": "A", "change_pct": 8.0}]
    losers = [{"symbol": "BBB", "name": "B", "change_pct": -6.0}]
    out = build_us_intraday_moves(gainers, losers, min_abs_pct=3.0)
    labels = {x["symbol"]: x["label"] for x in out}
    assert labels["AAA"] == "暴涨"
    assert labels["BBB"] == "暴跌"


def test_us_sentiment_near_high():
    actives = [
        {
            "symbol": "AAA",
            "name": "A",
            "change_pct": 1.0,
            "volume": 1e8,
            "price": 99,
            "fifty_two_week_high": 100,
        }
    ]
    out = build_us_sentiment(actives, [], [])
    assert out[0]["near_high"] is True
    assert "近新高" in out[0]["tag"]


def test_us_speculative_rules():
    shorted = [{"symbol": "GME", "name": "GME", "change_pct": 5.0}]
    small = [{"symbol": "XYZ", "name": "XYZ", "change_pct": 12.0}]
    out = build_us_speculative(shorted, small, [])
    labels = {x["symbol"]: x["label"] for x in out}
    assert labels["GME"] == "拥挤空头"
    assert labels["XYZ"] == "小盘投机"
