"""dashboard_extras 离线单测（动态双榜）。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd


def test_fetch_cn_futures_panel_ranked():
    from research_agent.market import dashboard_extras as mod

    catalog = pd.DataFrame(
        {
            "symbol": ["RB0", "AU0", "M0"],
            "exchange": ["shfe", "shfe", "dce"],
            "name": ["螺纹钢连续", "黄金连续", "豆粕连续"],
        }
    )
    spot = pd.DataFrame(
        {
            "symbol": ["螺纹钢连续", "黄金连续", "豆粕连续"],
            "current_price": [3000.0, 500.0, 2800.0],
            "last_settle_price": [2950.0, 510.0, 2800.0],
            "last_close": [2950.0, 510.0, 2800.0],
            "volume": [1000, 5000, 200],
        }
    )

    with (
        patch("akshare.futures_display_main_sina", return_value=catalog),
        patch("akshare.futures_zh_spot", return_value=spot),
    ):
        out = mod.fetch_cn_futures_panel(limit=2)

    assert out["limit"] == 2
    assert out["by_volume"][0]["code"] == "AU"
    assert out["by_volume"][0]["volume"] == 5000
    assert out["by_volume"][0]["name"] == "黄金"
    assert out["by_change"][0]["code"] == "RB"
    assert out["by_change"][0]["change_pct"] > 0
    assert out["by_change"][0]["name"] == "螺纹钢"


def test_fetch_cn_etf_panel_dual():
    from research_agent.market import dashboard_extras as mod

    mock_df = pd.DataFrame(
        {
            "代码": ["510300", "159915"],
            "名称": ["沪深300ETF", "创业板ETF"],
            "最新价": [4.5, 2.1],
            "涨跌幅": [1.0, 3.0],
            "成交额": [1e9, 2e9],
            "成交量": [1e7, 2e7],
        }
    )
    with patch(
        "research_agent.mcp_servers.fund_server._fetch_sina_etf_realtime",
        return_value=mock_df,
    ):
        out = mod.fetch_cn_etf_panel(limit=2)
    assert out["by_volume"][0]["code"] == "159915"
    assert out["by_change"][0]["code"] == "159915"


def test_fetch_cn_qdii_panel_change_only():
    from research_agent.market import dashboard_extras as mod

    mock_df = pd.DataFrame(
        {
            "基金代码": ["000041", "000043"],
            "基金简称": ["华夏全球", "嘉实海外"],
            "单位净值": [1.2, 1.1],
            "日增长率": [0.8, 1.5],
            "近1年": [15.0, 20.0],
        }
    )
    with patch("akshare.fund_open_fund_rank_em", return_value=mock_df):
        out = mod.fetch_cn_qdii_panel(limit=2)
    assert out["by_volume"] == []
    assert out["by_change"][0]["code"] == "000043"
    assert out["by_change"][0]["change_pct"] == 1.5


def test_fetch_us_futures_panel_ranked():
    from research_agent.market import dashboard_extras as mod

    quotes = {
        "CL=F": {"price": 70.0, "change_percent": -1.0, "source": "yahoo"},
        "GC=F": {"price": 2000.0, "change_percent": 2.5, "source": "yahoo"},
        "ES=F": {"price": 5000.0, "change_percent": 0.5, "source": "yahoo"},
    }

    def _q(sym: str):
        return quotes.get(sym, {"price": None})

    with (
        patch(
            "research_agent.mcp_servers.us_data_server._quote_from_ticker",
            side_effect=_q,
        ),
        patch("yfinance.download", return_value=pd.DataFrame()),
        patch.object(
            mod,
            "_US_FUTURES_UNIVERSE",
            (("CL=F", "WTI"), ("GC=F", "黄金"), ("ES=F", "标普")),
        ),
    ):
        out = mod.fetch_us_futures_panel(limit=2)

    assert len(out["by_change"]) == 2
    assert out["by_change"][0]["code"] == "GC=F"
    assert out["by_volume"] == []


def test_fetch_us_etf_rank_panel():
    from research_agent.market import dashboard_extras as mod

    def _q(sym: str):
        return {"price": 100.0, "change_percent": 1.0 if sym == "QQQ" else 0.2, "source": "yahoo"}

    with (
        patch(
            "research_agent.mcp_servers.us_data_server._quote_from_ticker",
            side_effect=_q,
        ),
        patch("yfinance.download", return_value=pd.DataFrame()),
        patch.object(mod, "_US_ETF_UNIVERSE", (("SPY", "标普"), ("QQQ", "纳指"))),
    ):
        out = mod.fetch_us_etf_rank_panel(limit=2)
    assert out["by_change"][0]["code"] == "QQQ"


def test_fetch_us_mutual_funds_panel_ytd():
    from research_agent.market import dashboard_extras as mod

    mock_ticker = MagicMock()
    mock_ticker.info = {
        "shortName": "Vanguard Total Stock",
        "navPrice": 120.5,
        "ytdReturn": 0.12,
    }
    with (
        patch("yfinance.Ticker", return_value=mock_ticker),
        patch.object(mod, "_US_MUTUAL_FUNDS_UNIVERSE", (("VTSAX", "Vanguard全市场"),)),
    ):
        out = mod.fetch_us_mutual_funds_panel(limit=5)
    assert out["by_change"][0]["price"] == 120.5
    assert out["by_change"][0]["change_pct"] == 12.0
    assert out["by_volume"] == []
