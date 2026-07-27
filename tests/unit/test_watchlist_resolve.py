"""watchlist_resolve 离线单测。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from research_agent.market import watchlist_resolve as mod


def test_search_cn_code():
    name_df = pd.DataFrame({"code": ["600519"], "name": ["贵州茅台"]})
    with (
        patch.object(mod, "_CN_NAME_CACHE", name_df),
        patch.object(
            mod, "_CN_FUND_CACHE", pd.DataFrame(columns=["基金代码", "基金简称", "基金类型"])
        ),
        patch.object(mod, "_cn_fund_hit", return_value=None),
    ):
        hits = mod.search_cn_watchlist("600519", limit=5)
    assert any(h["symbol"] == "600519" for h in hits)
    assert hits[0]["market"] == "CN_A"
    assert hits[0]["asset_class"] == "equity"
    assert hits[0]["asset_class_zh"] == "股票"


def test_search_cn_fund_by_code():
    fund_df = pd.DataFrame(
        {
            "基金代码": ["008888"],
            "基金简称": ["某测试混合"],
            "基金类型": ["混合型-灵活"],
        }
    )
    with (
        patch.object(mod, "_CN_FUND_CACHE", fund_df),
        patch.object(mod, "_CN_NAME_CACHE", pd.DataFrame({"code": [], "name": []})),
    ):
        hits = mod.search_cn_watchlist("008888", limit=5)
    assert hits[0]["symbol"] == "008888"
    assert hits[0]["name"] == "某测试混合"
    assert hits[0]["asset_class"] == "mutual_fund"
    assert hits[0]["asset_class_zh"] == "场外基金"


def test_search_cn_fund_by_name():
    fund_df = pd.DataFrame(
        {
            "基金代码": ["110022"],
            "基金简称": ["易方达消费行业"],
            "基金类型": ["股票型"],
        }
    )
    with (
        patch.object(mod, "_CN_FUND_CACHE", fund_df),
        patch.object(mod, "_CN_NAME_CACHE", pd.DataFrame({"code": [], "name": []})),
    ):
        hits = mod.search_cn_watchlist("易方达消费", limit=5)
    assert any(h["symbol"] == "110022" and h["asset_class"] == "mutual_fund" for h in hits)


def test_search_cn_futures_rb():
    with patch.object(
        mod, "_CN_FUND_CACHE", pd.DataFrame(columns=["基金代码", "基金简称", "基金类型"])
    ):
        hits = mod.search_cn_watchlist("RB", limit=5)
    assert any(h["symbol"] == "RB0" and h["asset_class"] == "future" for h in hits)
    assert any(h.get("asset_class_zh") == "期货" for h in hits)


def test_search_cn_name_mocked():
    name_df = pd.DataFrame({"code": ["600519", "000001"], "name": ["贵州茅台", "平安银行"]})
    with (
        patch.object(mod, "_CN_NAME_CACHE", name_df),
        patch.object(
            mod, "_CN_FUND_CACHE", pd.DataFrame(columns=["基金代码", "基金简称", "基金类型"])
        ),
    ):
        hits = mod.search_cn_watchlist("茅台", limit=5)
    assert hits[0]["symbol"] == "600519"
    assert "茅台" in hits[0]["name"]


def test_search_us_futures_direct():
    # =F 直接入候选，不依赖外网
    hits = mod.search_us_watchlist("CL=F", limit=5)
    assert any(h["symbol"] == "CL=F" and h["asset_class"] == "future" for h in hits)


def test_search_us_filters_forex_noise():
    """搜 MU 不应把 MUR/USD 等外汇货币对当成股票候选。"""
    fake_quotes = [
        {"symbol": "MU", "shortname": "Micron", "quoteType": "EQUITY"},
        {"symbol": "MURUSD=X", "shortname": "MUR/USD", "quoteType": "CURRENCY"},
        {"symbol": "CHFMUR=X", "shortname": "CHF/MUR", "quoteType": "CURRENCY"},
    ]

    class _FakeSearch:
        def __init__(self, *a, **k):
            self.quotes = fake_quotes

    with (
        patch(
            "research_agent.mcp_servers.us_data_server._quote_from_ticker",
            return_value={"price": 100.0, "change_percent": 1.0, "name": "Micron"},
        ),
        patch("yfinance.Search", _FakeSearch),
    ):
        hits = mod.search_us_watchlist("MU", limit=8)
    syms = [h["symbol"] for h in hits]
    assert "MU" in syms
    assert not any(s.endswith("=X") for s in syms)
    assert hits[0]["symbol"] == "MU"
    assert hits[0]["asset_class_zh"] == "股票"


def test_search_us_ticker_mocked():
    with (
        patch(
            "research_agent.mcp_servers.us_data_server._quote_from_ticker",
            return_value={"price": 190.0, "change_percent": 1.2, "name": "Apple Inc"},
        ),
        patch("yfinance.Search", side_effect=Exception("offline")),
    ):
        hits = mod.search_us_watchlist("AAPL", limit=5)
    assert any(h["symbol"] == "AAPL" for h in hits)


def test_fetch_us_quotes_mocked():
    with patch(
        "research_agent.mcp_servers.us_data_server._quote_from_ticker",
        return_value={"price": 100.0, "change_percent": 1.5, "name": "Apple"},
    ):
        rows = mod.fetch_watchlist_quotes("US", ["AAPL"])
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["change_pct"] == 1.5
    assert rows[0]["price"] == 100.0


def test_fetch_cn_quotes_sina_mocked():
    fake = 'var hq_str_sh600519="贵州茅台,1,1800.00,1850.00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";\n'
    mock_resp = MagicMock()
    mock_resp.text = fake
    mock_resp.encoding = "gbk"
    with patch("requests.get", return_value=mock_resp):
        rows = mod.fetch_watchlist_quotes("CN_A", ["600519"])
    assert rows[0]["symbol"] == "600519"
    assert rows[0]["price"] == 1850.0
    assert rows[0]["change_pct"] == round((1850 - 1800) / 1800 * 100, 2)


def test_search_watchlist_dispatch():
    with patch.object(mod, "search_cn_watchlist", return_value=[{"symbol": "x"}]) as cn:
        assert mod.search_watchlist("CN_A", "q") == [{"symbol": "x"}]
        cn.assert_called_once()
    with patch.object(mod, "search_us_watchlist", return_value=[{"symbol": "y"}]) as us:
        assert mod.search_watchlist("US", "q") == [{"symbol": "y"}]
        us.assert_called_once()
    assert mod.search_watchlist("XX", "q") == []
