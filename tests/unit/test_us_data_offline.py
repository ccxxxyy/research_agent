"""``us_data_server`` 离线单元测试 — mock yfinance，无网络。"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from research_agent.cache import reset_tool_cache_for_tests


@pytest.fixture(autouse=True)
def _reset_caches():
    reset_tool_cache_for_tests()
    yield
    reset_tool_cache_for_tests()


def test_normalize_ticker():
    from research_agent.mcp_servers.us_data_server import _normalize_ticker

    assert _normalize_ticker(" aapl ") == "AAPL"
    assert _normalize_ticker("^gspc") == "^GSPC"
    assert _normalize_ticker("SPX") == "^GSPC"


def test_fmt_error():
    from research_agent.mcp_servers.us_data_server import _fmt_error

    err = _fmt_error(ValueError("bad"), context="test")
    assert err["error"] == "ValueError: bad"
    assert err["context"] == "test"


def test_session_status_weekday_open():
    from research_agent.mcp_servers.us_data_server import _session_status

    et = ZoneInfo("America/New_York")
    result = _session_status(now=datetime(2024, 6, 3, 11, 30, tzinfo=et))
    assert result["status"] == "open"
    assert result["timezone"] == "America/New_York"


def test_session_status_weekend():
    from research_agent.mcp_servers.us_data_server import _session_status

    et = ZoneInfo("America/New_York")
    result = _session_status(now=datetime(2024, 6, 1, 12, 0, tzinfo=et))
    assert result["status"] == "closed"
    assert result["session"] == "weekend"
    assert "周末" in result["hint"]


def test_session_status_pre_market():
    from research_agent.mcp_servers.us_data_server import _session_status

    et = ZoneInfo("America/New_York")
    result = _session_status(now=datetime(2024, 6, 3, 8, 0, tzinfo=et))
    assert result["status"] == "pre_market"


@pytest.mark.asyncio
async def test_get_quote_mocked():
    from research_agent.mcp_servers import us_data_server as mod

    mock_fi = MagicMock()
    mock_fi.last_price = 190.5
    mock_fi.previous_close = 188.0
    mock_ticker = MagicMock()
    mock_ticker.fast_info = mock_fi

    # 强制走 yfinance：关掉 Chart + Finnhub + 东财，避免本机网络抢先返回真实行情
    with (
        patch.object(mod, "_quote_via_yahoo_chart", return_value=None),
        patch.object(mod, "_quote_via_finnhub", return_value=None),
        patch.object(mod, "_quote_via_eastmoney_us", return_value=None),
        patch("yfinance.Ticker", return_value=mock_ticker),
    ):
        result = await mod.get_quote("aapl")

    assert "error" not in result
    assert result["symbol"] == "AAPL"
    assert result["price"] == 190.5
    assert result["change_percent"] == pytest.approx(1.3298, rel=1e-3)
    assert result["source"] == "yfinance"


@pytest.mark.asyncio
async def test_get_quote_via_yahoo_chart_mocked():
    from research_agent.mcp_servers import us_data_server as mod

    chart = {
        "symbol": "AAPL",
        "price": 200.0,
        "previous_close": 190.0,
        "change": 10.0,
        "change_percent": 5.2632,
        "source": "yahoo_chart",
    }
    with patch.object(mod, "_quote_via_yahoo_chart", return_value=chart):
        result = await mod.get_quote("aapl")

    assert "error" not in result
    assert result["price"] == 200.0
    assert result["change_percent"] == pytest.approx(5.2632)
    assert result["source"] == "yahoo_chart"


@pytest.mark.asyncio
async def test_get_quote_via_eastmoney_us_mocked():
    from research_agent.mcp_servers import us_data_server as mod

    em = {
        "symbol": "AAPL",
        "price": 333.02,
        "previous_close": 321.66,
        "change": 11.36,
        "change_percent": 3.53,
        "source": "eastmoney_us",
    }
    with (
        patch.object(mod, "_quote_via_yahoo_chart", return_value=None),
        patch.object(mod, "_quote_via_finnhub", return_value=None),
        patch.object(mod, "_quote_via_eastmoney_us", return_value=em),
    ):
        result = await mod.get_quote("aapl")

    assert "error" not in result
    assert result["price"] == 333.02
    assert result["change_percent"] == pytest.approx(3.53)
    assert result["source"] == "eastmoney_us"


@pytest.mark.asyncio
async def test_get_quote_via_finnhub_mocked():
    from research_agent.mcp_servers import us_data_server as mod

    fh = {
        "price": 210.5,
        "previous_close": 200.0,
        "change": 10.5,
        "change_percent": 5.25,
        "source": "finnhub",
    }
    with (
        patch.object(mod, "_quote_via_yahoo_chart", return_value=None),
        patch.object(mod, "_quote_via_finnhub", return_value=fh),
    ):
        result = await mod.get_quote("aapl")

    assert "error" not in result
    assert result["price"] == 210.5
    assert result["change_percent"] == pytest.approx(5.25)
    assert result["source"] == "finnhub"


def test_quote_via_finnhub_parses_http_payload():
    from research_agent.mcp_servers import us_data_server as mod

    payload = {"c": 150.0, "pc": 148.0, "d": 2.0, "dp": 1.3514}
    with (
        patch.object(mod, "_finnhub_api_key", return_value="test-key"),
        patch.object(mod, "_http_get_json", return_value=payload),
    ):
        q = mod._quote_via_finnhub("AAPL")

    assert q is not None
    assert q["price"] == 150.0
    assert q["previous_close"] == 148.0
    assert q["source"] == "finnhub"


def test_quote_via_finnhub_skips_without_key_and_indices():
    from research_agent.mcp_servers import us_data_server as mod

    with patch.object(mod, "_finnhub_api_key", return_value=""):
        assert mod._quote_via_finnhub("AAPL") is None
    with (
        patch.object(mod, "_finnhub_api_key", return_value="test-key"),
        patch.object(mod, "_http_get_json") as http,
    ):
        assert mod._quote_via_finnhub("^GSPC") is None
        assert mod._quote_via_finnhub("CL=F") is None
        http.assert_not_called()


def test_parse_us_quote_providers_default_and_custom(monkeypatch: pytest.MonkeyPatch):
    from research_agent.mcp_servers import us_data_server as mod

    monkeypatch.delenv("US_QUOTE_PROVIDERS", raising=False)
    assert mod._parse_us_quote_providers() == [
        "yahoo_chart",
        "finnhub",
        "eastmoney",
        "yfinance",
    ]

    monkeypatch.setenv("US_QUOTE_PROVIDERS", "eastmoney_us, yfinance, yahoo_chart")
    assert mod._parse_us_quote_providers() == ["eastmoney", "yfinance", "yahoo_chart"]

    monkeypatch.setenv("US_QUOTE_PROVIDERS", "polygon,finnhub,not_a_source")
    assert mod._parse_us_quote_providers() == ["finnhub"]

    monkeypatch.setenv("US_QUOTE_PROVIDERS", "polygon")
    assert mod._parse_us_quote_providers() == list(mod._DEFAULT_US_QUOTE_PROVIDERS)


@pytest.mark.asyncio
async def test_get_quote_respects_us_quote_providers_order(monkeypatch: pytest.MonkeyPatch):
    from research_agent.mcp_servers import us_data_server as mod

    monkeypatch.setenv("US_QUOTE_PROVIDERS", "finnhub,yahoo_chart")
    fh = {
        "price": 210.5,
        "previous_close": 200.0,
        "change": 10.5,
        "change_percent": 5.25,
        "source": "finnhub",
    }
    chart = {
        "symbol": "AAPL",
        "price": 200.0,
        "previous_close": 190.0,
        "change": 10.0,
        "change_percent": 5.2632,
        "source": "yahoo_chart",
    }
    with (
        patch.object(mod, "_quote_via_finnhub", return_value=fh) as fh_fn,
        patch.object(mod, "_quote_via_yahoo_chart", return_value=chart) as chart_fn,
        patch.object(mod, "_quote_via_eastmoney_us") as em_fn,
        patch.object(mod, "_quote_via_yfinance") as yf_fn,
    ):
        result = await mod.get_quote("aapl")

    assert result["source"] == "finnhub"
    assert result["price"] == 210.5
    fh_fn.assert_called()
    chart_fn.assert_not_called()
    em_fn.assert_not_called()
    yf_fn.assert_not_called()


@pytest.mark.asyncio
async def test_get_price_history_mocked():
    from research_agent.mcp_servers import us_data_server as mod

    df = pd.DataFrame(
        {
            "Open": [1.0, 1.1],
            "High": [1.2, 1.3],
            "Low": [0.9, 1.0],
            "Close": [1.1, 1.25],
            "Volume": [100, 200],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = df

    with (
        patch.object(mod, "_history_via_yahoo_chart", return_value=None),
        patch.object(mod, "_history_via_eastmoney", return_value=None),
        patch("yfinance.Ticker", return_value=mock_ticker),
    ):
        result = await mod.get_price_history("AAPL", period="5d")

    assert "error" not in result
    assert result["symbol"] == "AAPL"
    assert len(result["bars"]) == 2
    assert result["summary"]["bars"] == 2


@pytest.mark.asyncio
async def test_get_price_history_rejects_bad_period():
    from research_agent.mcp_servers import us_data_server as mod

    result = await mod.get_price_history("AAPL", period="2d")
    assert "error" in result


@pytest.mark.asyncio
async def test_get_price_history_prefers_yahoo_chart():
    from research_agent.mcp_servers import us_data_server as mod

    chart = {
        "symbol": "^IXIC",
        "period": "5d",
        "interval": "1d",
        "bars": [{"date": "2024-01-02", "close": 15000.0}],
        "summary": {"bars": 1},
        "source": "yahoo_chart",
        "source_url": "https://finance.yahoo.com/quote/IXIC/history",
    }
    yf_ticker = MagicMock()

    with (
        patch.object(mod, "_history_via_yahoo_chart", return_value=chart) as chart_fn,
        patch.object(mod, "_history_via_eastmoney", return_value=None),
        patch("yfinance.Ticker", return_value=yf_ticker) as yf_ctor,
    ):
        result = await mod.get_price_history("^IXIC", period="5d")

    assert result["source"] == "yahoo_chart"
    assert result["symbol"] == "^IXIC"
    chart_fn.assert_called()
    yf_ctor.assert_not_called()


@pytest.mark.asyncio
async def test_get_price_history_falls_back_to_chart():
    from research_agent.mcp_servers import us_data_server as mod

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    fallback = {
        "symbol": "AAPL",
        "period": "5d",
        "interval": "1d",
        "bars": [{"date": "2024-01-02", "close": 100.0}],
        "summary": {"bars": 1},
        "source": "yahoo_chart",
        "source_url": "https://finance.yahoo.com/quote/AAPL/history",
    }

    with (
        patch("yfinance.Ticker", return_value=mock_ticker),
        patch.object(mod, "_history_via_yahoo_chart", return_value=fallback),
        patch.object(mod, "_history_via_eastmoney", return_value=None),
    ):
        result = await mod.get_price_history("AAPL", period="5d")

    assert result["source"] == "yahoo_chart"
    assert len(result["bars"]) == 1


@pytest.mark.asyncio
async def test_search_ticker_mocked():
    from research_agent.mcp_servers import us_data_server as mod

    fake_search = MagicMock()
    fake_search.quotes = [
        {
            "symbol": "AAPL",
            "shortname": "Apple Inc.",
            "exchange": "NMS",
            "quoteType": "EQUITY",
        }
    ]

    with patch("yfinance.Search", return_value=fake_search):
        result = await mod.search_ticker("Apple")

    assert "error" not in result
    assert result["count"] >= 1
    assert result["results"][0]["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_get_etf_overview_mocked():
    from research_agent.mcp_servers import us_data_server as mod

    mock_ticker = MagicMock()
    mock_ticker.info = {
        "shortName": "SPDR S&P 500 ETF Trust",
        "quoteType": "ETF",
        "category": "Large Blend",
        "totalAssets": 500_000_000_000,
        "currency": "USD",
    }

    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = await mod.get_etf_overview("SPY")

    assert "error" not in result
    assert result["etf"]["symbol"] == "SPY"
    assert result["etf"]["quote_type"] == "ETF"


def test_pct_display_and_serialize_helpers():
    from research_agent.mcp_servers.us_data_server import (
        _pct_display,
        _serialize_top_holdings,
        _serialize_weight_map,
    )

    assert _pct_display(0.048) == pytest.approx(4.8)
    assert _pct_display(4.8) == pytest.approx(4.8)
    assert _pct_display(None) is None

    df = pd.DataFrame(
        {"Name": ["Apple Inc", "Microsoft Corp"], "Holding Percent": [0.07, 0.06]},
        index=pd.Index(["AAPL", "MSFT"], name="Symbol"),
    )
    rows = _serialize_top_holdings(df, top_n=1)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["weight_pct"] == pytest.approx(7.0)

    sectors = _serialize_weight_map({"technology": 0.35, "healthcare": 0.12})
    assert sectors[0]["name"] == "technology"
    assert sectors[0]["weight_pct"] == pytest.approx(35.0)


@pytest.mark.asyncio
async def test_get_etf_holdings_mocked():
    from research_agent.mcp_servers import us_data_server as mod

    holdings_df = pd.DataFrame(
        {
            "Name": ["Apple Inc", "Microsoft Corp", "NVIDIA Corp"],
            "Holding Percent": [0.07, 0.065, 0.06],
        },
        index=pd.Index(["AAPL", "MSFT", "NVDA"], name="Symbol"),
    )
    mock_funds = MagicMock()
    mock_funds.top_holdings = holdings_df
    mock_ticker = MagicMock()
    mock_ticker.funds_data = mock_funds

    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = await mod.get_etf_holdings("spy", top_n=2)

    assert "error" not in result
    assert result["symbol"] == "SPY"
    assert result["count"] == 2
    assert result["holdings"][0]["symbol"] == "AAPL"
    assert result["holdings"][0]["weight_pct"] == pytest.approx(7.0)


@pytest.mark.asyncio
async def test_get_etf_holdings_unavailable_funds_data():
    from research_agent.mcp_servers import us_data_server as mod

    mock_ticker = MagicMock()
    mock_ticker.funds_data = None

    with (
        patch("yfinance.Ticker", return_value=mock_ticker),
        patch.object(mod, "_holdings_via_yahoo_quotesummary", return_value=None),
    ):
        result = await mod.get_etf_holdings("AAPL")

    assert "error" in result
    assert result["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_get_etf_holdings_falls_back_to_quotesummary():
    from research_agent.mcp_servers import us_data_server as mod

    mock_ticker = MagicMock()
    mock_ticker.funds_data = None
    fallback = {
        "symbol": "QQQ",
        "holdings": [{"symbol": "NVDA", "name": "NVIDIA", "weight_pct": 8.0}],
        "count": 1,
        "top_n": 10,
        "source": "yahoo_quotesummary",
        "source_url": "https://finance.yahoo.com/quote/QQQ/holdings",
    }

    with (
        patch("yfinance.Ticker", return_value=mock_ticker),
        patch.object(mod, "_holdings_via_yahoo_quotesummary", return_value=fallback),
    ):
        result = await mod.get_etf_holdings("QQQ", top_n=10)

    assert result["source"] == "yahoo_quotesummary"
    assert result["holdings"][0]["symbol"] == "NVDA"


@pytest.mark.asyncio
async def test_get_etf_sector_weights_mocked():
    from research_agent.mcp_servers import us_data_server as mod

    mock_funds = MagicMock()
    mock_funds.sector_weightings = {"technology": 0.4, "financial_services": 0.15}
    mock_funds.asset_classes = {"stockPosition": 0.99, "cashPosition": 0.01}
    mock_ticker = MagicMock()
    mock_ticker.funds_data = mock_funds

    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = await mod.get_etf_sector_weights("QQQ")

    assert "error" not in result
    assert result["symbol"] == "QQQ"
    assert result["sector_count"] == 2
    assert result["sectors"][0]["name"] == "technology"
    assert result["sectors"][0]["weight_pct"] == pytest.approx(40.0)


@pytest.mark.asyncio
async def test_get_mutual_fund_overview_mocked():
    from research_agent.mcp_servers import us_data_server as mod

    mock_ticker = MagicMock()
    mock_ticker.info = {
        "shortName": "Vanguard Total Stock Mkt Idx Adm",
        "quoteType": "MUTUALFUND",
        "fundFamily": "Vanguard",
        "category": "Large Blend",
        "navPrice": 120.5,
        "ytdReturn": 0.12,
        "annualReportExpenseRatio": 0.04,
        "currency": "USD",
    }

    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = await mod.get_mutual_fund_overview("vtsax")

    assert "error" not in result
    assert result["fund"]["symbol"] == "VTSAX"
    assert result["fund"]["quote_type"] == "MUTUALFUND"
    assert result["fund"]["nav_price"] == 120.5


@pytest.mark.asyncio
async def test_get_mutual_fund_holdings_mocked():
    from research_agent.mcp_servers import us_data_server as mod

    holdings_df = pd.DataFrame(
        {"Name": ["Apple Inc"], "Holding Percent": [0.06]},
        index=["AAPL"],
    )
    mock_funds = MagicMock()
    mock_funds.top_holdings = holdings_df
    mock_ticker = MagicMock()
    mock_ticker.funds_data = mock_funds

    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = await mod.get_mutual_fund_holdings("VTSAX", top_n=5)

    assert "error" not in result
    assert result["holdings"][0]["symbol"] == "AAPL"
    assert result["holdings"][0]["weight_pct"] == pytest.approx(6.0)


@pytest.mark.asyncio
async def test_get_futures_quotes_mocked():
    from research_agent.mcp_servers import us_data_server as mod

    with patch.object(
        mod,
        "_quote_from_ticker",
        side_effect=lambda sym: {
            "symbol": sym,
            "name": sym,
            "price": 70.0,
            "previous_close": 69.0,
            "change_percent": 1.45,
            "source": "yahoo_chart",
        },
    ):
        result = await mod.get_futures_quotes("CL=F,GC=F")

    assert result["ok_count"] == 2
    assert {q["symbol"] for q in result["futures"]} == {"CL=F", "GC=F"}


@pytest.mark.asyncio
async def test_get_option_expirations_and_chain_mocked():
    from research_agent.mcp_servers import us_data_server as mod

    calls = pd.DataFrame(
        {
            "contractSymbol": ["AAPL250117C00200000"],
            "strike": [200.0],
            "lastPrice": [5.0],
            "bid": [4.8],
            "ask": [5.2],
            "volume": [100],
            "openInterest": [500],
            "impliedVolatility": [0.25],
            "inTheMoney": [True],
        }
    )
    puts = pd.DataFrame(
        {
            "contractSymbol": ["AAPL250117P00200000"],
            "strike": [200.0],
            "lastPrice": [3.0],
            "bid": [2.8],
            "ask": [3.2],
            "volume": [50],
            "openInterest": [200],
            "impliedVolatility": [0.28],
            "inTheMoney": [False],
        }
    )
    chain = MagicMock()
    chain.calls = calls
    chain.puts = puts
    mock_ticker = MagicMock()
    mock_ticker.options = ["2025-01-17", "2025-02-21"]
    mock_ticker.option_chain.return_value = chain

    with patch("yfinance.Ticker", return_value=mock_ticker):
        exps = await mod.get_option_expirations("AAPL")
        result = await mod.get_option_chain("AAPL", expiration="2025-01-17", limit_per_side=10)

    assert exps["count"] == 2
    assert result["call_count"] == 1
    assert result["put_count"] == 1
    assert result["calls"][0]["strike"] == 200.0


@pytest.mark.asyncio
async def test_get_option_chain_empty_expirations():
    from research_agent.mcp_servers import us_data_server as mod

    mock_ticker = MagicMock()
    mock_ticker.options = []

    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = await mod.get_option_chain("AAPL")

    assert "error" in result
