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

    mock_ticker = MagicMock()
    mock_ticker.info = {
        "shortName": "Apple Inc.",
        "regularMarketPrice": 190.5,
        "previousClose": 188.0,
        "currency": "USD",
        "quoteType": "EQUITY",
        "exchange": "NMS",
    }
    mock_ticker.fast_info = {"last_price": 190.5, "previous_close": 188.0, "currency": "USD"}

    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = await mod.get_quote("aapl")

    assert "error" not in result
    assert result["symbol"] == "AAPL"
    assert result["price"] == 190.5
    assert result["change_percent"] == pytest.approx(1.3298, rel=1e-3)


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

    with patch("yfinance.Ticker", return_value=mock_ticker):
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
