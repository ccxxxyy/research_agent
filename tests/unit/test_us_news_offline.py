"""``us_news_server`` 离线单元测试 — mock yfinance / SEC HTTP。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_agent.cache import reset_tool_cache_for_tests


@pytest.fixture(autouse=True)
def _reset_caches():
    from research_agent.mcp_servers import us_news_server as mod

    reset_tool_cache_for_tests()
    mod.reset_ticker_cache_for_tests()
    yield
    reset_tool_cache_for_tests()
    mod.reset_ticker_cache_for_tests()


def test_normalize_news_item_nested():
    from research_agent.mcp_servers.us_news_server import _normalize_news_item

    raw = {
        "content": {
            "title": "Apple beats estimates",
            "summary": "Strong iPhone sales.",
            "pubDate": "2024-01-01T00:00:00Z",
            "provider": {"displayName": "Yahoo Finance"},
            "clickThroughUrl": {"url": "https://example.com/a"},
        }
    }
    item = _normalize_news_item(raw)
    assert item is not None
    assert item["title"] == "Apple beats estimates"
    assert item["url"] == "https://example.com/a"


@pytest.mark.asyncio
async def test_get_ticker_news_mocked():
    from research_agent.mcp_servers import us_news_server as mod

    fake_news = [
        {
            "content": {
                "title": "TSLA rally",
                "summary": "Shares climb",
                "pubDate": "2024-06-01T12:00:00Z",
                "provider": {"displayName": "Reuters"},
                "clickThroughUrl": {"url": "https://example.com/t"},
            }
        }
    ]
    mock_ticker = MagicMock()
    mock_ticker.news = fake_news

    # 强制走 yfinance，避免本机可达时 Search HTTP 返回真实新闻
    with (
        patch.object(mod, "_fetch_news_via_yahoo_search", return_value=[]),
        patch("yfinance.Ticker", return_value=mock_ticker),
    ):
        result = await mod.get_ticker_news("tsla", limit=5)

    assert "error" not in result
    assert result["symbol"] == "TSLA"
    assert result["count"] == 1
    assert result["news"][0]["title"] == "TSLA rally"
    assert result["source"] == "yfinance"


@pytest.mark.asyncio
async def test_get_ticker_news_merges_yfinance_summary():
    """Search 只有标题时，合并 yfinance 同题概要。"""
    from research_agent.mcp_servers import us_news_server as mod

    search_hit = [
        {
            "title": "TSLA rally",
            "summary": "",
            "publisher": "Yahoo",
            "published_at": "2024-06-01T12:00:00Z",
            "url": "https://example.com/t",
            "source": "yahoo_search",
            "provider": "yahoo_search",
        }
    ]
    fake_news = [
        {
            "content": {
                "title": "TSLA rally",
                "summary": "Shares climb on delivery beat.",
                "pubDate": "2024-06-01T12:00:00Z",
                "provider": {"displayName": "Reuters"},
                "clickThroughUrl": {"url": "https://example.com/t"},
            }
        }
    ]
    mock_ticker = MagicMock()
    mock_ticker.news = fake_news

    with (
        patch.object(mod, "_fetch_news_via_yahoo_search", return_value=search_hit),
        patch("yfinance.Ticker", return_value=mock_ticker),
    ):
        result = await mod.get_ticker_news("tsla", limit=5)

    assert "error" not in result
    assert result["count"] >= 1
    assert "Shares climb" in (result["news"][0].get("summary") or "")


def test_merge_yahoo_news_prefer_summary():
    from research_agent.mcp_servers.us_news_server import _merge_yahoo_news_prefer_summary

    search = [
        {
            "title": "A",
            "summary": "",
            "url": "https://ex.com/a",
            "source": "yahoo_search",
        }
    ]
    yf = [
        {
            "title": "A",
            "summary": "Has body",
            "url": "https://ex.com/a",
            "source": "yfinance",
        }
    ]
    out = _merge_yahoo_news_prefer_summary(search, yf, limit=5)
    assert len(out) == 1
    assert out[0]["summary"] == "Has body"


@pytest.mark.asyncio
async def test_get_recent_8k_headlines_mocked():
    from research_agent.mcp_servers import us_news_server as mod

    submissions = {
        "name": "Apple Inc.",
        "tickers": ["AAPL"],
        "filings": {
            "recent": {
                "accessionNumber": ["0000320193-24-000001"],
                "form": ["8-K"],
                "filingDate": ["2024-01-15"],
                "primaryDocument": ["aapl-8k.htm"],
                "primaryDocDescription": ["Current report"],
            }
        },
    }

    async def fake_json(url: str):
        if "company_tickers" in url:
            return {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
        return submissions

    with patch.object(mod, "_http_get_json", new=AsyncMock(side_effect=fake_json)):
        result = await mod.get_recent_8k_headlines("AAPL", limit=5)

    assert "error" not in result
    assert result["count"] == 1
    assert result["headlines"][0]["form"] == "8-K"
