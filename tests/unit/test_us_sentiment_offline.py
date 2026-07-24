"""``us_sentiment_server`` 离线单元测试 — 无网络。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from research_agent.cache import reset_tool_cache_for_tests


@pytest.fixture(autouse=True)
def _reset_caches():
    reset_tool_cache_for_tests()
    yield
    reset_tool_cache_for_tests()


def test_score_positive_keywords():
    from research_agent.mcp_servers.us_sentiment_server import _score_single

    r = _score_single("Apple beats estimates as iPhone sales surge")
    assert r["sentiment_score"] > 0
    assert r["sentiment_label"] in {"positive", "strong_positive"}
    assert "beat" in r["keywords_matched"] or "surge" in r["keywords_matched"]


def test_score_negative_keywords():
    from research_agent.mcp_servers.us_sentiment_server import _score_single

    r = _score_single("Shares plunge after guidance cut and lawsuit filing")
    assert r["sentiment_score"] < 0
    assert r["sentiment_label"] in {"negative", "strong_negative"}


def test_score_neutral_without_keywords():
    from research_agent.mcp_servers.us_sentiment_server import _score_single

    r = _score_single("The company reported quarterly results today")
    assert r["sentiment_score"] == 0.0
    assert r["sentiment_label"] == "neutral"


@pytest.mark.asyncio
async def test_analyze_text_sentiment():
    from research_agent.mcp_servers import us_sentiment_server as mod

    result = await mod.analyze_text_sentiment(
        ["Apple beats estimates", "Tesla shares plunge on demand warning"]
    )
    assert result["model_version"] == "en_fin_keywords_v1"
    assert len(result["items"]) == 2
    assert result["aggregate"]["sample_size"] == 2


@pytest.mark.asyncio
async def test_get_ticker_sentiment_report_mocked():
    from research_agent.mcp_servers import us_sentiment_server as mod

    fake_news = [
        {
            "content": {
                "title": "NVDA upgraded on AI demand",
                "summary": "Analyst raises guidance outlook",
                "pubDate": "2024-06-01T12:00:00Z",
                "provider": {"displayName": "Bloomberg"},
                "clickThroughUrl": {"url": "https://example.com/n"},
            }
        }
    ]
    mock_ticker = MagicMock()
    mock_ticker.news = fake_news

    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = await mod.get_ticker_sentiment_report("NVDA", limit=5)

    assert "error" not in result
    assert result["symbol"] == "NVDA"
    assert result["aggregate"]["sample_size"] == 1
    assert result["items"][0]["sentiment_score"] > 0
