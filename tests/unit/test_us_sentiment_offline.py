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
    assert "vader_compound" in r
    assert r["vader_compound"] >= -1.0


def test_score_negative_keywords():
    from research_agent.mcp_servers.us_sentiment_server import _score_single

    r = _score_single("Shares plunge after guidance cut and lawsuit filing")
    assert r["sentiment_score"] < 0
    assert r["sentiment_label"] in {"negative", "strong_negative"}
    assert r["vader_compound"] <= 0.0


def test_score_neutral_factual():
    from research_agent.mcp_servers.us_sentiment_server import (
        _NEGATIVE_THRESHOLD,
        _POSITIVE_THRESHOLD,
        _score_single,
    )

    r = _score_single("The company reported quarterly results today")
    # VADER 可能有微弱偏置，不再要求严格 == 0
    assert abs(r["sentiment_score"]) < _POSITIVE_THRESHOLD
    assert r["sentiment_label"] == "neutral"
    assert r["sentiment_score"] > _NEGATIVE_THRESHOLD


def test_score_vader_without_finlex_keywords():
    """词典时代难识别、VADER 能识别的极性句。"""
    from research_agent.mcp_servers.us_sentiment_server import _score_single

    r = _score_single("Investors are absolutely thrilled with this amazing outcome!!!")
    assert r["sentiment_score"] > 0.15
    assert r["sentiment_label"] in {"positive", "strong_positive"}
    assert r["vader_compound"] > 0.3
    # 未必命中金融词表
    assert isinstance(r["keywords_matched"], list)


@pytest.mark.asyncio
async def test_analyze_text_sentiment():
    from research_agent.mcp_servers import us_sentiment_server as mod

    result = await mod.analyze_text_sentiment(
        ["Apple beats estimates", "Tesla shares plunge on demand warning"]
    )
    assert result["model_version"] == "en_vader_finlex_v1"
    assert len(result["items"]) == 2
    assert result["aggregate"]["sample_size"] == 2
    assert result["items"][0]["sentiment_score"] > 0
    assert result["items"][1]["sentiment_score"] < 0


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

    # 强制走 yfinance 回退，避免本机可达时 Search HTTP 返回真实新闻
    with (
        patch.object(mod, "_fetch_news_via_yahoo_search", return_value=[]),
        patch("yfinance.Ticker", return_value=mock_ticker),
    ):
        result = await mod.get_ticker_sentiment_report("NVDA", limit=5)

    assert "error" not in result
    assert result["symbol"] == "NVDA"
    assert result["model_version"] == "en_vader_finlex_v1"
    assert result["aggregate"]["sample_size"] == 1
    assert result["items"][0]["sentiment_score"] > 0
    assert result["items"][0]["fetch_source"] == "yfinance"
