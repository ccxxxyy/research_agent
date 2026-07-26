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


def test_build_score_text_basis():
    from research_agent.mcp_servers.us_sentiment_server import _build_score_text

    t, basis = _build_score_text("Title only", "", "")
    assert basis == "title"
    assert "Title only" in t

    t2, basis2 = _build_score_text("Great news", "But lawsuit risk rises sharply", "")
    assert basis2 == "title+summary"
    assert "lawsuit" in t2.lower()

    t3, basis3 = _build_score_text("Surge!", "", "Company plunges after fraud probe.")
    assert basis3 == "title+body"
    assert "fraud" in t3.lower()


def test_html_meta_and_body_snippet():
    from research_agent.mcp_servers.us_sentiment_server import (
        _extract_meta_description,
        _html_to_text_snippet,
    )

    html = """
    <html><head>
      <meta name="description" content="Shares plunge after guidance cut." />
    </head><body>
      <script>ignore()</script>
      <p>Shares plunge after guidance cut and lawsuit filing continues.</p>
    </body></html>
    """
    assert "plunge" in _extract_meta_description(html).lower()
    body = _html_to_text_snippet(html)
    assert "lawsuit" in body.lower()
    assert "ignore()" not in body


def test_enrich_thin_summaries_fetches_body():
    from research_agent.mcp_servers import us_sentiment_server as mod

    news_list = [
        {"title": "Amazing surge!", "summary": "", "url": "https://example.com/a"},
        {"title": "Has summary", "summary": "x" * 100, "url": "https://example.com/b"},
    ]
    with patch.object(
        mod, "_fetch_article_snippet", return_value="Company faces bankruptcy risk."
    ) as fetch:
        mod._enrich_thin_summaries(news_list)
    fetch.assert_called_once_with("https://example.com/a")
    assert "bankruptcy" in news_list[0]["body_snippet"]
    assert "body_snippet" not in news_list[1]


def test_clickbait_title_softened_by_body():
    """标题夸大、正文偏负面时，合成文本应拉低分数。"""
    from research_agent.mcp_servers.us_sentiment_server import _build_score_text, _score_single

    title_only = _score_single("Investors are thrilled with amazing record gains!!!")
    combined, basis = _build_score_text(
        "Investors are thrilled with amazing record gains!!!",
        "",
        "The company plunges into bankruptcy after fraud investigation and massive layoffs.",
    )
    with_body = _score_single(combined)
    assert "body" in basis
    assert with_body["sentiment_score"] < title_only["sentiment_score"]


@pytest.mark.asyncio
async def test_analyze_text_sentiment():
    from research_agent.mcp_servers import us_sentiment_server as mod

    result = await mod.analyze_text_sentiment(
        ["Apple beats estimates", "Tesla shares plunge on demand warning"]
    )
    assert result["model_version"] == "en_vader_finlex_v2"
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
                "summary": (
                    "Wall Street analysts raise guidance outlook on sustained AI demand "
                    "and data center growth across enterprise customers this quarter."
                ),
                "pubDate": "2024-06-01T12:00:00Z",
                "provider": {"displayName": "Bloomberg"},
                "clickThroughUrl": {"url": "https://example.com/n"},
            }
        }
    ]
    mock_ticker = MagicMock()
    mock_ticker.news = fake_news

    # 强制走 yfinance 回退；摘要足够长则不应再抓正文
    with (
        patch.object(mod, "_fetch_news_via_yahoo_search", return_value=[]),
        patch.object(mod, "_fetch_article_snippet", return_value="") as body_fetch,
        patch("yfinance.Ticker", return_value=mock_ticker),
    ):
        result = await mod.get_ticker_sentiment_report("NVDA", limit=5)

    assert "error" not in result
    assert result["symbol"] == "NVDA"
    assert result["model_version"] == "en_vader_finlex_v2"
    assert result["aggregate"]["sample_size"] == 1
    assert result["items"][0]["sentiment_score"] > 0
    assert result["items"][0]["fetch_source"] == "yfinance"
    assert result["items"][0]["score_text_basis"] == "title+summary"
    body_fetch.assert_not_called()
