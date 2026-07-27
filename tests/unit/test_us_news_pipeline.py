"""美股新闻管道：过滤 / 聚类 / 标签 / Finnhub 合并（离线）。"""

from __future__ import annotations

from unittest.mock import patch

from research_agent.mcp_servers import us_news_pipeline as mod


def test_junk_filter_penny_and_exclaims():
    assert mod.is_junk_item({"title": "HOT PENNY STOCK!!!", "publisher": "X", "url": ""})
    assert mod.is_junk_item(
        {"title": "Buy now", "publisher": "PennyStock Tips", "url": "https://pennystock.example/a"}
    )
    assert not mod.is_junk_item(
        {
            "title": "Apple reports quarterly earnings",
            "publisher": "Reuters",
            "url": "https://reuters.com/a",
        }
    )


def test_event_tag_earnings_and_mna():
    et, zh = mod.tag_event({"title": "NVDA beats earnings, raises guidance", "summary": ""})
    assert et == "earnings"
    assert zh == "财报"
    et2, zh2 = mod.tag_event({"title": "Firm to acquire rival in $2B deal", "summary": ""})
    assert et2 == "m_and_a"
    assert zh2 == "并购"


def test_cluster_similar_titles():
    items = [
        {
            "title": "Nvidia earnings beat expectations on AI demand",
            "publisher": "Yahoo Finance",
            "url": "https://yahoo.example/1",
            "provider": "yahoo_search",
            "published_at": "1",
        },
        {
            "title": "Nvidia earnings beat expectations amid AI demand",
            "publisher": "Reuters",
            "url": "https://reuters.example/2",
            "provider": "finnhub",
            "published_at": "2",
        },
        {
            "title": "Totally unrelated chip shortage in Europe",
            "publisher": "CNBC",
            "url": "https://cnbc.example/3",
            "provider": "finnhub",
            "published_at": "3",
        },
    ]
    out = mod.filter_and_cluster(items, limit=10)
    assert len(out) == 2
    cluster = next(x for x in out if x.get("cluster_size", 1) >= 2)
    assert cluster["cluster_size"] == 2
    assert "reuters" in cluster["publisher"].lower()
    assert cluster["event_type"] == "earnings"


def test_collect_without_finnhub_key():
    yahoo = [
        {
            "title": "Apple launches new product",
            "publisher": "CNBC",
            "url": "https://cnbc.example/a",
            "source": "yahoo_search",
        }
    ]
    with patch.object(mod, "finnhub_api_key", return_value=""):
        bundle = mod.collect_us_news("AAPL", yahoo_items=yahoo, limit=5, finnhub_key="")
    assert bundle["count"] == 1
    assert "yahoo" in bundle["providers_used"] or "yahoo_search" in bundle["providers_used"]
    assert bundle.get("note")


def test_collect_with_finnhub_when_yahoo_empty():
    fh = [
        {
            "title": "MSFT cloud growth accelerates",
            "summary": "",
            "publisher": "Bloomberg",
            "url": "https://bloomberg.example/m",
            "provider": "finnhub",
            "source": "finnhub",
            "published_at": "",
        }
    ]
    with patch.object(mod, "fetch_finnhub_company_news", return_value=fh):
        bundle = mod.collect_us_news("MSFT", yahoo_items=[], limit=5, finnhub_key="test-key")
    assert bundle["count"] == 1
    assert "finnhub" in bundle["providers_used"]
    assert bundle["news"][0]["title"].startswith("MSFT")
