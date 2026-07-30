"""离线：市场快讯财联社失败时应回退东财，且不得长阻塞。"""

from __future__ import annotations

import time

import pytest

from research_agent.mcp_servers import news_server as news


def test_telegraph_falls_back_to_eastmoney(monkeypatch) -> None:
    monkeypatch.setattr(news, "_telegraph_from_cls", lambda *_a, **_k: None)

    def _em(symbol: str, limit: int) -> dict:
        return {
            "category": symbol,
            "count": 1,
            "telegraph": [{"标题": "测试快讯", "内容": "内容", "发布时间": "12:00:00"}],
            "source": "eastmoney_flash_fallback",
            "source_url": "https://kuaixun.eastmoney.com/7_24.html",
            "note": "fallback",
        }

    monkeypatch.setattr(news, "_telegraph_from_eastmoney", _em)
    out = news._telegraph_cls("全部", 10)
    assert out["source"] == "eastmoney_flash_fallback"
    assert out["count"] == 1
    assert out["telegraph"][0]["标题"] == "测试快讯"


def test_telegraph_cls_primary_preferred(monkeypatch) -> None:
    monkeypatch.setattr(
        news,
        "_telegraph_from_cls",
        lambda symbol, limit: {
            "category": symbol,
            "count": 1,
            "telegraph": [{"标题": "CLS"}],
            "source": "cls",
        },
    )
    monkeypatch.setattr(
        news,
        "_telegraph_from_eastmoney",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not fallback")),
    )
    out = news._telegraph_cls("重点", 5)
    assert out["source"] == "cls"


@pytest.mark.asyncio
async def test_get_market_telegraph_outer_timeout(monkeypatch) -> None:
    def _hang(_category: str, _limit: int):
        time.sleep(5.0)
        return {"telegraph": []}

    monkeypatch.setattr(news, "_telegraph_cls", _hang)
    monkeypatch.setattr(news, "_TELEGRAPH_TIMEOUT_S", 0.2)
    out = await news.get_market_telegraph(category="全部", limit=5)
    assert "error" in out
    assert "TimeoutError" in out["error"]
