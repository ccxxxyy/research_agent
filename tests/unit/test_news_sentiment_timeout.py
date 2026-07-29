"""离线：A 股舆情报告旁路超时不应拖垮所有工具。"""

from __future__ import annotations

import time

from research_agent.mcp_servers import news_sentiment_server as ns


def test_call_with_timeout_returns_default() -> None:
    def _slow() -> str:
        time.sleep(2.0)
        return "late"

    assert ns._call_with_timeout(_slow, timeout=0.2, default="fallback") == "fallback"


def test_full_report_skips_slow_xueqiu(monkeypatch) -> None:
    class _FakeDF:
        empty = False

        def head(self, _n: int):
            return self

        def iterrows(self):
            yield (
                0,
                {
                    "新闻标题": "中际旭创获订单",
                    "新闻内容": "业绩预增",
                    "发布时间": "2026-07-28",
                    "文章来源": "测试",
                    "新闻链接": "https://example.com/n1",
                },
            )

    monkeypatch.setattr(ns, "_fetch_eastmoney_news_df", lambda _sym: _FakeDF())
    monkeypatch.setattr(
        ns,
        "_fetch_hot_keywords",
        lambda _sym: [{"keyword": "光模块", "hot_value": "1", "time": ""}],
    )

    def _hang(_sym: str):
        time.sleep(5.0)
        return {"on_list": True}

    monkeypatch.setattr(ns, "_fetch_xueqiu_heat", _hang)
    monkeypatch.setattr(ns, "_XUEQIU_HEAT_TIMEOUT_S", 0.3)

    out = ns._full_report("300308", 5)
    assert out["aggregate"]["sample_size"] == 1
    assert out["xueqiu_heat"].get("skipped") is True
    assert any("xueqiu" in n for n in out.get("partial_notes") or [])
