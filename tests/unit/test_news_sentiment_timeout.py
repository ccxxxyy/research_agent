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
    monkeypatch.setattr(
        ns,
        "_fetch_fund_flow_signal",
        lambda _sym, **_kw: {"available": True, "latest_main_net_inflow": 1.0, "days": 1},
    )
    monkeypatch.setattr(
        ns,
        "_fetch_analyst_reports",
        lambda _sym, **_kw: {
            "available": True,
            "count": 1,
            "ratings_sample": ["买入"],
            "reports": [],
        },
    )

    out = ns._full_report("300308", 5)
    assert out["aggregate"]["sample_size"] == 1
    assert out["xueqiu_heat"].get("skipped") is True
    assert any("xueqiu" in n for n in out.get("partial_notes") or [])
    assert out["aux_signals"]["social"]["used"] is True  # 热搜词命中
    assert out["aux_signals"]["fund_flow"]["used"] is True
    assert out["aux_signals"]["analyst"]["used"] is True
    assert out["signal_notes"]
    assert any("社交" in n for n in out["signal_notes"])
    assert any("资金" in n for n in out["signal_notes"])
    assert any("分析师" in n or "研报" in n for n in out["signal_notes"])


def test_build_aux_signals_notes_only_when_used() -> None:
    aux, notes = ns._build_aux_signals(
        xueqiu_heat={"on_list": False},
        eastmoney_keywords=[],
        fund_flow={"available": False},
        analyst={"available": False},
    )
    assert notes == []
    assert aux["social"]["used"] is False
    assert aux["fund_flow"]["used"] is False
    assert aux["analyst"]["used"] is False
    assert "新闻" in aux["news_what"] or "SnowNLP" in aux["news_what"]

    aux2, notes2 = ns._build_aux_signals(
        xueqiu_heat={"on_list": True},
        eastmoney_keywords=[{"keyword": "AI"}],
        fund_flow={"available": True, "latest_main_net_inflow": -1},
        analyst={"available": True, "ratings_sample": ["增持"]},
    )
    assert len(notes2) == 3
    assert aux2["social"]["used"] is True
    assert "雪球" in aux2["social"]["what"]
    assert "资金" in aux2["fund_flow"]["what"]
    assert "研报" in aux2["analyst"]["what"]
