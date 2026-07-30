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


def test_full_report_parallel_aux_keeps_analyst_when_xueqiu_slow(monkeypatch) -> None:
    """雪球慢不应拖死研报旁路（并行后研报仍应 used）。"""

    class _FakeDF:
        empty = False

        def head(self, _n: int):
            return self

        def iterrows(self):
            yield (
                0,
                {
                    "新闻标题": "宏发股份获订单",
                    "新闻内容": "业绩预增",
                    "发布时间": "2026-07-30",
                    "文章来源": "测试",
                    "新闻链接": "https://example.com/n1",
                },
            )

    monkeypatch.setattr(ns, "_fetch_eastmoney_news_df", lambda _sym: _FakeDF())
    monkeypatch.setattr(ns, "_fetch_hot_keywords", lambda _sym: [])

    def _hang(_sym: str):
        time.sleep(5.0)
        return {"on_list": True}

    monkeypatch.setattr(ns, "_fetch_xueqiu_heat", _hang)
    monkeypatch.setattr(ns, "_XUEQIU_HEAT_TIMEOUT_S", 0.3)
    monkeypatch.setattr(
        ns,
        "_fetch_fund_flow_signal",
        lambda _sym, **_kw: {"available": False, "reason": "empty"},
    )
    monkeypatch.setattr(
        ns,
        "_fetch_analyst_reports",
        lambda _sym, **_kw: {
            "available": True,
            "count": 1,
            "ratings_sample": ["买入"],
            "reports": [{"title": "维持买入", "rating": "买入"}],
        },
    )

    out = ns._full_report("600885", 5)
    assert out["aggregate"]["sample_size"] == 1
    assert out["aux_signals"]["analyst"]["used"] is True
    assert out["xueqiu_heat"].get("skipped") is True


def test_full_report_does_not_block_on_hung_aux_after_timeout(monkeypatch) -> None:
    """超时旁路线程仍在跑时，报告必须马上返回（禁止 ThreadPool wait=True 堵死）。"""

    class _FakeDF:
        empty = False

        def head(self, _n: int):
            return self

        def iterrows(self):
            yield (
                0,
                {
                    "新闻标题": "测试标题",
                    "新闻内容": "利好",
                    "发布时间": "2026-07-30",
                    "文章来源": "测试",
                    "新闻链接": "https://example.com/n1",
                },
            )

    monkeypatch.setattr(ns, "_fetch_eastmoney_news_df", lambda _sym: _FakeDF())
    monkeypatch.setattr(ns, "_fetch_hot_keywords", lambda _sym: [])
    monkeypatch.setattr(ns, "_XUEQIU_HEAT_TIMEOUT_S", 0.2)
    monkeypatch.setattr(ns, "_HOT_KEYWORDS_TIMEOUT_S", 0.2)
    monkeypatch.setattr(ns, "_AUX_SIGNAL_TIMEOUT_S", 0.2)

    def _hang(_sym: str, **_kw):
        # 略长于旁路超时即可；过长会拖住 pytest 进程退出（wait=False 线程仍存活）
        time.sleep(2.0)
        return {"on_list": True}

    monkeypatch.setattr(ns, "_fetch_xueqiu_heat", _hang)
    monkeypatch.setattr(ns, "_fetch_fund_flow_signal", _hang)
    monkeypatch.setattr(
        ns,
        "_fetch_analyst_reports",
        lambda _sym, **_kw: {"available": True, "ratings_sample": ["增持"], "reports": []},
    )

    # 预热打分，避免把 SnowNLP 冷启动算进阻塞时间
    ns._score_single("利好业绩预增")

    t0 = time.perf_counter()
    out = ns._full_report("600885", 3)
    elapsed = time.perf_counter() - t0
    # 若误用 shutdown(wait=True)，会等到 ~2s 旁路线程结束
    assert elapsed < 1.2, f"report blocked too long: {elapsed:.1f}s"
    assert out["aggregate"]["sample_size"] == 1
    assert out["aux_signals"]["analyst"]["used"] is True


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


def test_fetch_analyst_reports_uses_single_page(monkeypatch) -> None:
    """必须只打一页；禁止依赖 akshare 全量翻页。"""
    calls: list[dict[str, str]] = []

    def _fake_http(url: str, *, params: dict[str, str], timeout: float = 10.0):
        calls.append(dict(params))
        assert "reportapi.eastmoney.com" in url
        assert params["pageNo"] == "1"
        assert params["pageSize"] == "8"
        return {
            "data": [
                {
                    "title": "维持买入",
                    "emRatingName": "买入",
                    "orgSName": "测试证券",
                    "publishDate": "2026-07-01",
                    "indvInduName": "电器",
                    "infoCode": "ABC123",
                }
            ]
        }

    monkeypatch.setattr(ns, "_http_get_json", _fake_http)
    out = ns._fetch_analyst_reports("600885", limit=8)
    assert len(calls) == 1
    assert out["available"] is True
    assert out["ratings_sample"] == ["买入"]
    assert out["reports"][0]["institution"] == "测试证券"
    assert "ABC123" in out["reports"][0]["pdf_url"]


def test_fetch_fund_flow_signal_swallows_connection_error(monkeypatch) -> None:
    import types

    fake_ak = types.SimpleNamespace(
        stock_individual_fund_flow=lambda **_kw: (_ for _ in ()).throw(
            ConnectionError("RemoteDisconnected")
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "akshare", fake_ak)
    out = ns._fetch_fund_flow_signal("600885")
    assert out["available"] is False
    assert "ConnectionError" in out["reason"]
