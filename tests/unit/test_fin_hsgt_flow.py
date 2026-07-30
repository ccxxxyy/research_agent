"""离线：北向资金只拉单页，禁止 akshare 全历史翻页。"""

from __future__ import annotations

import pytest

from research_agent.mcp_servers import fin_data_server as fin


def test_fetch_hsgt_hist_page_single_request(monkeypatch) -> None:
    calls: list[str] = []

    def _fake(url: str, *, timeout: int = 10):
        calls.append(url)
        assert "pageNumber=1" in url
        assert "pageSize=5" in url
        assert 'MUTUAL_TYPE="005"' in url or "MUTUAL_TYPE=%22005%22" in url or "005" in url
        return {
            "result": {
                "data": [
                    {
                        "TRADE_DATE": "2026-07-30",
                        "FUND_INFLOW": 1.2,
                        "NET_DEAL_AMT": 3.4,
                        "BUY_AMT": 5,
                        "SELL_AMT": 1.6,
                        "LEAD_STOCKS_NAME": "测试",
                        "LEAD_STOCKS_CODE": "600000",
                        "LS_CHANGE_RATE": 1.0,
                        "QUOTA_BALANCE": 100,
                    }
                ]
            }
        }

    monkeypatch.setattr(fin, "_curl_get_json", _fake)
    rows = fin._fetch_hsgt_hist_page(mutual_type="5", limit=5)
    assert len(calls) == 1
    assert len(rows) == 1
    assert rows[0]["日期"] == "2026-07-30"
    assert rows[0]["领涨股"] == "测试"


@pytest.mark.asyncio
async def test_get_hsgt_flow_market_scope(monkeypatch) -> None:
    monkeypatch.setattr(
        fin,
        "_fetch_hsgt_hist_page",
        lambda **_kw: [{"日期": "2026-07-30", "当日成交净买额": 1.0}],
    )
    out = await fin.get_hsgt_flow(direction="north", limit=3)
    assert out["scope"] == "market"
    assert out["direction"] == "北向资金"
    assert out["count"] == 1
    assert "个股" in out["note"] or "市场级" in out["note"]


@pytest.mark.asyncio
async def test_get_hsgt_flow_rejects_bad_direction() -> None:
    out = await fin.get_hsgt_flow(direction="west", limit=3)
    assert "error" in out
