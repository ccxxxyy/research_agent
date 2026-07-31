"""个股资金流向：push2his curl 解析与工具契约。"""

from __future__ import annotations

import research_agent.mcp_servers.fin_data_server as fin


def test_parse_fflow_daykline_maps_main_net() -> None:
    klines = [
        "2026-07-29,-22328102.0,-4157984.0,26486085.0,-20555909.0,-1772193.0,-3.60,-0.67,4.27,-3.31,-0.29,33.98,2.10,0.00,0.00",
        "2026-07-30,-122665305.0,29039280.0,93626037.0,-49307871.0,-73357434.0,-13.48,3.19,10.29,-5.42,-8.06,34.49,1.50,0.00,0.00",
    ]
    rows = fin._parse_fflow_daykline(klines, limit=5)
    assert len(rows) == 2
    assert rows[-1]["日期"] == "2026-07-30"
    assert rows[-1]["主力净流入-净额"] == -122665305.0
    assert rows[-1]["超大单净流入-净额"] == -73357434.0


def test_fetch_individual_fund_flow_via_curl(monkeypatch) -> None:
    sample = {
        "data": {
            "klines": [
                "2026-07-30,-1.0,2.0,3.0,4.0,5.0,-1.1,0,0,0,0,10.0,1.0,0,0",
            ]
        }
    }
    monkeypatch.setattr(fin, "_curl_get_json", lambda *_a, **_k: sample)
    rows = fin._fetch_individual_fund_flow_via_curl("600885", limit=5)
    assert len(rows) == 1
    assert rows[0]["主力净流入-净额"] == -1.0
