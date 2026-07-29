"""美股来源/代理诚实性：通用契约，不针对单一 ticker。"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from research_agent.market.us_source_honesty import find_us_quote_misstatements
from research_agent.mcp_servers import us_data_server as mod


def test_all_proxy_maps_are_aligned():
    """代理表三张必须键一致，避免只修 VIXY、漏掉罗素等。"""
    assert set(mod._EM_US_PROXY_LABELS) == set(mod._EM_US_PROXY_INSTRUMENTS)
    assert set(mod._EM_US_PROXY_LABELS).issubset(set(mod._EM_US_FIXED_SECIDS))


@pytest.mark.parametrize("symbol", sorted(mod._EM_US_PROXY_LABELS))
def test_proxy_quote_contract_for_every_mapped_symbol(symbol: str):
    """任意代理标的：必须带 proxy/warning/代理展示名，且不等于指数原名。"""
    inst = mod._EM_US_PROXY_INSTRUMENTS[symbol]
    label = mod._EM_US_PROXY_LABELS[symbol]
    em = {
        "symbol": symbol,
        "price": 21.44 if symbol == "^VIX" else 291.17,
        "previous_close": 22.0,
        "change": -0.5,
        "change_percent": -1.5,
        "source": "eastmoney_us",
    }
    with (
        patch.object(mod, "_quote_via_yahoo_chart", return_value=None),
        patch.object(mod, "_quote_via_finnhub", return_value=None),
        patch.object(mod, "_quote_via_eastmoney_us", return_value=em),
    ):
        q = mod._quote_from_ticker(symbol)

    assert q["proxy"] is True
    assert q["source"] == "eastmoney_us"
    assert q["quoted_instrument"] == inst
    assert q["name"] == label
    assert q.get("warning")
    assert "禁止" in (q.get("as_of_note") or "") or "代理" in (q.get("as_of_note") or "")
    # 不得仍用指数原名当作 name
    assert "VIX恐慌" not in q["name"]
    assert "罗素2000 (Russell" not in q["name"]


def test_direct_eastmoney_index_is_not_proxy():
    """有东财指数码的标的（如标普）是直连，不是代理。"""
    em = {
        "symbol": "^GSPC",
        "price": 7411.98,
        "previous_close": 7408.3,
        "change": 3.68,
        "change_percent": 0.05,
        "source": "eastmoney_us",
    }
    with (
        patch.object(mod, "_quote_via_yahoo_chart", return_value=None),
        patch.object(mod, "_quote_via_finnhub", return_value=None),
        patch.object(mod, "_quote_via_eastmoney_us", return_value=em),
    ):
        q = mod._quote_from_ticker("^GSPC")

    assert q["proxy"] is False
    assert q["source"] == "eastmoney_us"
    assert q["name"] == "标普500 (S&P 500)"


def test_honesty_catches_yahoo_mislabel_and_vix_proxy_lie():
    items = [
        {
            "symbol": "^GSPC",
            "name": "标普500 (S&P 500)",
            "price": 7411.98,
            "source": "eastmoney_us",
            "proxy": False,
        },
        {
            "symbol": "^VIX",
            "name": "VIX短期期货ETF (VIXY)",
            "price": 21.44,
            "source": "eastmoney_us",
            "proxy": True,
            "proxy_of": "^VIX",
            "quoted_instrument": "VIXY",
        },
    ]
    bad = "VIX恐慌指数报 21.44，跌 -+1.56%。\n数据来源：Yahoo Finance 美股行情。"
    issues = find_us_quote_misstatements(bad, items)
    assert "source_mislabel_yahoo" in issues
    assert any(x.startswith("proxy_presented_as_index:^VIX") for x in issues)
    assert "signed_percent_typo" in issues

    good = (
        "VIX短期期货ETF (VIXY) 报 21.44（东财无 VIX 现货，此为代理，"
        "与官方 VIX≈18.x 不可等同）。跌幅 -1.56%。\n"
        "数据来源：eastmoney_us（东财美股）。"
    )
    assert find_us_quote_misstatements(good, items) == []


def test_honesty_catches_russell_proxy_lie():
    items = [
        {
            "symbol": "^RUT",
            "name": "罗素2000ETF (IWM)",
            "price": 291.17,
            "source": "eastmoney_us",
            "proxy": True,
            "proxy_of": "^RUT",
            "quoted_instrument": "IWM",
        }
    ]
    bad = "罗素2000 收于 291.17。"
    issues = find_us_quote_misstatements(bad, items)
    assert any(x.startswith("proxy_presented_as_index:^RUT") for x in issues)
