"""derivatives_server 离线单测 — mock akshare，无网络。"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from research_agent.cache import reset_tool_cache_for_tests


@pytest.fixture(autouse=True)
def _reset_caches():
    reset_tool_cache_for_tests()
    yield
    reset_tool_cache_for_tests()


@pytest.mark.asyncio
async def test_search_futures():
    from research_agent.mcp_servers import derivatives_server as mod

    result = await mod.search_futures("螺纹", limit=10)
    assert result["count"] >= 1
    assert any(r["code"] == "RB" for r in result["results"])


@pytest.mark.asyncio
async def test_get_main_futures():
    from research_agent.mcp_servers import derivatives_server as mod

    result = await mod.get_main_futures(limit=5)
    assert result["count"] == 5
    assert result["futures"][0]["sina_main"].endswith("0")


@pytest.mark.asyncio
async def test_get_futures_spot_mocked():
    from research_agent.mcp_servers import derivatives_server as mod

    mock_df = pd.DataFrame({"名称": ["螺纹钢2505"], "最新价": [3500.0], "涨跌幅": [1.2]})
    with patch("akshare.futures_zh_realtime", return_value=mock_df):
        result = await mod.get_futures_spot("RB")
    assert result["count"] == 1
    assert result["source"] == "sina"


@pytest.mark.asyncio
async def test_get_futures_daily_mocked():
    from research_agent.mcp_servers import derivatives_server as mod

    mock_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "open": [1.0, 1.1],
            "close": [1.05, 1.15],
        }
    )
    with patch("akshare.futures_zh_daily_sina", return_value=mock_df):
        result = await mod.get_futures_daily("RB", limit=10)
    assert result["symbol"] == "RB0"
    assert result["count"] == 2


@pytest.mark.asyncio
async def test_get_etf_option_list_mocked():
    from research_agent.mcp_servers import derivatives_server as mod

    with patch("akshare.option_sse_list_sina", return_value=["202502", "202503"]):
        result = await mod.get_etf_option_list("50ETF")
    assert result["count"] == 2
    assert result["expirations"][0] == "202502"


@pytest.mark.asyncio
async def test_get_etf_option_spot_mocked():
    from research_agent.mcp_servers import derivatives_server as mod

    mock_df = pd.DataFrame({"字段": ["最新价"], "值": [0.12]})
    with patch("akshare.option_sse_spot_price_sina", return_value=mock_df):
        result = await mod.get_etf_option_spot("10003720")
    assert result["contract"] == "10003720"
    assert result["count"] == 1


@pytest.mark.asyncio
async def test_get_index_option_spot_mocked():
    from research_agent.mcp_servers import derivatives_server as mod

    spot_df = pd.DataFrame({"看涨合约": ["io2504-C-4000"], "最新价": [50.0]})
    with (
        patch("akshare.option_cffex_hs300_list_sina", return_value=["io2504"]),
        patch("akshare.option_cffex_hs300_spot_sina", return_value=spot_df),
    ):
        result = await mod.get_index_option_spot("沪深300", contract="io2504")
    assert result["family"] == "hs300"
    assert result["count"] == 1
