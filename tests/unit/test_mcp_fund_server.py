"""fund_server 单元测试 — 纯函数 + mock 覆盖（不依赖网络）。

测试纯工具函数和降级逻辑，无需实际网络请求。
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _reset_push2_cache():
    """每个测试前重置 push2 探测缓存。"""
    import research_agent.mcp_servers.fund_server as mod

    orig_push2 = mod._PUSH2_AVAILABLE
    orig_push2his = mod._PUSH2HIS_AVAILABLE
    mod._PUSH2_AVAILABLE = None
    mod._PUSH2HIS_AVAILABLE = None
    yield
    mod._PUSH2_AVAILABLE = orig_push2
    mod._PUSH2HIS_AVAILABLE = orig_push2his


def test_df_to_records_basic():
    """_df_to_records 正确转换 DataFrame 为字典列表。"""
    from research_agent.mcp_servers.fund_server import _df_to_records

    df = pd.DataFrame(
        {"代码": ["510300", "159915"], "名称": ["沪深300ETF", "创业板ETF"], "涨跌幅": [1.5, -0.8]}
    )
    records = _df_to_records(df)
    assert len(records) == 2
    assert records[0]["代码"] == "510300"
    assert records[0]["涨跌幅"] == 1.5
    assert records[1]["名称"] == "创业板ETF"


def test_df_to_records_with_limit():
    """_df_to_records 的 limit 参数正确截断。"""
    from research_agent.mcp_servers.fund_server import _df_to_records

    df = pd.DataFrame({"x": range(10)})
    records = _df_to_records(df, limit=3)
    assert len(records) == 3


def test_df_to_records_handles_nan():
    """_df_to_records 将 NaN 转为 None。"""
    from research_agent.mcp_servers.fund_server import _df_to_records

    df = pd.DataFrame({"a": [1.0, float("nan")], "b": ["x", None]})
    records = _df_to_records(df)
    assert records[1]["a"] is None
    assert records[1]["b"] is None


def test_df_to_records_handles_timestamps():
    """_df_to_records 将 Timestamp 转为 YYYY-MM-DD 字符串。"""
    from research_agent.mcp_servers.fund_server import _df_to_records

    df = pd.DataFrame({"日期": pd.to_datetime(["2024-01-15", "2024-06-30"]), "值": [1.0, 2.0]})
    records = _df_to_records(df)
    assert records[0]["日期"] == "2024-01-15"
    assert records[1]["日期"] == "2024-06-30"


def test_fmt_error():
    """_fmt_error 返回结构化错误字典。"""
    from research_agent.mcp_servers.fund_server import _fmt_error

    result = _fmt_error(ValueError("test error"), context="test_ctx")
    assert result["error"] == "ValueError: test error"
    assert result["context"] == "test_ctx"


def test_push2_probe_unavailable():
    """push2 不可达时 _is_push2_available 返回 False。"""
    from research_agent.mcp_servers.fund_server import _is_push2_available

    with patch(
        "research_agent.mcp_servers.fund_server._probe_push2_connectivity", return_value=False
    ):
        assert _is_push2_available() is False


def test_push2_probe_available():
    """push2 可达时 _is_push2_available 返回 True。"""
    from research_agent.mcp_servers.fund_server import _is_push2_available

    with patch(
        "research_agent.mcp_servers.fund_server._probe_push2_connectivity", return_value=True
    ):
        assert _is_push2_available() is True


@pytest.mark.asyncio
async def test_etf_spot_realtime_path():
    """新浪行情可用时 get_fund_etf_spot 走实时路径。"""
    import research_agent.mcp_servers.fund_server as mod

    mock_df = pd.DataFrame(
        {
            "代码": ["sh510300", "sz159915"],
            "名称": ["沪深300ETF", "创业板ETF"],
            "最新价": [4.0, 2.5],
            "涨跌幅": [1.2, -0.5],
            "成交额": [5e8, 3e8],
            "成交量": [1e7, 8e6],
        }
    )

    with patch("akshare.fund_etf_category_sina", return_value=mock_df):
        result = await mod.get_fund_etf_spot(sort_by="成交额", limit=10)
    assert result["realtime"] is True
    assert result["source"] == "sina_realtime"
    assert "source_url" in result
    assert len(result["etfs"]) == 2


@pytest.mark.asyncio
async def test_etf_spot_fallback_path():
    """新浪不可用时 get_fund_etf_spot 走降级路径。"""
    import research_agent.mcp_servers.fund_server as mod

    fallback_df = pd.DataFrame(
        {
            "基金代码": ["510300"],
            "基金简称": ["沪深300ETF"],
            "单位净值": [4.0],
            "今年来": [5.2],
            "近1周": [0.3],
            "近1月": [1.1],
            "近1年": [8.0],
        }
    )

    with patch("akshare.fund_etf_category_sina", side_effect=ConnectionError("down")):
        with patch("akshare.fund_open_fund_rank_em", return_value=fallback_df):
            result = await mod.get_fund_etf_spot(sort_by="今年来", limit=10)
    assert result["realtime"] is False
    assert result["source"] == "eastmoney_rank"
    assert "note" in result
    assert "source_url" in result


@pytest.mark.asyncio
async def test_lof_spot_realtime_path():
    """新浪行情可用时 get_fund_lof_spot 走实时路径。"""
    import research_agent.mcp_servers.fund_server as mod

    mock_df = pd.DataFrame(
        {
            "代码": ["sz160119"],
            "名称": ["南方中证500ETF联接LOF"],
            "最新价": [1.8],
            "涨跌幅": [0.9],
            "成交额": [1e8],
            "成交量": [5e6],
        }
    )

    with patch("akshare.fund_etf_category_sina", return_value=mock_df):
        result = await mod.get_fund_lof_spot(sort_by="成交额", limit=10)
    assert result["realtime"] is True
    assert result["source"] == "sina_realtime"
    assert "source_url" in result


@pytest.mark.asyncio
async def test_lof_spot_fallback_path():
    """新浪不可用时 get_fund_lof_spot 走降级路径。"""
    import research_agent.mcp_servers.fund_server as mod

    fallback_df = pd.DataFrame(
        {
            "基金代码": ["160119"],
            "基金简称": ["南方中证500LOF"],
            "单位净值": [1.8],
            "今年来": [3.5],
            "近1周": [0.2],
            "近1月": [0.8],
            "近1年": [6.0],
        }
    )

    with patch("akshare.fund_etf_category_sina", side_effect=ConnectionError("down")):
        with patch("akshare.fund_open_fund_rank_em", return_value=fallback_df):
            result = await mod.get_fund_lof_spot(sort_by="今年来", limit=10)
    assert result["realtime"] is False
    assert result["source"] == "eastmoney_rank"
    assert "source_url" in result


@pytest.mark.asyncio
async def test_etf_hist_realtime_path():
    """curl_cffi 可用时 get_fund_etf_hist 走 push2his 直连路径。"""
    import research_agent.mcp_servers.fund_server as mod

    mock_df = pd.DataFrame(
        {
            "日期": ["2024-06-01", "2024-06-02"],
            "开盘": [4.0, 4.1],
            "收盘": [4.1, 4.05],
            "最高": [4.15, 4.12],
            "最低": [3.98, 4.0],
            "成交量": [1e6, 8e5],
            "涨跌幅": [1.0, -0.5],
        }
    )

    with patch.object(mod, "_fetch_etf_kline_via_curl", return_value=mock_df):
        result = await mod.get_fund_etf_hist(symbol="510300", period="daily", limit=60)
    assert result["realtime"] is True
    assert result["source"] == "eastmoney_push2his_curl"
    assert result["period"] == "daily"


@pytest.mark.asyncio
async def test_etf_hist_fallback_path():
    """curl_cffi 和 push2his 均不可达时走净值降级路径。"""
    import research_agent.mcp_servers.fund_server as mod

    mock_df = pd.DataFrame(
        {
            "净值日期": ["2024-06-01", "2024-06-02"],
            "单位净值": [4.1, 4.05],
            "日增长率": [0.5, -0.3],
        }
    )

    mod._PUSH2HIS_AVAILABLE = False
    with (
        patch.object(mod, "_fetch_etf_kline_via_curl", return_value=None),
        patch("akshare.fund_open_fund_info_em", return_value=mock_df),
    ):
        result = await mod.get_fund_etf_hist(symbol="510300", limit=60)
    assert result["realtime"] is False
    assert result["source"] == "eastmoney_nav"
    assert "note" in result


@pytest.mark.asyncio
async def test_search_fund():
    """search_fund 同时匹配代码和名称。"""
    import research_agent.mcp_servers.fund_server as mod

    mock_df = pd.DataFrame(
        {
            "基金代码": ["510300", "159915", "018735"],
            "基金简称": ["沪深300ETF", "创业板ETF", "易方达沪深300"],
            "基金类型": ["指数型-股票", "指数型-股票", "混合型-偏股"],
        }
    )

    with patch.object(mod, "_ensure_fund_cache", return_value=mock_df):
        result = await mod.search_fund(keyword="300", limit=10)
    assert result["count"] >= 2  # 510300 by code + 沪深300ETF/易方达沪深300 by name


@pytest.mark.asyncio
async def test_get_fund_nav():
    """get_fund_nav 正确返回净值序列。"""
    import research_agent.mcp_servers.fund_server as mod

    mock_df = pd.DataFrame(
        {
            "净值日期": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "单位净值": [1.0, 1.01, 1.02],
            "日增长率": [0.0, 1.0, 0.99],
        }
    )

    with patch("akshare.fund_open_fund_info_em", return_value=mock_df):
        result = await mod.get_fund_nav(symbol="018735", limit=30)
    assert result["count"] == 3
    assert result["source"] == "eastmoney"


@pytest.mark.asyncio
async def test_get_fund_nav_empty():
    """get_fund_nav 空数据时返回 count=0。"""
    import research_agent.mcp_servers.fund_server as mod

    with patch("akshare.fund_open_fund_info_em", return_value=pd.DataFrame()):
        result = await mod.get_fund_nav(symbol="999999", limit=30)
    assert result["count"] == 0
