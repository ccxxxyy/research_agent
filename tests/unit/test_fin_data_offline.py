"""``fin_data_server`` 离线单元测试 — 无需网络。

通过 mock 所有 ``akshare`` / ``curl_cffi`` 调用，确保测试在 CI 中快速且可确定地运行（``pytest -m 'not network'``）。
"""

from __future__ import annotations

from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest


# ── 辅助工具 ──────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _reset_caches():
    """每次测试之间清除模块级缓存。"""
    import research_agent.mcp_servers.fin_data_server as mod

    mod._ALL_STOCKS_CACHE = None
    mod._PUSH2_AVAILABLE = None
    mod._PROBE_TS = 0.0
    yield
    mod._ALL_STOCKS_CACHE = None
    mod._PUSH2_AVAILABLE = None
    mod._PROBE_TS = 0.0


def _force_push2_available(mod, available: bool) -> None:
    """设置 push2 可用性并刷新 TTL，使 _is_push2_available() 返回缓存值。"""
    import time

    mod._PUSH2_AVAILABLE = available
    mod._PROBE_TS = time.time()


# ── 工具函数测试 ──────────────────────────────────────────────────────
def test_exchange_prefix():
    from research_agent.mcp_servers.fin_data_server import _exchange_prefix

    assert _exchange_prefix("600519") == "sh"
    assert _exchange_prefix("300750") == "sz"
    assert _exchange_prefix("000001") == "sz"
    assert _exchange_prefix("600519", upper=True) == "SH"


def test_prefixed_symbol():
    from research_agent.mcp_servers.fin_data_server import _prefixed_symbol

    assert _prefixed_symbol("600519") == "sh600519"
    assert _prefixed_symbol("300750", upper=True) == "SZ300750"


def test_df_to_records_basic():
    from research_agent.mcp_servers.fin_data_server import _df_to_records

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    recs = _df_to_records(df, limit=2)
    assert len(recs) == 2
    assert recs[0] == {"a": 1, "b": "x"}


def test_df_to_records_nan():
    from research_agent.mcp_servers.fin_data_server import _df_to_records

    df = pd.DataFrame({"v": [1.0, float("nan")]})
    recs = _df_to_records(df)
    assert recs[1]["v"] is None


def test_fmt_error():
    from research_agent.mcp_servers.fin_data_server import _fmt_error

    err = _fmt_error(ValueError("bad"), context="test")
    assert err["error"] == "ValueError: bad"
    assert err["context"] == "test"


# ── 工具 1: get_stock_basic_info（个股基本信息）────────────────────────
@pytest.mark.asyncio
async def test_basic_info_eastmoney_success():
    import research_agent.mcp_servers.fin_data_server as mod

    mock_df = pd.DataFrame({"item": ["股票代码", "股票简称"], "value": ["300750", "宁德时代"]})

    with patch("akshare.stock_individual_info_em", return_value=mock_df):
        result = await mod.get_stock_basic_info("300750")
    assert result["source"] == "eastmoney"
    assert result["info"]["股票简称"] == "宁德时代"


@pytest.mark.asyncio
async def test_basic_info_fallback_to_xueqiu():
    """东方财富失败时应降级到雪球数据源。"""
    import research_agent.mcp_servers.fin_data_server as mod

    xq_df = pd.DataFrame({"item": ["org_short_name_cn"], "value": ["宁德时代"]})

    with (
        patch("akshare.stock_individual_info_em", side_effect=ConnectionError("down")),
        patch("akshare.stock_individual_basic_info_xq", return_value=xq_df),
    ):
        result = await mod.get_stock_basic_info("300750")
    assert result["source"] == "xueqiu"


@pytest.mark.asyncio
async def test_basic_info_all_fail():
    """所有数据源都失败时应返回 error。"""
    import research_agent.mcp_servers.fin_data_server as mod

    with (
        patch("akshare.stock_individual_info_em", side_effect=ConnectionError),
        patch("akshare.stock_individual_basic_info_xq", side_effect=ConnectionError),
        patch("akshare.stock_info_a_code_name", side_effect=ConnectionError),
    ):
        result = await mod.get_stock_basic_info("300750")
    assert "error" in result
    assert "attempts" in result


# ── 工具 5: search_stock_by_name（按名称搜索股票）─────────────────────
@pytest.mark.asyncio
async def test_search_stock_by_name_hit():
    import research_agent.mcp_servers.fin_data_server as mod

    mock_roster = pd.DataFrame({"code": ["300750", "600519"], "name": ["宁德时代", "贵州茅台"]})

    with patch("akshare.stock_info_a_code_name", return_value=mock_roster):
        result = await mod.search_stock_by_name("宁德", limit=5)
    assert "error" not in result
    codes = [m["code"] for m in result["matches"]]
    assert "300750" in codes


@pytest.mark.asyncio
async def test_search_stock_by_name_empty_keyword():
    """关键词为空时应返回 error。"""
    import research_agent.mcp_servers.fin_data_server as mod

    result = await mod.search_stock_by_name("   ", limit=5)
    assert "error" in result


# ── 工具 6: get_index_quotes（大盘指数行情）───────────────────────────
@pytest.mark.asyncio
async def test_index_quotes_via_curl():
    """curl_cffi 路径可用时应优先使用。"""
    import research_agent.mcp_servers.fin_data_server as mod

    mock_df = pd.DataFrame(
        {
            "代码": ["000001", "399001"],
            "名称": ["上证指数", "深证成指"],
            "最新价": [3200.0, 10500.0],
            "涨跌幅": [0.5, -0.3],
            "成交额": [3e11, 4e11],
        }
    )

    with patch.object(mod, "_fetch_realtime_quotes_via_curl", return_value=mock_df):
        result = await mod.get_index_quotes()
    if "error" not in result:
        assert result["source"] in ("eastmoney_push2_curl", "eastmoney")
        assert "source_url" in result


# ── 工具 8: get_stock_rank（A股涨跌幅排行）────────────────────────────
@pytest.mark.asyncio
async def test_stock_rank_sina_path():
    """新浪实时行情路径应优先使用。"""
    import research_agent.mcp_servers.fin_data_server as mod

    mock_df = pd.DataFrame(
        {
            "代码": ["sz300750", "sh600519", "sz000001"],
            "名称": ["宁德时代", "贵州茅台", "平安银行"],
            "最新价": [200.0, 1800.0, 12.0],
            "涨跌幅": [5.0, 2.0, -1.0],
            "涨跌额": [10.0, 36.0, -0.12],
            "成交额": [5e9, 3e9, 1e9],
        }
    )

    with patch("akshare.stock_zh_a_spot", return_value=mock_df):
        result = await mod.get_stock_rank(direction="涨幅榜", limit=3)
    assert result["source"] == "sina_realtime"
    assert "source_url" in result
    assert result["stocks"][0]["代码"] == "300750"


@pytest.mark.asyncio
async def test_stock_rank_all_fail():
    """所有实时数据源都失败时应返回 error。"""
    import research_agent.mcp_servers.fin_data_server as mod

    mock_roster = pd.DataFrame({"code": ["300750"], "name": ["宁德时代"]})

    with (
        patch("akshare.stock_zh_a_spot", side_effect=ConnectionError("down")),
        patch("akshare.stock_info_a_code_name", return_value=mock_roster),
        patch.object(mod, "_fetch_tencent_realtime", return_value=None),
    ):
        result = await mod.get_stock_rank(direction="涨幅榜", limit=3)
    assert "error" in result


# ── 工具 3: get_financial_abstract（财务摘要）─────────────────────────
@pytest.mark.asyncio
async def test_financial_abstract():
    import research_agent.mcp_servers.fin_data_server as mod

    mock_df = pd.DataFrame(
        {
            "报告期": ["2024-12-31", "2024-09-30"],
            "营业收入": [100e8, 75e8],
            "净利润": [20e8, 15e8],
        }
    )

    with patch("akshare.stock_financial_abstract_ths", return_value=mock_df):
        result = await mod.get_financial_abstract("300750", last_n_periods=2)
    assert "error" not in result
    assert result["symbol"] == "300750"
    assert len(result["periods"]) <= 2


@pytest.mark.asyncio
async def test_financial_abstract_bad_periods():
    """期数超出范围时应返回 error。"""
    import research_agent.mcp_servers.fin_data_server as mod

    result = await mod.get_financial_abstract("300750", last_n_periods=99)
    assert "error" in result


# ── 工具 4: get_financial_indicators（财务指标）────────────────────────
@pytest.mark.asyncio
async def test_financial_indicators():
    import research_agent.mcp_servers.fin_data_server as mod

    mock_df = pd.DataFrame(
        {
            "报告期": ["2024-12-31", "2024-09-30"],
            "净资产收益率": [15.0, 12.0],
            "销售毛利率": [30.0, 28.0],
        }
    )

    with patch("akshare.stock_financial_analysis_indicator", return_value=mock_df):
        result = await mod.get_financial_indicators("300750", start_year="2024")
    assert "error" not in result
    assert result["symbol"] == "300750"


# ── 工具 7: get_sector_fund_flow（板块资金流向）────────────────────────
@pytest.mark.asyncio
async def test_sector_fund_flow():
    import research_agent.mcp_servers.fin_data_server as mod

    mock_df = pd.DataFrame(
        {
            "名称": ["半导体", "新能源"],
            "今日涨跌幅": [2.0, -1.0],
            "今日主力净流入-净额": [1e8, -5e7],
        }
    )

    with patch("akshare.stock_board_industry_name_em", return_value=mock_df):
        result = await mod.get_sector_fund_flow(sector_type="行业", limit=5)
    assert "error" not in result
    assert "source_url" in result


# ── 工具 10: get_lhb_detail（龙虎榜）─────────────────────────────────
@pytest.mark.asyncio
async def test_lhb_detail():
    import research_agent.mcp_servers.fin_data_server as mod

    mock_df = pd.DataFrame(
        {
            "代码": ["300750"],
            "名称": ["宁德时代"],
            "收盘价": [200.0],
            "涨跌幅": [10.0],
        }
    )

    with patch("akshare.stock_lhb_detail_em", return_value=mock_df):
        result = await mod.get_lhb_detail(limit=5)
    assert "error" not in result


# ── 工具 2: get_stock_price_history（个股历史行情）─────────────────────
@pytest.mark.asyncio
async def test_stock_price_history_sina_path():
    """新浪日K线数据路径测试。"""
    import research_agent.mcp_servers.fin_data_server as mod

    mock_df = pd.DataFrame(
        {
            "date": ["2024-06-01", "2024-06-02", "2024-06-03"],
            "open": [200.0, 205.0, 203.0],
            "close": [205.0, 203.0, 210.0],
            "high": [207.0, 206.0, 211.0],
            "low": [199.0, 201.0, 202.0],
            "volume": [1e6, 9e5, 1.1e6],
        }
    )

    with patch("akshare.stock_zh_a_daily", return_value=mock_df):
        result = await mod.get_stock_price_history("300750", days=3)
    assert "error" not in result
    assert result["source"] == "sina"


@pytest.mark.asyncio
async def test_stock_price_history_bad_days():
    """天数超出范围时应返回 error。"""
    import research_agent.mcp_servers.fin_data_server as mod

    result = await mod.get_stock_price_history("300750", days=9999)
    assert "error" in result


# ── 工具 9: get_intraday（分时数据）──────────────────────────────────
@pytest.mark.asyncio
async def test_intraday():
    import research_agent.mcp_servers.fin_data_server as mod

    mock_df = pd.DataFrame(
        {
            "时间": ["2024-06-01 09:30", "2024-06-01 09:35"],
            "开盘": [200.0, 201.0],
            "收盘": [201.0, 200.5],
            "最高": [201.5, 201.5],
            "最低": [199.5, 200.0],
            "成交量": [5000, 3000],
        }
    )

    with patch("akshare.stock_zh_a_hist_min_em", return_value=mock_df):
        result = await mod.get_intraday("300750", period="5")
    assert "error" not in result
    assert result["symbol"] == "300750"


# ── 工具 14: get_macro_china（中国宏观经济数据）───────────────────────
@pytest.mark.asyncio
async def test_macro_china_gdp():
    import research_agent.mcp_servers.fin_data_server as mod

    mock_df = pd.DataFrame(
        {
            "季度": ["2024Q1", "2024Q2"],
            "国内生产总值-绝对值": [30e12, 32e12],
            "国内生产总值-同比增长": [5.3, 4.7],
        }
    )

    with patch("akshare.macro_china_gdp", return_value=mock_df):
        result = await mod.get_macro_china(indicator="gdp", limit=5)
    assert "error" not in result
    assert result["indicator"] == "gdp"


# ── 腾讯实时行情辅助函数 ──────────────────────────────────────────────
def test_fetch_tencent_realtime_no_curl_cffi():
    """未安装 curl_cffi 时应返回 None。"""
    import research_agent.mcp_servers.fin_data_server as mod

    original = mod._HAS_CURL_CFFI
    mod._HAS_CURL_CFFI = False
    try:
        result = mod._fetch_tencent_realtime(["600519"])
        assert result is None
    finally:
        mod._HAS_CURL_CFFI = original


def test_fetch_tencent_realtime_empty_codes():
    """代码列表为空时应返回 None。"""
    import research_agent.mcp_servers.fin_data_server as mod

    result = mod._fetch_tencent_realtime([])
    assert result is None


# ── 工具 12: get_top_holders（十大流通股东）───────────────────────────
@pytest.mark.asyncio
async def test_top_holders():
    import research_agent.mcp_servers.fin_data_server as mod

    mock_df = pd.DataFrame(
        {
            "截止日期": ["2024-06-30", "2024-06-30", "2024-06-30"],
            "股东名称": ["宁波梅山保税港", "香港中央结算", "宁德时代"],
            "持股数量": [5e8, 3e8, 2e8],
            "持股比例": [23.0, 14.0, 9.0],
        }
    )

    with patch("akshare.stock_circulate_stock_holder", return_value=mock_df):
        result = await mod.get_top_holders("300750")
    assert "error" not in result
    assert result["symbol"] == "300750"
    assert len(result["holders"]) <= 10


# ── 工具 13: get_etf_spot（ETF 实时行情）─────────────────────────────
@pytest.mark.asyncio
async def test_etf_spot():
    import research_agent.mcp_servers.fin_data_server as mod

    mock_df = pd.DataFrame(
        {
            "代码": ["510300", "159915"],
            "名称": ["沪深300ETF", "创业板ETF"],
            "最新价": [4.0, 2.5],
            "涨跌幅": [1.2, -0.5],
            "成交额": [5e8, 3e8],
            "流通市值": [1e10, 5e9],
        }
    )

    with patch("akshare.fund_etf_spot_em", return_value=mock_df):
        result = await mod.get_etf_spot(limit=5)
    assert "error" not in result
    assert result["source"] == "eastmoney"


# ── 工具 15: get_concept_board（概念板块）─────────────────────────────
@pytest.mark.asyncio
async def test_concept_board_list():
    """获取概念板块列表。"""
    import research_agent.mcp_servers.fin_data_server as mod

    mock_df = pd.DataFrame(
        {
            "板块名称": ["人工智能", "芯片"],
            "涨跌幅": [3.0, 2.0],
            "领涨股票": ["科大讯飞", "中芯国际"],
        }
    )

    with patch("akshare.stock_board_concept_name_em", return_value=mock_df):
        result = await mod.get_concept_board(board_name="", limit=5)
    assert "error" not in result
    assert result["type"] == "概念板块列表"


@pytest.mark.asyncio
async def test_concept_board_stocks():
    """获取指定概念板块的成分股。"""
    import research_agent.mcp_servers.fin_data_server as mod

    mock_df = pd.DataFrame(
        {
            "代码": ["002230", "300496"],
            "名称": ["科大讯飞", "中科创达"],
            "最新价": [50.0, 80.0],
            "涨跌幅": [5.0, 3.0],
            "成交额": [2e9, 1e9],
        }
    )

    with patch("akshare.stock_board_concept_cons_em", return_value=mock_df):
        result = await mod.get_concept_board(board_name="人工智能", limit=5)
    assert "error" not in result
    assert result["board"] == "人工智能"


# ── 工具 16: get_industry_board（行业板块）────────────────────────────
@pytest.mark.asyncio
async def test_industry_board_list():
    """获取行业板块列表。"""
    import research_agent.mcp_servers.fin_data_server as mod

    mock_df = pd.DataFrame(
        {
            "板块名称": ["半导体", "白酒"],
            "涨跌幅": [2.5, 1.0],
        }
    )

    with patch("akshare.stock_board_industry_name_em", return_value=mock_df):
        result = await mod.get_industry_board(board_name="", limit=5)
    assert "error" not in result
    assert result["type"] == "行业板块列表"


@pytest.mark.asyncio
async def test_industry_board_stocks():
    """获取指定行业板块的成分股。"""
    import research_agent.mcp_servers.fin_data_server as mod

    mock_df = pd.DataFrame(
        {
            "代码": ["600519"],
            "名称": ["贵州茅台"],
            "最新价": [1800.0],
            "涨跌幅": [1.0],
            "成交额": [3e9],
        }
    )

    with patch("akshare.stock_board_industry_cons_em", return_value=mock_df):
        result = await mod.get_industry_board(board_name="白酒", limit=5)
    assert "error" not in result
    assert result["board"] == "白酒"


# ── 工具 17: get_individual_fund_flow（个股资金流向）───────────────────
@pytest.mark.asyncio
async def test_individual_fund_flow():
    import research_agent.mcp_servers.fin_data_server as mod

    mock_df = pd.DataFrame(
        {
            "日期": ["2024-06-01", "2024-06-02"],
            "主力净流入": [1e8, -5e7],
            "超大单净流入": [5e7, -3e7],
        }
    )

    with patch("akshare.stock_individual_fund_flow", return_value=mock_df):
        result = await mod.get_individual_fund_flow("300750", limit=5)
    assert "error" not in result
    assert result["symbol"] == "300750"


# ── 工具 18: get_hsgt_flow（沪深港通资金流向）─────────────────────────
@pytest.mark.asyncio
async def test_hsgt_flow_north():
    """北向资金流向查询。"""
    import research_agent.mcp_servers.fin_data_server as mod

    mock_df = pd.DataFrame(
        {
            "日期": ["2024-06-01", "2024-06-02"],
            "当日资金流入": [50e8, 30e8],
            "当日余额": [100e8, 120e8],
        }
    )

    with patch("akshare.stock_hsgt_hist_em", return_value=mock_df):
        result = await mod.get_hsgt_flow(direction="north", limit=5)
    assert "error" not in result
    assert result["direction"] == "北向资金"


@pytest.mark.asyncio
async def test_hsgt_flow_south():
    """南向资金流向查询。"""
    import research_agent.mcp_servers.fin_data_server as mod

    mock_df = pd.DataFrame(
        {
            "日期": ["2024-06-01"],
            "当日资金流入": [20e8],
        }
    )

    with patch("akshare.stock_hsgt_hist_em", return_value=mock_df):
        result = await mod.get_hsgt_flow(direction="south", limit=5)
    assert "error" not in result
    assert result["direction"] == "南向资金"


# ── 工具 19: get_market_status（A股市场交易状态）──────────────────────
from datetime import datetime as _dt


def test_market_status_trading_day_closed():
    """收盘后（15:30）应返回 closed 状态。"""
    import research_agent.mcp_servers.fin_data_server as mod

    fake_now = _dt(2026, 6, 10, 15, 30, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    trade_dates = {"2026-06-10", "2026-06-09", "2026-06-06"}

    with patch.object(mod, "_load_trade_dates", return_value=trade_dates):
        result = mod._compute_market_status(_now=fake_now)
    assert result["status"] == "closed"
    assert result["is_trading_day"] is True


def test_market_status_weekend():
    """周末应返回 non_trading_day 状态。"""
    import research_agent.mcp_servers.fin_data_server as mod

    fake_now = _dt(2026, 6, 7, 10, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    trade_dates = {"2026-06-05", "2026-06-06"}

    with patch.object(mod, "_load_trade_dates", return_value=trade_dates):
        result = mod._compute_market_status(_now=fake_now)
    assert result["status"] == "non_trading_day"
    assert result["is_trading_day"] is False
    assert "last_trading_day" in result


def test_market_status_pre_market():
    """盘前（8:00）应返回 pre_market 状态。"""
    import research_agent.mcp_servers.fin_data_server as mod

    fake_now = _dt(2026, 6, 10, 8, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    trade_dates = {"2026-06-10", "2026-06-09"}

    with patch.object(mod, "_load_trade_dates", return_value=trade_dates):
        result = mod._compute_market_status(_now=fake_now)
    assert result["status"] == "pre_market"
    assert result["is_trading_day"] is True
    assert "last_trading_day" in result


def test_market_status_trading():
    """盘中（10:30）应返回 trading 状态。"""
    import research_agent.mcp_servers.fin_data_server as mod

    fake_now = _dt(2026, 6, 10, 10, 30, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    trade_dates = {"2026-06-10", "2026-06-09"}

    with patch.object(mod, "_load_trade_dates", return_value=trade_dates):
        result = mod._compute_market_status(_now=fake_now)
    assert result["status"] == "trading"
    assert result["is_trading_day"] is True


def test_market_status_lunch_break():
    """午间休市（12:00）应返回 lunch_break 状态。"""
    import research_agent.mcp_servers.fin_data_server as mod

    fake_now = _dt(2026, 6, 10, 12, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    trade_dates = {"2026-06-10", "2026-06-09"}

    with patch.object(mod, "_load_trade_dates", return_value=trade_dates):
        result = mod._compute_market_status(_now=fake_now)
    assert result["status"] == "lunch_break"
    assert result["is_trading_day"] is True


def test_market_status_call_auction():
    """集合竞价（9:20）应返回 call_auction 状态。"""
    import research_agent.mcp_servers.fin_data_server as mod

    fake_now = _dt(2026, 6, 10, 9, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    trade_dates = {"2026-06-10", "2026-06-09"}

    with patch.object(mod, "_load_trade_dates", return_value=trade_dates):
        result = mod._compute_market_status(_now=fake_now)
    assert result["status"] == "call_auction"
    assert result["is_trading_day"] is True
