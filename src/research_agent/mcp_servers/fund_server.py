"""MCP Server — 通过 ``akshare`` 获取中国公募基金数据。

本服务器专注于基金分析，覆盖 ETF、LOF、开放式基金、货币基金等品种。

提供的工具
----------
1. ``search_fund``          — 按名称/关键词模糊搜索基金代码。
2. ``get_fund_info``        — 基金概况（类型、规模、基金经理、成立日期等）。
3. ``get_fund_nav``         — 开放式基金历史净值（单位净值 + 累计净值）。
4. ``get_fund_etf_spot``    — ETF 基金实时行情排行（成交额/涨跌幅排序）。
5. ``get_fund_lof_spot``    — LOF 基金实时行情排行。
6. ``get_fund_etf_hist``    — 单只 ETF 历史 K 线。
7. ``get_fund_holdings``    — 基金持仓明细（重仓股）。
8. ``get_fund_rating``      — 基金综合评级（上海证券/招商/济安/晨星）。
9. ``get_fund_rank``        — 基金业绩排行（按近1年/3年/5年收益）。
10. ``get_fund_daily``      — 当日全市场开放式基金净值列表。

数据来源
--------
- 东方财富基金网 (fund.eastmoney.com)
- 天天基金网 (fund.eastmoney.com)
- 同花顺基金 (fund.10jqka.com.cn)

设计说明
--------
- ``akshare`` 是同步的，用 ``asyncio.to_thread`` 包装避免阻塞。
- 所有工具在边界捕获 ``Exception``，返回 ``{"error": "..."}``。
- 基金代码为 6 位数字（如 510300、159915）。
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pandas as pd
from fastmcp import FastMCP

for _proxy_key in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
):
    os.environ.pop(_proxy_key, None)
os.environ["NO_PROXY"] = "*"

mcp = FastMCP("FundData")

_FUND_NAME_CACHE: pd.DataFrame | None = None


def _fmt_error(exc: Exception, *, context: str) -> dict[str, Any]:
    return {"error": f"{type(exc).__name__}: {exc}", "context": context}


def _df_to_records(df: pd.DataFrame, *, limit: int | None = None) -> list[dict[str, Any]]:
    if limit is not None:
        df = df.head(limit)
    from datetime import datetime as _dt

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rec: dict[str, Any] = {}
        for col, val in row.items():
            if pd.isna(val):
                rec[str(col)] = None
            elif isinstance(val, (pd.Timestamp, _dt)):
                rec[str(col)] = val.strftime("%Y-%m-%d")
            elif isinstance(val, (int, float, str, bool)):
                rec[str(col)] = val
            else:
                rec[str(col)] = str(val)
        records.append(rec)
    return records


def _ensure_fund_cache() -> pd.DataFrame:
    global _FUND_NAME_CACHE  # noqa: PLW0603
    if _FUND_NAME_CACHE is None:
        import akshare as ak

        _FUND_NAME_CACHE = ak.fund_name_em()
    return _FUND_NAME_CACHE


# =====================================================================
# 工具 1: 基金搜索
# =====================================================================
@mcp.tool()
async def search_fund(keyword: str, limit: int = 10) -> dict:
    """按名称关键词模糊搜索基金，返回匹配的基金代码和名称。

    Args:
        keyword: 搜索关键词，如 ``"沪深300"``、``"科技"``、``"医药"``。
        limit: 返回条目数（默认 10，上限 30）。

    Returns:
        匹配的基金列表，包含基金代码、简称、类型。
    """
    limit = max(1, min(limit, 30))

    def _call() -> dict[str, Any]:
        df = _ensure_fund_cache()
        mask = df["基金简称"].str.contains(keyword, case=False, na=False)
        matched = df[mask].head(limit)
        cols = [c for c in ["基金代码", "基金简称", "基金类型"] if c in matched.columns]
        return {
            "keyword": keyword,
            "funds": _df_to_records(matched[cols] if cols else matched),
            "count": len(matched),
            "source": "eastmoney",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"search_fund(keyword={keyword!r})")


# =====================================================================
# 工具 2: 基金概况
# =====================================================================
@mcp.tool()
async def get_fund_info(symbol: str) -> dict:
    """返回单只基金的概况信息。

    Args:
        symbol: 6 位基金代码，如 ``"510300"``（沪深300ETF）。

    Returns:
        基金类型、成立日期、基金规模、基金经理、管理人、托管人等。
    """

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.fund_overview_em(symbol=symbol)
        info = dict(zip(df["item"].astype(str), df["value"].astype(str).tolist(), strict=False))
        return {"symbol": symbol, "info": info, "source": "eastmoney"}

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_fund_info(symbol={symbol!r})")


# =====================================================================
# 工具 3: 开放式基金历史净值
# =====================================================================
@mcp.tool()
async def get_fund_nav(symbol: str, limit: int = 30) -> dict:
    """返回开放式基金的历史净值数据。

    Args:
        symbol: 6 位基金代码。
        limit: 返回条目数（默认 30，上限 120）。

    Returns:
        日期、单位净值、累计净值、日增长率。
    """
    limit = max(1, min(limit, 120))

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.fund_open_fund_info_em(fund=symbol, indicator="单位净值走势")
        df = df.tail(limit)
        return {
            "symbol": symbol,
            "records": _df_to_records(df),
            "count": len(df),
            "source": "eastmoney",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_fund_nav(symbol={symbol!r})")


# =====================================================================
# 工具 4: ETF 实时行情排行
# =====================================================================
@mcp.tool()
async def get_fund_etf_spot(sort_by: str = "成交额", limit: int = 30) -> dict:
    """返回全市场 ETF 基金实时行情排行。

    Args:
        sort_by: 排序字段 — ``"成交额"``（默认）或 ``"涨跌幅"``。
        limit: 返回条目数（默认 30，上限 100）。

    Returns:
        ETF 列表，包含代码、名称、最新价、涨跌幅、成交额、折溢价率。
    """
    limit = max(1, min(limit, 100))

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.fund_etf_spot_em()
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)
        df = df.head(limit)
        cols = [
            c
            for c in ["代码", "名称", "最新价", "涨跌幅", "成交额", "流通市值", "基金折价率"]
            if c in df.columns
        ]
        return {
            "sort_by": sort_by,
            "etfs": _df_to_records(df[cols] if cols else df),
            "count": len(df),
            "source": "eastmoney",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context="get_fund_etf_spot()")


# =====================================================================
# 工具 5: LOF 实时行情排行
# =====================================================================
@mcp.tool()
async def get_fund_lof_spot(sort_by: str = "成交额", limit: int = 20) -> dict:
    """返回全市场 LOF 基金实时行情排行。

    Args:
        sort_by: 排序字段 — ``"成交额"``（默认）或 ``"涨跌幅"``。
        limit: 返回条目数（默认 20，上限 50）。

    Returns:
        LOF 列表，包含代码、名称、最新价、涨跌幅、成交额。
    """
    limit = max(1, min(limit, 50))

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.fund_lof_spot_em()
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)
        df = df.head(limit)
        cols = [
            c for c in ["代码", "名称", "最新价", "涨跌幅", "成交额", "换手率"] if c in df.columns
        ]
        return {
            "sort_by": sort_by,
            "lofs": _df_to_records(df[cols] if cols else df),
            "count": len(df),
            "source": "eastmoney",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context="get_fund_lof_spot()")


# =====================================================================
# 工具 6: 单只 ETF 历史 K 线
# =====================================================================
@mcp.tool()
async def get_fund_etf_hist(
    symbol: str,
    period: str = "daily",
    limit: int = 60,
) -> dict:
    """返回单只 ETF 的历史 K 线数据。

    Args:
        symbol: 6 位 ETF 代码，如 ``"510300"``。
        period: ``"daily"``（日线）、``"weekly"``（周线）、``"monthly"``（月线）。
        limit: 返回条目数（默认 60，上限 250）。

    Returns:
        包含日期、开盘、收盘、最高、最低、成交量的列表。
    """
    limit = max(1, min(limit, 250))

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.fund_etf_hist_em(symbol=symbol, period=period, adjust="qfq")
        df = df.tail(limit)
        cols = [
            c
            for c in ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "涨跌幅"]
            if c in df.columns
        ]
        return {
            "symbol": symbol,
            "period": period,
            "records": _df_to_records(df[cols] if cols else df),
            "count": len(df),
            "source": "eastmoney",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_fund_etf_hist(symbol={symbol!r})")


# =====================================================================
# 工具 7: 基金持仓（重仓股）
# =====================================================================
@mcp.tool()
async def get_fund_holdings(symbol: str, year: str = "2024") -> dict:
    """返回单只基金的重仓股持仓明细。

    Args:
        symbol: 6 位基金代码。
        year: 年份，如 ``"2024"``。

    Returns:
        持仓股票列表，包含股票代码、名称、持仓市值、占净值比例。
    """

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.fund_portfolio_hold_em(symbol=symbol, date=year)
        if df.empty:
            return {"symbol": symbol, "holdings": [], "source": "eastmoney"}
        latest_date = df["季度"].iloc[0] if "季度" in df.columns else ""
        if "季度" in df.columns:
            df = df[df["季度"] == latest_date]
        cols = [
            c
            for c in ["序号", "股票代码", "股票名称", "占净值比例", "持股数", "持仓市值"]
            if c in df.columns
        ]
        return {
            "symbol": symbol,
            "report_period": str(latest_date),
            "holdings": _df_to_records(df[cols] if cols else df, limit=20),
            "source": "eastmoney",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_fund_holdings(symbol={symbol!r})")


# =====================================================================
# 工具 8: 基金评级
# =====================================================================
@mcp.tool()
async def get_fund_rating(limit: int = 30) -> dict:
    """返回公募基金综合评级排行（上海证券/招商/济安/晨星四家机构）。

    Args:
        limit: 返回条目数（默认 30，上限 100）。

    Returns:
        基金列表，包含代码、名称、5星评级家数、各机构评级、基金类型。
    """
    limit = max(1, min(limit, 100))

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.fund_rating_all()
        if "5星评级家数" in df.columns:
            df = df.sort_values("5星评级家数", ascending=False)
        df = df.head(limit)
        cols = [
            c
            for c in [
                "代码",
                "简称",
                "5星评级家数",
                "上海证券",
                "招商证券",
                "济安金信",
                "晨星评级",
                "类型",
            ]
            if c in df.columns
        ]
        return {
            "ratings": _df_to_records(df[cols] if cols else df),
            "count": len(df),
            "source": "eastmoney",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context="get_fund_rating()")


# =====================================================================
# 工具 9: 基金业绩排行
# =====================================================================
@mcp.tool()
async def get_fund_rank(
    fund_type: str = "全部",
    sort_by: str = "近1年",
    limit: int = 20,
) -> dict:
    """返回公募基金业绩排行榜。

    Args:
        fund_type: 基金类型 — ``"全部"`` / ``"股票型"`` / ``"混合型"``
                   / ``"债券型"`` / ``"指数型"`` / ``"QDII"``。
        sort_by: 排序字段 — ``"近1年"`` / ``"近3年"`` / ``"近5年"``
                 / ``"今年来"`` / ``"近1周"`` / ``"近1月"``。
        limit: 返回条目数（默认 20，上限 50）。

    Returns:
        基金列表，包含代码、名称、单位净值、各周期收益率、基金经理。
    """
    limit = max(1, min(limit, 50))

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.fund_open_fund_rank_em(symbol=fund_type)
        if sort_by in df.columns:
            df[sort_by] = pd.to_numeric(df[sort_by], errors="coerce")
            df = df.sort_values(sort_by, ascending=False)
        df = df.head(limit)
        cols = [
            c
            for c in [
                "基金代码",
                "基金简称",
                "单位净值",
                "今年来",
                "近1周",
                "近1月",
                "近1年",
                "近3年",
            ]
            if c in df.columns
        ]
        return {
            "fund_type": fund_type,
            "sort_by": sort_by,
            "funds": _df_to_records(df[cols] if cols else df),
            "count": len(df),
            "source": "eastmoney",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_fund_rank(fund_type={fund_type!r})")


# =====================================================================
# 工具 10: 当日开放式基金净值列表
# =====================================================================
@mcp.tool()
async def get_fund_daily(fund_type: str = "股票型", limit: int = 30) -> dict:
    """返回当日开放式基金净值列表。

    Args:
        fund_type: ``"股票型"`` / ``"混合型"`` / ``"债券型"`` / ``"指数型"``  / ``"QDII"`` / ``"LOF"`` / ``"FOF"``。
        limit: 返回条目数（默认 30，上限 100）。

    Returns:
        基金列表，包含代码、名称、单位净值、累计净值、日增长率。
    """
    limit = max(1, min(limit, 100))

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.fund_open_fund_daily_em()
        if "基金类型" in df.columns:
            df = df[df["基金类型"].str.contains(fund_type, na=False)]
        if "日增长率" in df.columns:
            df["日增长率"] = pd.to_numeric(df["日增长率"], errors="coerce")
            df = df.sort_values("日增长率", ascending=False)
        df = df.head(limit)
        cols = [
            c
            for c in ["基金代码", "基金简称", "单位净值", "累计净值", "日增长率"]
            if c in df.columns
        ]
        return {
            "fund_type": fund_type,
            "funds": _df_to_records(df[cols] if cols else df),
            "count": len(df),
            "source": "eastmoney",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_fund_daily(fund_type={fund_type!r})")


if __name__ == "__main__":
    mcp.run(transport="stdio")
