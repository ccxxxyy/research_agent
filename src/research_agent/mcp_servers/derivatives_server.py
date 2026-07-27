"""MCP Server — 国内期货 + 金融/ETF 期权（akshare）。

与美股 ``us_*`` 衍生品工具**平行隔离**，禁止混用。

工具
----
期货：
1. ``search_futures`` — 品种/合约关键词 → 候选
2. ``get_futures_spot`` — 实时/近实时行情（新浪）
3. ``get_futures_daily`` — 日线（新浪主力/连续合约）
4. ``get_main_futures`` — 常用金融+商品主力列表摘要

期权：
5. ``get_etf_option_list`` — ETF 期权到期月 / 合约列表（上交所 sina）
6. ``get_etf_option_spot`` — 单张 ETF 期权实时行情
7. ``get_index_option_spot`` — 股指期权（沪深300/上证50/中证1000）现货摘要
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import pandas as pd
from fastmcp import FastMCP

from research_agent.cache import TTL_DAILY, TTL_REALTIME, cached_tool

logger = logging.getLogger("derivatives_server")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

mcp = FastMCP("CnDerivatives")

# 品种代码 → 显示名（搜码与主力列表）
_FUTURES_CATALOG: tuple[tuple[str, str, str], ...] = (
    ("IF", "沪深300股指期货", "CFFEX"),
    ("IH", "上证50股指期货", "CFFEX"),
    ("IC", "中证500股指期货", "CFFEX"),
    ("IM", "中证1000股指期货", "CFFEX"),
    ("RB", "螺纹钢", "SHFE"),
    ("HC", "热卷", "SHFE"),
    ("CU", "沪铜", "SHFE"),
    ("AU", "沪金", "SHFE"),
    ("AG", "沪银", "SHFE"),
    ("SC", "原油", "INE"),
    ("M", "豆粕", "DCE"),
    ("Y", "豆油", "DCE"),
    ("P", "棕榈油", "DCE"),
    ("I", "铁矿石", "DCE"),
    ("TA", "PTA", "CZCE"),
    ("MA", "甲醇", "CZCE"),
    ("SR", "白糖", "CZCE"),
    ("CF", "棉花", "CZCE"),
)

_INDEX_OPTION_MAP = {
    "沪深300": ("hs300", "io"),
    "io": ("hs300", "io"),
    "hs300": ("hs300", "io"),
    "上证50": ("sz50", "ho"),
    "ho": ("sz50", "ho"),
    "sz50": ("sz50", "ho"),
    "中证1000": ("zz1000", "mo"),
    "mo": ("zz1000", "mo"),
    "zz1000": ("zz1000", "mo"),
}


def _fmt_error(exc: Exception, *, context: str) -> dict[str, Any]:
    logger.error("[%s] %s: %s", context, type(exc).__name__, exc)
    return {"error": f"{type(exc).__name__}: {exc}", "context": context}


def _df_to_records(df: pd.DataFrame | None, *, limit: int | None = None) -> list[dict[str, Any]]:
    if df is None or getattr(df, "empty", True):
        return []
    view = df.head(limit) if limit is not None else df
    records: list[dict[str, Any]] = []
    for _, row in view.iterrows():
        item: dict[str, Any] = {}
        for col, val in row.items():
            if pd.isna(val):
                continue
            if hasattr(val, "item"):
                try:
                    val = val.item()
                except Exception:  # noqa: BLE001
                    val = str(val)
            item[str(col)] = val
        if item:
            records.append(item)
    return records


def _sina_main_symbol(code: str) -> str:
    """新浪日线常用主力写法：RB0 / IF0。"""
    c = code.strip().upper()
    if re.fullmatch(r"[A-Z]{1,2}\d{3,4}", c):
        return c
    if re.fullmatch(r"[A-Z]{1,2}", c):
        return f"{c}0"
    return c


@mcp.tool()
@cached_tool(ttl=TTL_DAILY, namespace="derivatives")
async def search_futures(keyword: str, limit: int = 15) -> dict:
    """按中文名或品种代码搜索国内期货品种。

    Args:
        keyword: 如 ``螺纹``、``IF``、``豆粕``。
        limit: 返回上限（默认 15）。
    """
    limit = max(1, min(int(limit), 50))
    kw = (keyword or "").strip().lower()
    if not kw:
        return {"error": "empty keyword", "context": "search_futures()"}

    hits: list[dict[str, Any]] = []
    for code, name, exchange in _FUTURES_CATALOG:
        blob = f"{code} {name} {exchange}".lower()
        if kw in blob or kw in code.lower() or kw in name.lower():
            hits.append(
                {
                    "code": code,
                    "name": name,
                    "exchange": exchange,
                    "sina_main": _sina_main_symbol(code),
                }
            )
        if len(hits) >= limit:
            break
    return {
        "keyword": keyword,
        "results": hits,
        "count": len(hits),
        "source": "catalog",
        "source_url": "https://vip.stock.finance.sina.com.cn/quotes_service/view/qihuohangqing.html",
    }


@mcp.tool()
@cached_tool(ttl=TTL_REALTIME, namespace="derivatives")
async def get_futures_spot(symbol: str = "螺纹钢") -> dict:
    """国内期货实时/近实时行情（新浪 ``futures_zh_realtime``）。

    Args:
        symbol: 品种中文名或代码提示，如 ``螺纹钢``、``PTA``、``沪金``。
    """

    def _call() -> dict[str, Any]:
        import akshare as ak

        name = (symbol or "").strip() or "螺纹钢"
        # 若传入品种代码，尽量映射到中文名（realtime 接口多用中文品种）
        for code, cname, _ex in _FUTURES_CATALOG:
            if name.upper() == code:
                name = cname
                break
        df = ak.futures_zh_realtime(symbol=name)
        records = _df_to_records(df, limit=40)
        return {
            "symbol": name,
            "quotes": records,
            "count": len(records),
            "source": "sina",
            "source_url": "https://vip.stock.finance.sina.com.cn/quotes_service/view/qihuohangqing.html",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_futures_spot({symbol!r})")


@mcp.tool()
@cached_tool(ttl=TTL_DAILY, namespace="derivatives")
async def get_futures_daily(symbol: str = "RB0", limit: int = 30) -> dict:
    """国内期货日线（新浪 ``futures_zh_daily_sina``）。

    Args:
        symbol: 合约，如 ``RB0``（螺纹主力）、``IF0``、``RB2505``。
        limit: 最近 N 条（默认 30，上限 120）。
    """
    limit = max(1, min(int(limit), 120))
    sym = _sina_main_symbol(symbol)

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.futures_zh_daily_sina(symbol=sym)
        if df is not None and not df.empty:
            df = df.tail(limit)
        records = _df_to_records(df)
        return {
            "symbol": sym,
            "bars": records,
            "count": len(records),
            "source": "sina",
            "source_url": f"https://finance.sina.com.cn/futures/quotes/{sym}.shtml",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_futures_daily({sym!r})")


@mcp.tool()
@cached_tool(ttl=TTL_REALTIME, namespace="derivatives")
async def get_main_futures(limit: int = 12) -> dict:
    """返回常用金融+商品期货品种目录（代码 / 交易所 / 新浪主力写法）。

    Args:
        limit: 返回条数（默认 12，上限为目录长度）。
    """
    limit = max(1, min(int(limit), len(_FUTURES_CATALOG)))
    items = [
        {
            "code": code,
            "name": name,
            "exchange": exchange,
            "sina_main": _sina_main_symbol(code),
        }
        for code, name, exchange in _FUTURES_CATALOG[:limit]
    ]
    return {
        "futures": items,
        "count": len(items),
        "source": "catalog",
        "source_url": "https://vip.stock.finance.sina.com.cn/quotes_service/view/qihuohangqing.html",
        "note": "目录快照；实时价请再用 get_futures_spot / get_futures_daily。",
    }


@mcp.tool()
@cached_tool(ttl=TTL_DAILY, namespace="derivatives")
async def get_etf_option_list(underlying: str = "50ETF") -> dict:
    """上交所 ETF 期权到期月列表（sina）。

    Args:
        underlying: ``50ETF`` / ``300ETF`` / ``500ETF`` 等 sina 标的名。
    """

    def _call() -> dict[str, Any]:
        import akshare as ak

        name = (underlying or "50ETF").strip()
        months = list(ak.option_sse_list_sina(symbol=name, exchange="null") or [])
        return {
            "underlying": name,
            "expirations": months,
            "count": len(months),
            "source": "sina",
            "source_url": "https://stock.finance.sina.com.cn/option/quotes.html",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_etf_option_list({underlying!r})")


@mcp.tool()
@cached_tool(ttl=TTL_REALTIME, namespace="derivatives")
async def get_etf_option_spot(contract: str) -> dict:
    """单张上交所 ETF 期权实时行情（sina 合约代码）。

    Args:
        contract: 期权合约代码，如 ``10003720``（来自 option_sse_codes_sina）。
    """
    code = (contract or "").strip()
    if not code:
        return {"error": "empty contract", "context": "get_etf_option_spot()"}

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.option_sse_spot_price_sina(symbol=code)
        records = _df_to_records(df)
        return {
            "contract": code,
            "quote": records,
            "count": len(records),
            "source": "sina",
            "source_url": "https://stock.finance.sina.com.cn/option/quotes.html",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_etf_option_spot({code!r})")


@mcp.tool()
@cached_tool(ttl=TTL_REALTIME, namespace="derivatives")
async def get_index_option_spot(symbol: str = "沪深300", contract: str = "") -> dict:
    """中金所股指期权现货摘要（sina）。

    Args:
        symbol: ``沪深300`` / ``上证50`` / ``中证1000``（或 io/ho/mo）。
        contract: 具体合约如 ``io2504``；空则仅返回近月列表（若可得）。
    """
    key = (symbol or "沪深300").strip().lower()
    mapped = _INDEX_OPTION_MAP.get(key) or _INDEX_OPTION_MAP.get(symbol.strip())
    if mapped is None:
        # 允许直接传 io/ho/mo
        mapped = _INDEX_OPTION_MAP.get(key[:2]) if len(key) >= 2 else None
    if mapped is None:
        return {
            "error": "unsupported index option symbol",
            "context": f"get_index_option_spot({symbol!r})",
            "hint": "use 沪深300 / 上证50 / 中证1000",
        }
    family, prefix = mapped

    def _call() -> dict[str, Any]:
        import akshare as ak

        list_fn = {
            "hs300": ak.option_cffex_hs300_list_sina,
            "sz50": ak.option_cffex_sz50_list_sina,
            "zz1000": ak.option_cffex_zz1000_list_sina,
        }[family]
        spot_fn = {
            "hs300": ak.option_cffex_hs300_spot_sina,
            "sz50": ak.option_cffex_sz50_spot_sina,
            "zz1000": ak.option_cffex_zz1000_spot_sina,
        }[family]

        months = list(list_fn() or [])
        out: dict[str, Any] = {
            "symbol": symbol,
            "family": family,
            "prefix": prefix,
            "expirations": months[:12],
            "source": "sina",
            "source_url": "https://stock.finance.sina.com.cn/option/quotes.html",
        }
        c = (contract or "").strip()
        if not c and months:
            # 列表元素可能是字符串月份或合约前缀
            first = months[0]
            c = (
                first
                if isinstance(first, str) and first.lower().startswith(prefix)
                else f"{prefix}{first}"
            )
        if c:
            df = spot_fn(symbol=c)
            out["contract"] = c
            out["quotes"] = _df_to_records(df, limit=80)
            out["count"] = len(out["quotes"])
        else:
            out["count"] = 0
            out["note"] = "无可用合约；请传入 contract 如 io2504"
        return out

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_index_option_spot({symbol!r},{contract!r})")


if __name__ == "__main__":
    mcp.run(transport="stdio")
