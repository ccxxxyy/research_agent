"""MCP Server — 通过 ``yfinance`` 获取美股（股票 / 指数 / ETF）数据。

与 ``fin_data_server``（A 股 / akshare）**平行隔离**，禁止混用。
一期范围：美股普通股、主要指数、ETF（不含共同基金 / 期权）。

工具
----
1. ``get_market_status`` — 美东时段：盘前 / 开盘 / 盘后 / 收盘 / 非交易日
2. ``search_ticker`` — 名称或模糊串 → ticker 候选
3. ``get_quote`` — 单标的最新报价摘要
4. ``get_price_history`` — 日线 OHLCV
5. ``get_basic_info`` — 公司 / ETF 概况
6. ``get_index_quotes`` — 主要美股指数快照
7. ``get_etf_overview`` — ETF 概况（持仓规模、类别等可得字段）
8. ``get_etf_holdings`` — ETF 重仓股（Yahoo top holdings）
9. ``get_etf_sector_weights`` — ETF 行业权重与大类资产占比

设计说明
--------
- ``yfinance`` 为同步 I/O，一律 ``asyncio.to_thread``。
- 错误返回 ``{"error": "...", "context": "..."}``，不抛异常以免弄死 stdio。
- 工具结果走 ``cached_tool`` TTL 分层（与 A 股工具缓存同框架，namespace=``us``）。
- ETF 深化走 ``Ticker.funds_data``（与 A 股 ``fund_get_fund_holdings`` 平行，不混用）。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from fastmcp import FastMCP

from research_agent.cache import (
    TTL_DAILY,
    TTL_LONG,
    TTL_REALTIME,
    cached_tool,
)

logger = logging.getLogger("us_data_server")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

mcp = FastMCP("UsEquityData")

_ET = ZoneInfo("America/New_York")

# 主要指数：显示名 → yfinance 符号
_MAJOR_INDICES: dict[str, str] = {
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "Nasdaq Composite": "^IXIC",
    "Nasdaq-100": "^NDX",
    "Russell 2000": "^RUT",
    "VIX": "^VIX",
}


def _fmt_error(exc: Exception, *, context: str) -> dict[str, Any]:
    logger.error("[%s] %s: %s", context, type(exc).__name__, exc)
    return {"error": f"{type(exc).__name__}: {exc}", "context": context}


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value)


def _normalize_ticker(symbol: str) -> str:
    s = symbol.strip().upper()
    # 常见中文/别名 → yfinance 符号（与 market.detect 表对齐的最小集）
    aliases = {
        "SPX": "^GSPC",
        "SP500": "^GSPC",
        "DJI": "^DJI",
        "IXIC": "^IXIC",
        "NDX": "^NDX",
        "COMP": "^IXIC",
    }
    return aliases.get(s, s)


def _pct_display(weight: Any) -> float | None:
    """将 Yahoo 持仓权重规范为百分比数值（如 0.048 → 4.8）。"""
    if weight is None:
        return None
    try:
        w = float(weight)
    except (TypeError, ValueError):
        return None
    # Yahoo Holding Percent / sector weight 多为 0–1 小数
    if 0.0 <= w <= 1.0:
        return round(w * 100.0, 4)
    return round(w, 4)


def _serialize_top_holdings(df: Any, *, top_n: int) -> list[dict[str, Any]]:
    """把 ``funds_data.top_holdings`` DataFrame 转成 JSON 友好列表。"""
    if df is None:
        return []
    try:
        empty = getattr(df, "empty", True)
    except Exception:  # noqa: BLE001
        return []
    if empty:
        return []

    rows: list[dict[str, Any]] = []
    # 索引为 Symbol；列通常含 Name / Holding Percent
    try:
        limited = df.head(top_n)
    except Exception:  # noqa: BLE001
        limited = df

    for symbol, row in limited.iterrows():
        name = None
        weight_raw = None
        try:
            name = row.get("Name") if hasattr(row, "get") else row["Name"]
        except Exception:  # noqa: BLE001
            name = None
        try:
            weight_raw = (
                row.get("Holding Percent") if hasattr(row, "get") else row["Holding Percent"]
            )
        except Exception:  # noqa: BLE001
            # 兼容列名变体
            for key in ("HoldingPercent", "holdingPercent", "% Assets", "Weight"):
                try:
                    weight_raw = row.get(key) if hasattr(row, "get") else row[key]
                    break
                except Exception:  # noqa: BLE001
                    continue
        weight_pct = _pct_display(weight_raw)
        rows.append(
            {
                "symbol": str(symbol),
                "name": None if name is None else str(name),
                "weight_pct": weight_pct,
                "weight_raw": _json_safe(weight_raw),
            }
        )
    return rows


def _serialize_weight_map(raw: Any) -> list[dict[str, Any]]:
    """把 sector_weightings / asset_classes 的 dict 转成排序后的列表。"""
    if not isinstance(raw, dict) or not raw:
        return []
    items: list[dict[str, Any]] = []
    for key, val in raw.items():
        pct = _pct_display(val)
        items.append(
            {
                "name": str(key),
                "weight_pct": pct,
                "weight_raw": _json_safe(val),
            }
        )
    items.sort(key=lambda x: (x["weight_pct"] is None, -(x["weight_pct"] or 0.0)))
    return items


def _session_status(*, now: datetime | None = None) -> dict[str, Any]:
    """美东交易时段判定（简化：不区分假日日历，仅周末 + 钟点）。"""
    ts = now or datetime.now(tz=_ET)
    local = ts.astimezone(_ET)
    weekday = local.weekday()  # Mon=0
    t = local.time()
    date_str = local.strftime("%Y-%m-%d")
    clock = local.strftime("%H:%M")

    if weekday >= 5:
        return {
            "status": "closed",
            "session": "weekend",
            "local_date": date_str,
            "local_time": clock,
            "timezone": "America/New_York",
            "hint": f"周末休市；请使用上一交易日收盘数据（本地 {date_str} {clock} ET）。",
            "source": "us_session_clock",
        }

    pre_open, regular_open = time(4, 0), time(9, 30)
    regular_close, after_close = time(16, 0), time(20, 0)

    if pre_open <= t < regular_open:
        status, session, hint = (
            "pre_market",
            "pre",
            f"盘前（{clock} ET）；报价可能为盘前价，请标注时段。",
        )
    elif regular_open <= t < regular_close:
        status, session, hint = (
            "open",
            "regular",
            f"常规交易中（截至 {clock} ET 的实时/近实时数据）。",
        )
    elif regular_close <= t < after_close:
        status, session, hint = (
            "after_hours",
            "post",
            f"盘后（{clock} ET）；请标注盘后价与今日收盘价的区别。",
        )
    else:
        status, session, hint = (
            "closed",
            "overnight",
            f"已收盘（{clock} ET）；请使用最近一笔收盘数据并写明日期。",
        )

    return {
        "status": status,
        "session": session,
        "local_date": date_str,
        "local_time": clock,
        "timezone": "America/New_York",
        "hint": hint,
        "source": "us_session_clock",
    }


def _history_records(df: Any, *, limit: int) -> list[dict[str, Any]]:
    if df is None or getattr(df, "empty", True):
        return []
    tail = df.tail(limit)
    records: list[dict[str, Any]] = []
    for idx, row in tail.iterrows():
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
        records.append(
            {
                "date": date_str,
                "open": _json_safe(row.get("Open")),
                "high": _json_safe(row.get("High")),
                "low": _json_safe(row.get("Low")),
                "close": _json_safe(row.get("Close")),
                "volume": _json_safe(row.get("Volume")),
            }
        )
    return records


def _quote_from_ticker(symbol: str) -> dict[str, Any]:
    import yfinance as yf

    ticker = _normalize_ticker(symbol)
    t = yf.Ticker(ticker)
    info = t.info or {}
    # fast_info 在部分版本更稳
    fast: dict[str, Any] = {}
    try:
        fi = t.fast_info
        fast = dict(fi) if fi is not None else {}
    except Exception:  # noqa: BLE001
        fast = {}

    price = (
        fast.get("last_price")
        or info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
    )
    prev = fast.get("previous_close") or info.get("previousClose")
    change_pct = None
    if price is not None and prev not in (None, 0):
        try:
            change_pct = round((float(price) - float(prev)) / float(prev) * 100, 4)
        except (TypeError, ValueError, ZeroDivisionError):
            change_pct = None

    return {
        "symbol": ticker,
        "name": info.get("shortName") or info.get("longName") or ticker,
        "price": _json_safe(price),
        "previous_close": _json_safe(prev),
        "change_percent": change_pct,
        "currency": info.get("currency") or fast.get("currency") or "USD",
        "exchange": info.get("exchange") or info.get("fullExchangeName") or "",
        "quote_type": info.get("quoteType") or "",
        "market_cap": _json_safe(info.get("marketCap")),
        "source": "yfinance",
        "source_url": f"https://finance.yahoo.com/quote/{ticker.lstrip('^')}",
    }


# ---------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------


@mcp.tool()
@cached_tool(ttl=TTL_REALTIME, namespace="us")
async def get_market_status() -> dict:
    """返回美股当前交易状态（美东时间）。

    状态：``open`` / ``pre_market`` / ``after_hours`` / ``closed``。
    回答含"今天/实时/收盘"时必须先调本工具，并按 ``hint`` 标注数据时点。
    """
    try:
        return await asyncio.to_thread(_session_status)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context="get_market_status()")


@mcp.tool()
@cached_tool(ttl=TTL_LONG, namespace="us")
async def search_ticker(query: str, limit: int = 8) -> dict:
    """按公司名 / ETF 名 / ticker 模糊搜索美股标的。

    Args:
        query: 如 ``Apple``、``特斯拉``、``SPY``、``QQQ``。
        limit: 返回条数上限（默认 8，最大 20）。
    """
    limit = max(1, min(int(limit), 20))
    q = query.strip()
    if not q:
        return {"error": "query 不能为空", "context": "search_ticker()"}

    def _call() -> dict[str, Any]:
        import yfinance as yf

        # 中文常见名 → 英文检索串
        zh_map = {
            "苹果": "Apple",
            "特斯拉": "Tesla",
            "微软": "Microsoft",
            "英伟达": "NVIDIA",
            "亚马逊": "Amazon",
            "谷歌": "Alphabet",
            "纳指etf": "QQQ",
            "标普etf": "SPY",
            "标普500": "S&P 500",
            "道琼斯": "Dow Jones",
        }
        search_q = zh_map.get(q, zh_map.get(q.lower(), q))
        results: list[dict[str, Any]] = []
        try:
            # yfinance 0.2+ Search API
            from yfinance import Search

            s = Search(search_q, max_results=limit)
            quotes = getattr(s, "quotes", None) or []
            for item in quotes[:limit]:
                results.append(
                    {
                        "symbol": item.get("symbol"),
                        "name": item.get("shortname") or item.get("longname") or "",
                        "exchange": item.get("exchange") or item.get("exchDisp") or "",
                        "type": item.get("quoteType") or item.get("typeDisp") or "",
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("yfinance.Search failed: %s — fallback Ticker", exc)
            # 回退：把 query 当 ticker 试一次
            ticker = _normalize_ticker(search_q)
            info = yf.Ticker(ticker).info or {}
            if info.get("regularMarketPrice") or info.get("shortName"):
                results.append(
                    {
                        "symbol": ticker,
                        "name": info.get("shortName") or info.get("longName") or ticker,
                        "exchange": info.get("exchange") or "",
                        "type": info.get("quoteType") or "",
                    }
                )

        return {
            "query": q,
            "results": results,
            "count": len(results),
            "source": "yfinance",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context=f"search_ticker({query!r})")


@mcp.tool()
@cached_tool(ttl=TTL_REALTIME, namespace="us")
async def get_quote(symbol: str) -> dict:
    """获取美股 / ETF / 指数最新报价摘要。

    Args:
        symbol: Yahoo Finance ticker，如 ``AAPL``、``SPY``、``^GSPC``。
    """
    try:
        return await asyncio.to_thread(_quote_from_ticker, symbol)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context=f"get_quote({symbol!r})")


@mcp.tool()
@cached_tool(ttl=TTL_DAILY, namespace="us")
async def get_price_history(symbol: str, period: str = "1mo", interval: str = "1d") -> dict:
    """获取美股日线（或指定周期）OHLCV 历史。

    Args:
        symbol: ticker，如 ``TSLA``、``QQQ``。
        period: yfinance period，如 ``5d`` / ``1mo`` / ``3mo`` / ``1y`` / ``5y``。
        interval: ``1d`` / ``1wk`` / ``1mo``（盘中分时可用 ``1h`` / ``5m``，注意延迟）。
    """
    allowed_periods = {"5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
    if period not in allowed_periods:
        return {
            "error": f"period 必须是 {sorted(allowed_periods)} 之一",
            "context": "get_price_history()",
        }

    def _call() -> dict[str, Any]:
        import yfinance as yf

        ticker = _normalize_ticker(symbol)
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        records = _history_records(df, limit=120)
        summary: dict[str, Any] = {}
        if records:
            first, last = records[0], records[-1]
            if first.get("close") and last.get("close"):
                try:
                    ret = (float(last["close"]) - float(first["close"])) / float(first["close"])
                    summary = {
                        "start_close": first["close"],
                        "end_close": last["close"],
                        "return_percent": round(ret * 100, 4),
                        "bars": len(records),
                    }
                except (TypeError, ValueError, ZeroDivisionError):
                    summary = {"bars": len(records)}
        return {
            "symbol": ticker,
            "period": period,
            "interval": interval,
            "bars": records,
            "summary": summary,
            "source": "yfinance",
            "source_url": f"https://finance.yahoo.com/quote/{ticker.lstrip('^')}/history",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context=f"get_price_history({symbol!r})")


@mcp.tool()
@cached_tool(ttl=TTL_LONG, namespace="us")
async def get_basic_info(symbol: str) -> dict:
    """获取公司或 ETF 概况（行业、市值、简介等可得字段）。

    Args:
        symbol: ticker，如 ``MSFT``、``VOO``。
    """

    def _call() -> dict[str, Any]:
        import yfinance as yf

        ticker = _normalize_ticker(symbol)
        info = yf.Ticker(ticker).info or {}
        keys = [
            "shortName",
            "longName",
            "symbol",
            "quoteType",
            "exchange",
            "sector",
            "industry",
            "country",
            "currency",
            "marketCap",
            "enterpriseValue",
            "trailingPE",
            "forwardPE",
            "priceToBook",
            "dividendYield",
            "fiftyTwoWeekHigh",
            "fiftyTwoWeekLow",
            "averageVolume",
            "website",
            "longBusinessSummary",
            "fundFamily",
            "category",
            "totalAssets",
            "navPrice",
            "ytdReturn",
        ]
        slim = {k: _json_safe(info.get(k)) for k in keys if info.get(k) is not None}
        if "longBusinessSummary" in slim and isinstance(slim["longBusinessSummary"], str):
            slim["longBusinessSummary"] = slim["longBusinessSummary"][:800]
        return {
            "symbol": ticker,
            "info": slim,
            "source": "yfinance",
            "source_url": f"https://finance.yahoo.com/quote/{ticker.lstrip('^')}",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context=f"get_basic_info({symbol!r})")


@mcp.tool()
@cached_tool(ttl=TTL_REALTIME, namespace="us")
async def get_index_quotes() -> dict:
    """返回主要美股指数最新报价（标普、道指、纳指、纳斯达克100、罗素2000、VIX）。"""

    def _call() -> dict[str, Any]:
        indices = []
        for name, sym in _MAJOR_INDICES.items():
            try:
                q = _quote_from_ticker(sym)
                indices.append(
                    {
                        "name": name,
                        "symbol": sym,
                        "price": q.get("price"),
                        "previous_close": q.get("previous_close"),
                        "change_percent": q.get("change_percent"),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                indices.append({"name": name, "symbol": sym, "error": str(exc)})
        return {
            "indices": indices,
            "count": len(indices),
            "source": "yfinance",
            "source_url": "https://finance.yahoo.com/markets/stocks/",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context="get_index_quotes()")


@mcp.tool()
@cached_tool(ttl=TTL_DAILY, namespace="us")
async def get_etf_overview(symbol: str) -> dict:
    """获取美股 ETF 概况（类别、规模、NAV、今年以来收益等可得字段）。

    Args:
        symbol: ETF ticker，如 ``SPY``、``QQQ``、``IWM``、``VOO``。
    """

    def _call() -> dict[str, Any]:
        import yfinance as yf

        ticker = _normalize_ticker(symbol)
        info = yf.Ticker(ticker).info or {}
        quote_type = str(info.get("quoteType") or "")
        if quote_type and quote_type.upper() not in {"ETF", "MUTUALFUND", "INDEX"}:
            # 仍返回数据，但标注可能不是 ETF
            pass
        overview = {
            "symbol": ticker,
            "name": info.get("shortName") or info.get("longName"),
            "quote_type": quote_type,
            "category": info.get("category"),
            "fund_family": info.get("fundFamily"),
            "total_assets": _json_safe(info.get("totalAssets")),
            "nav_price": _json_safe(info.get("navPrice")),
            "ytd_return": _json_safe(info.get("ytdReturn")),
            "three_year_avg_return": _json_safe(info.get("threeYearAverageReturn")),
            "expense_hint": _json_safe(info.get("annualReportExpenseRatio")),
            "yield": _json_safe(info.get("yield") or info.get("dividendYield")),
            "currency": info.get("currency") or "USD",
        }
        return {
            "etf": {k: v for k, v in overview.items() if v is not None},
            "source": "yfinance",
            "source_url": f"https://finance.yahoo.com/quote/{ticker}",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context=f"get_etf_overview({symbol!r})")


@mcp.tool()
@cached_tool(ttl=TTL_DAILY, namespace="us")
async def get_etf_holdings(symbol: str, top_n: int = 10) -> dict:
    """获取美股 ETF 重仓股（Yahoo Finance top holdings）。

    Args:
        symbol: ETF ticker，如 ``SPY``、``QQQ``、``IWM``、``VOO``。
        top_n: 返回前 N 大持仓，默认 10，范围 1–25。
    """
    n = max(1, min(int(top_n), 25))

    def _call() -> dict[str, Any]:
        import yfinance as yf

        ticker = _normalize_ticker(symbol)
        t = yf.Ticker(ticker)
        funds = getattr(t, "funds_data", None)
        if funds is None:
            return {
                "error": "funds_data unavailable (not an ETF/fund or yfinance too old)",
                "context": f"get_etf_holdings({ticker!r})",
                "symbol": ticker,
            }
        try:
            top = funds.top_holdings
        except Exception as exc:  # noqa: BLE001
            return _fmt_error(exc, context=f"get_etf_holdings({ticker!r}).top_holdings")

        holdings = _serialize_top_holdings(top, top_n=n)
        return {
            "symbol": ticker,
            "holdings": holdings,
            "count": len(holdings),
            "top_n": n,
            "note": "Yahoo 通常仅披露前十大持仓；权重为可得快照，非实时全持仓。",
            "source": "yfinance.funds_data",
            "source_url": f"https://finance.yahoo.com/quote/{ticker}/holdings",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context=f"get_etf_holdings({symbol!r})")


@mcp.tool()
@cached_tool(ttl=TTL_DAILY, namespace="us")
async def get_etf_sector_weights(symbol: str) -> dict:
    """获取美股 ETF 行业权重与大类资产占比（Yahoo funds_data）。

    Args:
        symbol: ETF ticker，如 ``SPY``、``QQQ``、``IWM``、``VOO``。
    """

    def _call() -> dict[str, Any]:
        import yfinance as yf

        ticker = _normalize_ticker(symbol)
        t = yf.Ticker(ticker)
        funds = getattr(t, "funds_data", None)
        if funds is None:
            return {
                "error": "funds_data unavailable (not an ETF/fund or yfinance too old)",
                "context": f"get_etf_sector_weights({ticker!r})",
                "symbol": ticker,
            }
        sectors_raw: Any = None
        assets_raw: Any = None
        try:
            sectors_raw = funds.sector_weightings
        except Exception as exc:  # noqa: BLE001
            return _fmt_error(exc, context=f"get_etf_sector_weights({ticker!r}).sector")
        try:
            assets_raw = funds.asset_classes
        except Exception:  # noqa: BLE001
            assets_raw = None

        sectors = _serialize_weight_map(sectors_raw)
        asset_classes = _serialize_weight_map(assets_raw)
        return {
            "symbol": ticker,
            "sectors": sectors,
            "asset_classes": asset_classes,
            "sector_count": len(sectors),
            "source": "yfinance.funds_data",
            "source_url": f"https://finance.yahoo.com/quote/{ticker}/holdings",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context=f"get_etf_sector_weights({symbol!r})")


if __name__ == "__main__":
    mcp.run(transport="stdio")
