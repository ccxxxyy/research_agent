"""MCP Server — Chinese A-share financial data via ``akshare``.

This server is the **data plane** of the financial research pipeline.
It replaces the Phase-0 placeholder ``search_server`` with real, free,
no-API-key finance endpoints backed by ``akshare``, which in turn
aggregates data from 东方财富 / 新浪财经 / 巨潮资讯.

Tools exposed
-------------
1. ``get_stock_basic_info`` — company profile (industry, market cap,
   IPO date, latest price).
2. ``get_stock_price_history`` — daily OHLCV with simple summary stats.
3. ``get_financial_abstract`` — revenue / net income / cash flow /
   EPS by reporting period.
4. ``get_financial_indicators`` — ROE / ROA / margins / leverage
   ratios by reporting period.
5. ``search_stock_by_name`` — fuzzy-match a company name to A-share
   tickers (uses a one-shot in-memory cache to avoid pinging the
   whole-market roster on every invocation).

Design notes
------------
- ``akshare`` is synchronous and I/O-bound. Every tool wraps it in
  ``asyncio.to_thread`` so one slow request does not block the MCP
  stdio event loop for peer requests.
- ``akshare`` sometimes raises ``KeyError`` / ``AttributeError`` /
  ``ValueError`` when the upstream HTML / JSON schema drifts, and a
  bare network failure surfaces as ``ConnectionError`` /
  ``ProxyError``. We catch ``Exception`` at the tool boundary because
  a raising MCP tool would crash the subprocess; instead we return a
  structured ``{"error": "..."}`` payload that the LLM can reason
  about.
- **Multi-source resilience**: the two endpoints that sit on
  ``push2*.eastmoney.com`` (real-time quote + daily K-line) are
  notoriously flaky — they go behind Windows-registry proxy probes,
  get throttled, and occasionally 451 from outside China. For those
  two tools we cascade through alternative providers (雪球, 新浪)
  so an operator gets a usable answer instead of a ProxyError. Each
  response carries a ``source`` field so the caller knows which
  provider actually served it.
- All stock symbols must be the 6-digit code (``300750`` for
  宁德时代). We do NOT accept exchange-prefixed forms (``sh300750``).
  Internally we add/strip the prefix as each upstream requires.
- Column names are kept in Chinese because ``akshare`` returns them
  that way and the downstream LLM (DeepSeek / Qwen) reads Chinese
  fluently. Translating to English would lose information.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from fastmcp import FastMCP

mcp = FastMCP("FinDataAShare")


_ALL_STOCKS_CACHE: pd.DataFrame | None = None
"""Module-level cache for ``ak.stock_info_a_code_name()``.

That call takes ~6 seconds because it scrapes the full A-share
listing. We only need to pay it once per MCP subprocess lifetime;
subsequent ``search_stock_by_name`` calls are pure-pandas filters.
"""


def _fmt_error(exc: Exception, *, context: str) -> dict[str, Any]:
    """Canonical error shape — LLM-readable, no stack traces."""
    return {"error": f"{type(exc).__name__}: {exc}", "context": context}


def _exchange_prefix(symbol: str, *, upper: bool = False) -> str:
    """Return the exchange code — ``sh`` for 6-prefix, ``sz`` otherwise.

    ``akshare`` is inconsistent: 雪球 needs upper case (``SZ300750``),
    新浪/腾讯 need lower case (``sz300750``), and 东财 uses numeric
    market codes. We only build string-prefixed forms here.
    """
    prefix = "sh" if symbol.startswith("6") else "sz"
    return prefix.upper() if upper else prefix


def _prefixed_symbol(symbol: str, *, upper: bool = False) -> str:
    """Return ``sh300750`` / ``SH300750`` / ``sz300750`` / ``SZ300750``."""
    return f"{_exchange_prefix(symbol, upper=upper)}{symbol}"


def _df_to_records(df: pd.DataFrame, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Convert a DataFrame to a list of JSON-safe dicts."""
    if limit is not None:
        df = df.head(limit)
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rec: dict[str, Any] = {}
        for col, val in row.items():
            if pd.isna(val):
                rec[str(col)] = None
            elif isinstance(val, (pd.Timestamp, datetime)):
                rec[str(col)] = val.strftime("%Y-%m-%d")
            elif isinstance(val, (int, float, str, bool)):
                rec[str(col)] = val
            else:
                rec[str(col)] = str(val)
        records.append(rec)
    return records


# ---------------------------------------------------------------------
# Tool 1: Stock basic info (company profile) — multi-source
# ---------------------------------------------------------------------
def _basic_info_from_eastmoney(symbol: str) -> dict[str, Any]:
    import akshare as ak
    df = ak.stock_individual_info_em(symbol=symbol)
    info = dict(zip(df["item"].astype(str), df["value"].tolist()))
    return {"symbol": symbol, "info": info, "source": "eastmoney"}


def _basic_info_from_xueqiu(symbol: str) -> dict[str, Any]:
    import akshare as ak
    df = ak.stock_individual_basic_info_xq(symbol=_prefixed_symbol(symbol, upper=True))
    info = dict(zip(df["item"].astype(str), df["value"].astype(str).tolist()))
    return {"symbol": symbol, "info": info, "source": "xueqiu"}


@mcp.tool()
async def get_stock_basic_info(symbol: str) -> dict:
    """Return company profile for an A-share ticker.

    Tries 东方财富 first (行业/市值/流通股 data), falls back to 雪球
    (which returns a richer 39-field profile with English names,
    registration info, and business scope) if the primary endpoint
    is unreachable.

    Fields typically include: 最新价, 总股本, 流通股, 总市值, 流通市值,
    行业, 上市时间, 股票简称, 股票代码 (eastmoney) OR org_name_cn,
    org_short_name_cn, established_date, main_business, reg_asset,
    listed_date (xueqiu).

    Args:
        symbol: 6-digit ticker, e.g. ``"300750"`` for 宁德时代. Do NOT
            include exchange prefixes like ``sh`` or ``sz``.

    Returns:
        Dictionary with ``symbol``, ``info``, and ``source`` — either
        ``"eastmoney"`` or ``"xueqiu"`` depending on which provider
        served the request. On total failure (both sources down)
        returns ``{"error": ...}``.
    """
    errors: list[str] = []
    for label, fn in (
        ("eastmoney", _basic_info_from_eastmoney),
        ("xueqiu", _basic_info_from_xueqiu),
    ):
        try:
            return await asyncio.to_thread(fn, symbol)
        except Exception as e:
            errors.append(f"{label}: {type(e).__name__}: {str(e)[:100]}")
    return {
        "error": "all providers failed",
        "context": f"get_stock_basic_info(symbol={symbol!r})",
        "attempts": errors,
    }


# ---------------------------------------------------------------------
# Tool 2: Price history with summary stats — multi-source
# ---------------------------------------------------------------------
def _summarize_bars(
    df: pd.DataFrame,
    *,
    date_col: str,
    close_col: str,
    high_col: str,
    low_col: str,
) -> dict[str, Any]:
    """Shared summary-stat builder for different provider schemas."""
    first_close = float(df[close_col].iloc[0])
    last_close = float(df[close_col].iloc[-1])
    pct_change = (last_close - first_close) / first_close * 100 if first_close else 0.0
    return {
        "period_start": str(df[date_col].iloc[0]),
        "period_end": str(df[date_col].iloc[-1]),
        "sessions": int(len(df)),
        "high": float(df[high_col].max()),
        "low": float(df[low_col].min()),
        "pct_change": round(pct_change, 2),
    }


def _price_history_from_eastmoney(symbol: str, days: int, adjust: str) -> dict[str, Any]:
    import akshare as ak
    end = datetime.now()
    start = end - timedelta(days=days)
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        adjust=adjust,
    )
    if df is None or df.empty:
        return {"symbol": symbol, "bars": [], "summary": {"sessions": 0}, "source": "eastmoney"}
    bars = _df_to_records(df)
    summary = _summarize_bars(
        df, date_col="日期", close_col="收盘", high_col="最高", low_col="最低"
    )
    return {"symbol": symbol, "bars": bars, "summary": summary, "source": "eastmoney"}


def _price_history_from_sina(symbol: str, days: int, adjust: str) -> dict[str, Any]:
    import akshare as ak
    end = datetime.now()
    start = end - timedelta(days=days)
    df = ak.stock_zh_a_daily(
        symbol=_prefixed_symbol(symbol),
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        adjust=adjust,
    )
    if df is None or df.empty:
        return {"symbol": symbol, "bars": [], "summary": {"sessions": 0}, "source": "sina"}
    bars = _df_to_records(df)
    summary = _summarize_bars(
        df, date_col="date", close_col="close", high_col="high", low_col="low"
    )
    return {"symbol": symbol, "bars": bars, "summary": summary, "source": "sina"}


@mcp.tool()
async def get_stock_price_history(
    symbol: str,
    days: int = 30,
    adjust: str = "qfq",
) -> dict:
    """Return daily OHLCV bars for the last ``days`` trading sessions.

    Tries 东方财富 first (richest schema: 成交量/成交额/振幅/换手率),
    falls back to 新浪 (simpler schema: date/open/high/low/close/
    volume/amount). Weekends and market holidays are automatically
    skipped by both providers — ``days`` is a *calendar* window, so
    30 calendar days yield ~20 trading sessions.

    Args:
        symbol: 6-digit ticker, e.g. ``"300750"``.
        days: Lookback window in calendar days (default 30, max 365).
        adjust: Price adjustment mode. ``"qfq"`` = forward-adjusted
            (recommended for returns analysis), ``"hfq"`` = backward-
            adjusted, ``""`` = raw prices.

    Returns:
        Dictionary with ``symbol``, ``bars`` (list of daily records —
        keys are Chinese for eastmoney, English for sina), ``summary``
        with ``{period_start, period_end, sessions, high, low,
        pct_change}``, and ``source`` indicating which provider
        answered.
    """
    if days < 1 or days > 365:
        return _fmt_error(
            ValueError(f"days must be in [1, 365], got {days}"),
            context=f"get_stock_price_history(symbol={symbol!r}, days={days})",
        )

    errors: list[str] = []
    for label, fn in (
        ("eastmoney", _price_history_from_eastmoney),
        ("sina", _price_history_from_sina),
    ):
        try:
            return await asyncio.to_thread(fn, symbol, days, adjust)
        except Exception as e:
            errors.append(f"{label}: {type(e).__name__}: {str(e)[:100]}")
    return {
        "error": "all providers failed",
        "context": f"get_stock_price_history(symbol={symbol!r}, days={days}, adjust={adjust!r})",
        "attempts": errors,
    }


# ---------------------------------------------------------------------
# Tool 3: Financial abstract (核心三表摘要)
# ---------------------------------------------------------------------
_ABSTRACT_KEY_METRICS: tuple[str, ...] = (
    "归母净利润",
    "营业总收入",
    "营业成本",
    "净利润",
    "扣非净利润",
    "股东权益合计(净资产)",
    "商誉",
    "经营现金流量净额",
    "基本每股收益",
    "每股净资产",
)
"""Whitelist of rows we surface from ``stock_financial_abstract``.

``akshare`` returns ~50 rows covering every line-item; 90% are noise
for a research-report use case. We only bubble up the rows an analyst
would actually cite.
"""


@mcp.tool()
async def get_financial_abstract(symbol: str, last_n_periods: int = 4) -> dict:
    """Return key financial statement items across recent reporting periods.

    Covers the items analysts cite in research reports: revenue, net
    income, operating cash flow, EPS, and a few balance-sheet anchors.
    Each column is one reporting period (quarterly or annual).

    Args:
        symbol: 6-digit ticker.
        last_n_periods: Number of most-recent reporting periods to
            include (default 4 ≈ one year of quarterly filings,
            max 12).

    Returns:
        Dictionary with ``symbol``, ``periods`` (list of period codes
        like ``"20241231"``), and ``metrics`` (dict of
        ``{metric_name: [value_period_1, value_period_2, ...]}``).
    """
    if last_n_periods < 1 or last_n_periods > 12:
        return _fmt_error(
            ValueError(f"last_n_periods must be in [1, 12], got {last_n_periods}"),
            context=f"get_financial_abstract(symbol={symbol!r})",
        )

    def _call() -> dict[str, Any]:
        import akshare as ak
        df = ak.stock_financial_abstract(symbol=symbol)
        if df is None or df.empty:
            return {"symbol": symbol, "periods": [], "metrics": {}}

        period_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 8]
        period_cols.sort(reverse=True)  # newest first
        period_cols = period_cols[:last_n_periods]

        metrics: dict[str, list[Any]] = {}
        indicator_col = "指标" if "指标" in df.columns else df.columns[1]

        for metric in _ABSTRACT_KEY_METRICS:
            mask = df[indicator_col].astype(str).str.contains(
                metric, na=False, regex=False
            )
            if not mask.any():
                continue
            row = df[mask].iloc[0]
            values: list[Any] = []
            for pc in period_cols:
                val = row.get(pc)
                if pd.isna(val):
                    values.append(None)
                else:
                    try:
                        values.append(float(val))
                    except (TypeError, ValueError):
                        values.append(str(val))
            metrics[metric] = values

        return {
            "symbol": symbol,
            "periods": period_cols,
            "metrics": metrics,
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(
            e,
            context=f"get_financial_abstract(symbol={symbol!r}, last_n_periods={last_n_periods})",
        )


# ---------------------------------------------------------------------
# Tool 4: Financial ratios (ROE/ROA/margins/leverage)
# ---------------------------------------------------------------------
_RATIO_KEY_METRICS: tuple[str, ...] = (
    "净资产收益率",
    "总资产净利润率",
    "销售毛利率",
    "销售净利率",
    "资产负债率",
    "流动比率",
    "速动比率",
    "存货周转率",
    "应收账款周转率",
    "营业利润率",
)


@mcp.tool()
async def get_financial_indicators(symbol: str, start_year: str = "2023") -> dict:
    """Return key financial ratios (ROE, ROA, margins, leverage).

    Args:
        symbol: 6-digit ticker.
        start_year: 4-digit year string, e.g. ``"2023"``. akshare
            returns all reporting periods from this year forward.

    Returns:
        Dictionary with ``symbol``, ``periods`` (list of report dates
        like ``"2024-09-30"``), and ``ratios`` (dict of
        ``{ratio_name: [value_period_1, value_period_2, ...]}``).
        Values may be ``None`` where the upstream source left a hole.
    """
    if len(start_year) != 4 or not start_year.isdigit():
        return _fmt_error(
            ValueError(f"start_year must be a 4-digit year, got {start_year!r}"),
            context=f"get_financial_indicators(symbol={symbol!r})",
        )

    def _call() -> dict[str, Any]:
        import akshare as ak
        df = ak.stock_financial_analysis_indicator(symbol=symbol, start_year=start_year)
        if df is None or df.empty:
            return {"symbol": symbol, "periods": [], "ratios": {}}

        date_col = "日期" if "日期" in df.columns else df.columns[0]
        df = df.sort_values(date_col, ascending=False)
        periods = [str(d) for d in df[date_col].tolist()]

        ratios: dict[str, list[Any]] = {}
        for metric in _RATIO_KEY_METRICS:
            matched_cols = [c for c in df.columns if metric in str(c)]
            if not matched_cols:
                continue
            col = matched_cols[0]
            values: list[Any] = []
            for val in df[col].tolist():
                if pd.isna(val):
                    values.append(None)
                else:
                    try:
                        values.append(round(float(val), 4))
                    except (TypeError, ValueError):
                        values.append(str(val))
            ratios[str(col)] = values

        return {"symbol": symbol, "periods": periods, "ratios": ratios}

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(
            e,
            context=f"get_financial_indicators(symbol={symbol!r}, start_year={start_year!r})",
        )


# ---------------------------------------------------------------------
# Tool 5: Fuzzy search stock by company name
# ---------------------------------------------------------------------
@mcp.tool()
async def search_stock_by_name(keyword: str, limit: int = 10) -> dict:
    """Fuzzy-match a company name to A-share tickers.

    The first call warms an in-memory cache of the whole A-share
    roster (~5k tickers, ~6s one-shot cost). Subsequent calls are
    sub-millisecond pandas filters.

    Args:
        keyword: Partial company name, e.g. ``"宁德"`` or ``"平安"``.
        limit: Max matches to return (default 10, cap 50).

    Returns:
        Dictionary with ``keyword`` and ``matches`` (list of
        ``{"code": "300750", "name": "宁德时代"}``), or
        ``{"error": ...}`` on failure.
    """
    if not keyword.strip():
        return _fmt_error(
            ValueError("keyword must be non-empty"),
            context="search_stock_by_name()",
        )
    limit = max(1, min(limit, 50))

    def _call() -> dict[str, Any]:
        global _ALL_STOCKS_CACHE
        if _ALL_STOCKS_CACHE is None:
            import akshare as ak
            _ALL_STOCKS_CACHE = ak.stock_info_a_code_name()

        df = _ALL_STOCKS_CACHE
        if "name" not in df.columns or "code" not in df.columns:
            raise RuntimeError(
                f"unexpected schema from stock_info_a_code_name: {list(df.columns)}"
            )
        mask = df["name"].astype(str).str.contains(keyword, na=False, regex=False)
        hits = df[mask].head(limit)
        matches = [
            {"code": str(r["code"]), "name": str(r["name"])}
            for _, r in hits.iterrows()
        ]
        return {"keyword": keyword, "matches": matches}

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(
            e,
            context=f"search_stock_by_name(keyword={keyword!r}, limit={limit})",
        )


if __name__ == "__main__":
    mcp.run(transport="stdio")
