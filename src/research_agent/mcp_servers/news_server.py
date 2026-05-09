"""MCP Server — A-share financial news / sentiment via ``akshare``.

This is the **news plane** of the financial research pipeline. Where
``fin_data_server`` delivers structured market / fundamentals data and
``pdf_report_server`` delivers official disclosure PDFs, this server
delivers timely **textual** signal: company news, real-time market
flashes, trending topics, and macro digests.

Tools exposed
-------------
1. ``get_stock_news`` — recent news articles for a specific A-share
   ticker (东方财富 individual-stock feed).
2. ``get_market_telegraph`` — real-time market-wide news flashes
   (财联社 telegraph, refreshed every few minutes).
3. ``get_hot_keywords`` — trending keywords / topics surrounding a
   ticker (东方财富 hot-keyword endpoint), useful as a fast sentiment-
   signal proxy.
4. ``get_economic_news`` — daily macro / economic news digest from
   百度财经 (财经早晚报 format).
5. ``get_xueqiu_discussion_hot_rank`` — 雪球沪深「讨论」热度排行榜
   (个股维度). Wraps ``akshare.stock_hot_tweet_xq`` from
   ``stock_feature/stock_hot_xq.py``. Returns **stocks** ranked by
   discussion-activity metrics on xueqiu.com/hq — **not** individual
   forum post titles/bodies.

Why a separate server (vs. extending ``fin_data_server``)?
----------------------------------------------------------
``fin_data_server``'s tools all return numeric / tabular data. The
news tools here return **free-text** payloads at ~10× the size, with
different latency characteristics (no caching is needed — news is
inherently fresh) and a different failure mode (an empty news feed is
NORMAL when nothing happened, whereas an empty K-line is a bug). The
LLM also needs different prompt rules for news vs. price data — see
``NEWS_EXPERT_PROMPT`` in ``agents/specialists.py``.

Multi-source / fallback strategy
--------------------------------
Each tool primarily talks to ONE provider; we do not cascade through
backups the way price endpoints do. Rationale: news payloads are
qualitative, so a missing source is better surfaced honestly to the
agent ("provider X is down, no news available right now") than papered
over with stale data from a fallback. The agent can then decide
whether the user's question demands news or can be answered from the
data / report planes.

Sentiment as the agent's job, not the tool's
--------------------------------------------
We deliberately do NOT ship an ``analyze_sentiment`` tool. Lightweight
keyword-based sentiment is too crude for the financial domain (a
phrase like "毛利率下滑但费用率改善" defies a single positive/negative
label), and an LLM-based sentiment tool would duplicate what the
``news_expert`` agent already does on top of these raw feeds. The
agent reads the news content and reasons about sentiment in its
synthesis — this keeps the tool surface lean and reproducibility high.

Design notes
------------
- ``akshare`` is synchronous and I/O-bound. Each tool wraps it in
  ``asyncio.to_thread`` so a slow upstream does not block the MCP
  stdio event loop.
- Errors are returned as ``{"error": "...", "context": "..."}`` —
  raising would kill the stdio subprocess.
- All upstream column names are kept in Chinese. The LLM tier we
  target (DeepSeek / Qwen / etc.) reads Chinese fluently; translating
  to English would lose information the agent might cite verbatim.
- We bound returned rows by an explicit ``limit`` parameter (default
  modest) to keep individual tool responses comfortably under the
  LLM context window. ``limit`` is capped to ``MAX_LIMIT=100``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from fastmcp import FastMCP

mcp = FastMCP("FinNewsAShare")

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
MAX_LIMIT = 100
"""Hard ceiling on ``limit`` for any tool. Beyond this the LLM context
window starts to suffer (a single ``stock_news_em`` row can be ~500
chars including title + summary).
"""

_SHANGHAI_TZ = timezone(timedelta(hours=8))


def _fmt_error(exc: Exception, *, context: str) -> dict[str, Any]:
    """Canonical error shape — LLM-readable, no stack traces."""
    return {"error": f"{type(exc).__name__}: {exc}", "context": context}


def _df_to_records(
    df: pd.DataFrame, *, limit: int | None = None
) -> list[dict[str, Any]]:
    """Convert a DataFrame to a list of JSON-safe dicts.

    Mirrors the helper in ``fin_data_server`` so news payloads keep
    the same wire shape (Chinese keys, ``None`` for NaN, ISO-format
    dates) — agents can switch between the data / news planes
    without learning two response conventions.
    """
    if limit is not None:
        df = df.head(limit)
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rec: dict[str, Any] = {}
        for col, val in row.items():
            if pd.isna(val):
                rec[str(col)] = None
            elif isinstance(val, (pd.Timestamp, datetime)):
                rec[str(col)] = val.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(val, (int, float, str, bool)):
                rec[str(col)] = val
            else:
                rec[str(col)] = str(val)
        records.append(rec)
    return records


def _today_shanghai_yyyymmdd() -> str:
    """Return today's date in Asia/Shanghai as ``YYYYMMDD``.

    News digests publish on Shanghai-local schedules; using UTC would
    yield an off-by-one date for any call before 08:00 CST.
    """
    return datetime.now(tz=_SHANGHAI_TZ).strftime("%Y%m%d")


def _coerce_limit(limit: int) -> int:
    """Normalise ``limit`` into ``[1, MAX_LIMIT]``."""
    return max(1, min(int(limit), MAX_LIMIT))


# ---------------------------------------------------------------------
# Tool 0 (listed fifth in module doc): 雪球讨论热度榜（个股）
# ---------------------------------------------------------------------
XUEQIU_DISCUSSION_RANKINGS = frozenset({"最热门", "本周新增"})
"""Passed verbatim to ``akshare.stock_hot_tweet_xq(symbol=...)``.

* ``最热门`` — total discussion-intensity rank (API ``order_by=tweet``).
* ``本周新增`` — last-7-day discussion rank (API ``order_by=tweet7d``).

The upstream DataFrame labels the metric column ``关注``; that name is
misleading — akshare maps ``tweet``/``tweet7d`` into it. We rename to
``讨论量`` in the JSON we return.
"""


def _xueqiu_discussion_hot_rank(ranking: str, limit: int) -> dict[str, Any]:
    """Synchronous body for ``get_xueqiu_discussion_hot_rank``."""
    import akshare as ak

    df = ak.stock_hot_tweet_xq(symbol=ranking)
    if df is None or df.empty:
        return {
            "ranking": ranking,
            "count": 0,
            "stocks": [],
            "source": "xueqiu",
            "warning": "no rows returned from stock_hot_tweet_xq",
        }
    # akshare re-uses the column name ``关注`` for tweet / tweet7d counts.
    if "关注" in df.columns:
        df = df.rename(columns={"关注": "讨论量"})
    return {
        "ranking": ranking,
        "count": min(int(len(df)), limit),
        "stocks": _df_to_records(df, limit=limit),
        "source": "xueqiu",
    }


@mcp.tool()
async def get_xueqiu_discussion_hot_rank(ranking: str = "最热门", limit: int = 30) -> dict:
    """雪球沪深「讨论」热度排行榜 — **个股**按讨论活跃度排序。

    Thin wrapper around ``akshare.stock_hot_tweet_xq`` (see
    ``akshare/stock_feature/stock_hot_xq.py``). Each row is **one
    listed stock** (代码 / 简称 / **讨论量** / 最新价), not a user
    post with title and URL. Use it when the user wants "雪球上哪些票
    讨论最火" / "讨论榜"; for Eastmoney headline news use
    ``get_stock_news``; for Cailian flashes use
    ``get_market_telegraph``.

    **Performance:** the upstream implementation paginates through the
    full screener result set — the **first** call can take tens of
    seconds. Subsequent calls in the same MCP subprocess reuse a warm
    ``requests`` session inside akshare but still re-fetch all pages.

    Args:
        ranking: Exactly ``\"最热门\"`` (all-time discussion rank) or
            ``\"本周新增\"`` (last-7-day discussion rank). Any other
            string returns ``{\"error\": ...}`` without calling the
            network.
        limit: Max stocks to return after sorting (default 30, capped at
            ``MAX_LIMIT``=100).

    Returns:
        Dict with ``ranking``, ``count``, ``stocks`` (list of records
        with Chinese keys including ``讨论量``), ``source`` =
        ``\"xueqiu\"``. On failure ``{\"error\": ..., \"context\": ...}``.
    """
    limit = _coerce_limit(limit)
    if ranking not in XUEQIU_DISCUSSION_RANKINGS:
        return _fmt_error(
            ValueError(
                f"ranking must be one of {sorted(XUEQIU_DISCUSSION_RANKINGS)}, "
                f"got {ranking!r}"
            ),
            context=f"get_xueqiu_discussion_hot_rank(ranking={ranking!r})",
        )
    try:
        return await asyncio.to_thread(_xueqiu_discussion_hot_rank, ranking, limit)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(
            e,
            context=(
                f"get_xueqiu_discussion_hot_rank(ranking={ranking!r}, limit={limit})"
            ),
        )


# ---------------------------------------------------------------------
# Tool 1: Individual stock news (东方财富)
# ---------------------------------------------------------------------
def _stock_news_em(symbol: str, limit: int) -> dict[str, Any]:
    import akshare as ak

    df = ak.stock_news_em(symbol=symbol)
    if df is None or df.empty:
        return {
            "symbol": symbol,
            "count": 0,
            "news": [],
            "source": "eastmoney",
            "warning": "no recent news for this ticker",
        }
    return {
        "symbol": symbol,
        "count": min(int(len(df)), limit),
        "news": _df_to_records(df, limit=limit),
        "source": "eastmoney",
    }


@mcp.tool()
async def get_stock_news(symbol: str, limit: int = 20) -> dict:
    """Fetch recent news articles for a specific A-share ticker.

    Backed by 东方财富's individual-stock news feed. Each row typically
    carries: ``关键词`` (the ticker we searched), ``新闻标题``,
    ``新闻内容`` (short summary), ``发布时间``, ``文章来源``,
    ``新闻链接``.

    Args:
        symbol: 6-digit ticker, e.g. ``"300750"`` for 宁德时代. Do NOT
            include exchange prefixes like ``sh`` or ``sz``.
        limit: Max news rows to return (default 20, capped at
            ``MAX_LIMIT``=100). Most tickers have 50–200 news items
            in the feed; the upstream pages internally and we slice
            after the fact.

    Returns:
        Dictionary with ``symbol``, ``count``, ``news`` (list of news
        records), ``source`` (always ``"eastmoney"``). On failure
        returns ``{"error": ..., "context": ...}``.
    """
    limit = _coerce_limit(limit)
    try:
        return await asyncio.to_thread(_stock_news_em, symbol, limit)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(
            e, context=f"get_stock_news(symbol={symbol!r}, limit={limit})"
        )


# ---------------------------------------------------------------------
# Tool 2: Real-time market telegraph (财联社)
# ---------------------------------------------------------------------
TELEGRAPH_CATEGORIES = {"全部", "重点"}
"""Allowed values for ``get_market_telegraph(category=...)``.

The upstream ``akshare.stock_info_global_cls`` endpoint only supports
two filters — ``全部`` (firehose) and ``重点`` (flagged-as-important).
Older akshare releases exposed a richer category set under the name
``stock_telegraph_cls`` but it was retired in 1.18+; we keep the
constraint loud here so an LLM-generated call with ``A股`` / ``宏观``
fails fast with a helpful error rather than silently returning an
empty frame.
"""


def _telegraph_cls(symbol: str, limit: int) -> dict[str, Any]:
    import akshare as ak

    df = ak.stock_info_global_cls(symbol=symbol)
    if df is None or df.empty:
        return {
            "category": symbol,
            "count": 0,
            "telegraph": [],
            "source": "cls",
            "warning": f"no recent flashes in category {symbol!r}",
        }
    return {
        "category": symbol,
        "count": min(int(len(df)), limit),
        "telegraph": _df_to_records(df, limit=limit),
        "source": "cls",
    }


@mcp.tool()
async def get_market_telegraph(category: str = "全部", limit: int = 30) -> dict:
    """Fetch real-time market news flashes from 财联社 (Cailianpress).

    Cailianpress is the Chinese-market analogue of Bloomberg's
    "FIRST WORD" terminal — short timestamped flashes (~50-300 chars
    each) about market-moving events. The feed updates every few
    minutes during trading hours.

    Args:
        category: Filter for upstream feed. Only two values are
            supported by the akshare endpoint we call:
              - ``"全部"`` (default) — all flashes (firehose)
              - ``"重点"``           — only flagged-as-important
            Any other value returns ``{"error": ...}``.
        limit: Max flashes to return (default 30, capped at
            ``MAX_LIMIT``=100). Older items beyond ``limit`` are
            silently dropped.

    Returns:
        Dictionary with ``category``, ``count``, ``telegraph``
        (list of flash records, each typically with ``标题``,
        ``内容``, ``发布日期``, ``发布时间``), and ``source``
        (always ``"cls"``). On failure returns
        ``{"error": ..., "context": ...}``.
    """
    limit = _coerce_limit(limit)
    if category not in TELEGRAPH_CATEGORIES:
        return _fmt_error(
            ValueError(
                f"category must be one of {sorted(TELEGRAPH_CATEGORIES)}, "
                f"got {category!r}"
            ),
            context=f"get_market_telegraph(category={category!r})",
        )
    try:
        return await asyncio.to_thread(_telegraph_cls, category, limit)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(
            e,
            context=f"get_market_telegraph(category={category!r}, limit={limit})",
        )


# ---------------------------------------------------------------------
# Tool 3: Hot keywords / trending topics (东方财富)
# ---------------------------------------------------------------------
def _hot_keywords_em(symbol: str, limit: int) -> dict[str, Any]:
    import akshare as ak

    df = ak.stock_hot_keyword_em(symbol=symbol)
    if df is None or df.empty:
        return {
            "symbol": symbol,
            "count": 0,
            "keywords": [],
            "source": "eastmoney",
            "warning": "no trending keywords for this ticker",
        }
    return {
        "symbol": symbol,
        "count": min(int(len(df)), limit),
        "keywords": _df_to_records(df, limit=limit),
        "source": "eastmoney",
    }


@mcp.tool()
async def get_hot_keywords(symbol: str, limit: int = 10) -> dict:
    """Fetch trending keywords / topics around an A-share ticker.

    Backed by 东方财富's stock_hot_keyword endpoint. The keyword list
    is a fast sentiment / topic-of-conversation proxy: which themes
    (e.g. ``"碳中和"``, ``"业绩预增"``, ``"高管减持"``) are currently
    co-occurring with the ticker on retail forums and analyst feeds.

    Use cases
    ---------
    - "What's currently being discussed about NIO?" → call this first,
      then drill down with ``get_stock_news`` on the keyword that
      stands out.
    - "Has the chip-shortage narrative cooled around SMIC?" → compare
      keyword frequencies across periods (will need multiple calls).

    Args:
        symbol: Exchange-prefixed ticker. UNLIKE the other tools in
            this server, ``stock_hot_keyword_em`` requires an
            ``SH``/``SZ``-prefixed UPPER-CASE form, e.g.
            ``"SZ300750"``. We normalise here so callers can still
            pass plain 6-digit tickers — we add the prefix
            automatically.
        limit: Max keyword rows (default 10, capped at
            ``MAX_LIMIT``=100). Each row typically has ``时间``,
            ``概念名称``, ``概念代码``, ``热度``.

    Returns:
        Dictionary with ``symbol``, ``count``, ``keywords``, and
        ``source``. On failure returns ``{"error": ..., "context": ...}``.
    """
    limit = _coerce_limit(limit)
    bare = symbol.strip().upper()
    if bare.startswith(("SH", "SZ")):
        prefixed = bare
    else:
        prefix = "SH" if bare.startswith("6") else "SZ"
        prefixed = f"{prefix}{bare}"
    try:
        return await asyncio.to_thread(_hot_keywords_em, prefixed, limit)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(
            e,
            context=f"get_hot_keywords(symbol={symbol!r}, limit={limit})",
        )


# ---------------------------------------------------------------------
# Tool 4: Economic news digest (百度财经早晚报)
# ---------------------------------------------------------------------
def _economic_news_baidu(date: str, limit: int) -> dict[str, Any]:
    import akshare as ak

    df = ak.news_economic_baidu(date=date)
    if df is None or df.empty:
        return {
            "date": date,
            "count": 0,
            "news": [],
            "source": "baidu",
            "warning": f"no economic news digest for {date}",
        }
    return {
        "date": date,
        "count": min(int(len(df)), limit),
        "news": _df_to_records(df, limit=limit),
        "source": "baidu",
    }


@mcp.tool()
async def get_economic_news(date: str = "", limit: int = 30) -> dict:
    """Fetch the daily macro / economic news digest (百度财经早晚报).

    The 早晚报 format is a curated digest of macro-policy, central-bank,
    GDP, CPI, exchange-rate, and major-company announcements published
    by 百度财经 twice daily. Compared to ``get_market_telegraph`` it is
    LESS real-time but MORE editorial (each item is hand-picked rather
    than firehosed).

    Args:
        date: ``YYYYMMDD`` string, e.g. ``"20260508"``. Empty
            (default) → today (Asia/Shanghai). Most recent ~30 days
            are reliably available; older dates may return empty.
        limit: Max news rows (default 30, capped at ``MAX_LIMIT``=100).

    Returns:
        Dictionary with ``date`` (the date actually queried), ``count``,
        ``news`` (each typically ``发布日期``, ``发布时间``,
        ``内容``), and ``source`` (``"baidu"``). On failure returns
        ``{"error": ..., "context": ...}``.
    """
    limit = _coerce_limit(limit)
    use_date = date.strip() or _today_shanghai_yyyymmdd()
    if not use_date.isdigit() or len(use_date) != 8:
        return _fmt_error(
            ValueError(
                f"date must be YYYYMMDD (8 digits), got {date!r}; "
                f"pass empty string for today"
            ),
            context=f"get_economic_news(date={date!r})",
        )
    try:
        return await asyncio.to_thread(_economic_news_baidu, use_date, limit)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(
            e,
            context=f"get_economic_news(date={use_date!r}, limit={limit})",
        )


if __name__ == "__main__":
    mcp.run(transport="stdio")
