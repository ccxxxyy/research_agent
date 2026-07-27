"""MCP Server — 美股新闻（Yahoo + 可选 Finnhub / EDGAR 8-K 标题）。

与 ``news_server``（东财/财联社/雪球）**平行隔离**，禁止混用。

工具
----
1. ``get_ticker_news`` — 个股 / ETF 近期新闻（Yahoo → Finnhub；聚类/过滤/标签）
2. ``get_market_news`` — 主要美股指数相关新闻（标普/纳指/道指/VIX）
3. ``get_etf_news`` — 常见 ETF 新闻（SPY/QQQ/IWM/…）
4. ``get_recent_8k_headlines`` — 近期 8-K 标题（SEC submissions，事件向）

设计说明
--------
- Yahoo Search HTTP（失败再 yfinance）+ 可选 Finnhub company-news（``FINNHUB_API_KEY``）。
- 共用 ``us_news_pipeline``：垃圾源过滤、标题相似度聚类、事件关键词标签。
- 错误返回 ``{"error": "...", "context": "..."}``。
- 工具结果走 ``cached_tool``（namespace=``us_news``）。
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

import httpx
from fastmcp import FastMCP

from research_agent.cache import TTL_SHORT, cached_tool

logger = logging.getLogger("us_news_server")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

mcp = FastMCP("UsNewsServer")

# yfinance 新闻拉取硬超时，避免研究流永久等待
_YF_NEWS_TIMEOUT_SECONDS = 40.0

_MAJOR_INDEX_TICKERS: dict[str, str] = {
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "Nasdaq Composite": "^IXIC",
    "VIX": "^VIX",
}


async def _news_call(fn, *, context: str, timeout: float = _YF_NEWS_TIMEOUT_SECONDS) -> dict:
    try:
        return await asyncio.wait_for(asyncio.to_thread(fn), timeout=timeout)
    except TimeoutError:
        logger.error("[%s] news fetch timed out after %.0fs", context, timeout)
        return {
            "error": f"TimeoutError: news fetch exceeded {timeout:.0f}s",
            "context": context,
        }
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context=context)


_DEFAULT_ETFS = ("SPY", "QQQ", "IWM", "DIA", "VOO", "VTI")

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
HTTP_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

_TICKER_CACHE: dict[str, dict[str, str]] | None = None


def _fmt_error(exc: Exception, *, context: str) -> dict[str, Any]:
    logger.error("[%s] %s: %s", context, type(exc).__name__, exc)
    return {"error": f"{type(exc).__name__}: {exc}", "context": context}


def _normalize_ticker(symbol: str) -> str:
    s = symbol.strip().upper()
    aliases = {
        "SPX": "^GSPC",
        "SP500": "^GSPC",
        "DJI": "^DJI",
        "IXIC": "^IXIC",
        "COMP": "^IXIC",
    }
    return aliases.get(s, s)


def _sec_headers() -> dict[str, str]:
    ua = os.environ.get(
        "SEC_USER_AGENT",
        "research-agent/0.1 (edgar-poc; contact@example.com)",
    ).strip()
    return {
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json, */*",
    }


def _pad_cik(cik: str | int) -> str:
    digits = re.sub(r"\D", "", str(cik))
    if not digits:
        raise ValueError(f"invalid CIK: {cik!r}")
    return digits.zfill(10)[-10:]


def _normalize_news_item(raw: Any) -> dict[str, Any] | None:
    """兼容 yfinance 新旧 news 结构。"""
    if not isinstance(raw, dict):
        return None
    content = raw.get("content") if isinstance(raw.get("content"), dict) else raw
    title = content.get("title") or raw.get("title") or content.get("headline") or ""
    if not title:
        return None
    summary = content.get("summary") or content.get("description") or raw.get("summary") or ""
    provider = ""
    prov = content.get("provider") or raw.get("provider")
    if isinstance(prov, dict):
        provider = str(prov.get("displayName") or prov.get("name") or "")
    elif isinstance(prov, str):
        provider = prov
    pub = (
        content.get("pubDate")
        or content.get("displayTime")
        or raw.get("providerPublishTime")
        or raw.get("pubDate")
        or ""
    )
    if isinstance(pub, (int, float)):
        from datetime import UTC, datetime

        try:
            pub = datetime.fromtimestamp(int(pub), tz=UTC).isoformat()
        except (OverflowError, OSError, ValueError):
            pub = str(pub)
    link = ""
    for key in ("clickThroughUrl", "canonicalUrl", "link", "url"):
        val = content.get(key) if key in content else raw.get(key)
        if isinstance(val, dict):
            link = str(val.get("url") or "")
        elif isinstance(val, str):
            link = val
        if link:
            break
    from research_agent.text.urls import sanitize_http_url

    return {
        "title": str(title),
        "summary": str(summary)[:500] if summary else "",
        "publisher": provider,
        "published_at": str(pub),
        "url": sanitize_http_url(link),
        "source": "yfinance",
    }


def _fetch_news_via_yahoo_search(symbol: str, limit: int) -> list[dict[str, Any]]:
    """Yahoo Search HTTP 新闻（与舆情侧同源快路径）。"""
    from urllib.parse import quote

    ticker = _normalize_ticker(symbol)
    url = (
        "https://query1.finance.yahoo.com/v1/finance/search"
        f"?q={quote(ticker)}&quotesCount=0&newsCount={max(limit, 8)}&listsCount=0"
    )
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    payload: dict[str, Any] | None = None
    try:
        from curl_cffi import requests as curl_requests

        resp = curl_requests.get(url, headers=headers, impersonate="chrome", timeout=10)
        if resp.status_code == 200:
            payload = resp.json()
    except Exception:  # noqa: BLE001
        payload = None
    if payload is None:
        try:
            import requests

            sess = requests.Session()
            sess.trust_env = False
            try:
                resp = sess.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    payload = resp.json()
            finally:
                sess.close()
        except Exception:  # noqa: BLE001
            return []

    out: list[dict[str, Any]] = []
    for raw in (payload or {}).get("news") or []:
        title = raw.get("title") or ""
        if not title:
            continue
        provider = raw["publisher"] if isinstance(raw.get("publisher"), str) else ""
        link = ""
        for key in ("link", "url"):
            if isinstance(raw.get(key), str) and raw[key]:
                link = raw[key]
                break
        from research_agent.text.urls import sanitize_http_url

        out.append(
            {
                "title": str(title),
                "summary": "",
                "publisher": provider,
                "published_at": str(raw.get("publishTime") or raw.get("providerPublishTime") or ""),
                "url": sanitize_http_url(link),
                "source": "yahoo_search",
            }
        )
        if len(out) >= limit:
            break
    return out


def _fetch_yahoo_news_raw(symbol: str, limit: int) -> list[dict[str, Any]]:
    """Yahoo Search HTTP，失败再 yfinance.news（未经聚类）。"""
    items = _fetch_news_via_yahoo_search(symbol, limit)
    if items:
        for it in items:
            it.setdefault("provider", "yahoo_search")
            it.setdefault("source", "yahoo_search")
        return items

    import yfinance as yf

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    ticker = _normalize_ticker(symbol)
    try:
        raw_list = yf.Ticker(ticker).news or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance.news failed for %s: %s", ticker, exc)
        return []
    out: list[dict[str, Any]] = []
    for raw in raw_list:
        item = _normalize_news_item(raw)
        if item:
            item["provider"] = "yfinance"
            item["source"] = "yfinance"
            out.append(item)
        if len(out) >= limit:
            break
    return out


def _fetch_ticker_news(symbol: str, limit: int) -> dict[str, Any]:
    """Yahoo + Finnhub → 垃圾过滤 → 聚类 → 事件标签。"""
    from research_agent.mcp_servers.us_news_pipeline import collect_us_news

    ticker = _normalize_ticker(symbol)
    pull = min(40, max(limit * 2, limit + 5))
    yahoo = _fetch_yahoo_news_raw(ticker, pull)
    return collect_us_news(ticker, yahoo_items=yahoo, limit=limit)


async def _http_get_json(url: str) -> Any:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=_sec_headers()) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()


async def _load_ticker_map() -> dict[str, dict[str, str]]:
    global _TICKER_CACHE
    if _TICKER_CACHE is not None:
        return _TICKER_CACHE
    data = await _http_get_json(COMPANY_TICKERS_URL)
    mapping: dict[str, dict[str, str]] = {}
    rows = data.values() if isinstance(data, dict) else (data if isinstance(data, list) else [])
    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        cik_raw = row.get("cik_str") if row.get("cik_str") is not None else row.get("cik")
        name = str(row.get("title") or row.get("name") or "").strip()
        if ticker and cik_raw is not None:
            mapping[ticker] = {"cik10": _pad_cik(cik_raw), "name": name}
    _TICKER_CACHE = mapping
    return mapping


@mcp.tool()
@cached_tool(ttl=TTL_SHORT, namespace="us_news")
async def get_ticker_news(symbol: str, limit: int = 15) -> dict:
    """获取美股个股 / ETF 近期新闻（Yahoo + 可选 Finnhub；已聚类/过滤/贴标签）。

    Args:
        symbol: ticker，如 ``AAPL``、``TSLA``、``SPY``。
        limit: 条数（1–30，默认 15）。
    """
    limit = max(1, min(int(limit), 30))
    ticker = _normalize_ticker(symbol)
    if not ticker:
        return {"error": "symbol 不能为空", "context": "get_ticker_news()"}

    def _call() -> dict[str, Any]:
        bundle = _fetch_ticker_news(ticker, limit)
        items = bundle.get("news") or []
        providers = bundle.get("providers_used") or []
        src = "+".join(providers) if providers else "yahoo"
        out: dict[str, Any] = {
            "symbol": ticker,
            "news": items,
            "count": len(items),
            "source": src,
            "providers_used": providers,
            "raw_count": bundle.get("raw_count"),
            "source_url": f"https://finance.yahoo.com/quote/{ticker.lstrip('^')}/news",
            "available_off_hours": True,
        }
        if bundle.get("note"):
            out["note"] = bundle["note"]
        return out

    return await _news_call(_call, context=f"get_ticker_news({symbol!r})")


@mcp.tool()
@cached_tool(ttl=TTL_SHORT, namespace="us_news")
async def get_market_news(limit_per_index: int = 5) -> dict:
    """获取主要美股指数相关新闻（标普 / 道指 / 纳指 / VIX）。

    Args:
        limit_per_index: 每个指数最多条数（1–10，默认 5）。
    """
    limit_per_index = max(1, min(int(limit_per_index), 10))

    def _call() -> dict[str, Any]:
        # 只拉标普+纳指，避免 4 路 yfinance 新闻把研究流拖死
        focus = {k: v for k, v in _MAJOR_INDEX_TICKERS.items() if v in {"^GSPC", "^IXIC"}}
        buckets = []
        providers: list[str] = []
        for name, sym in focus.items():
            try:
                bundle = _fetch_ticker_news(sym, limit_per_index)
                news = bundle.get("news") or []
                for p in bundle.get("providers_used") or []:
                    if p not in providers:
                        providers.append(p)
                buckets.append(
                    {
                        "index": name,
                        "symbol": sym,
                        "news": news,
                        "count": len(news),
                        "providers_used": bundle.get("providers_used") or [],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                buckets.append({"index": name, "symbol": sym, "error": str(exc), "news": []})
        return {
            "indices": buckets,
            "source": "+".join(providers) if providers else "yahoo",
            "providers_used": providers,
            "source_url": "https://finance.yahoo.com/topic/stock-market-news/",
        }

    return await _news_call(_call, context="get_market_news()", timeout=50.0)


@mcp.tool()
@cached_tool(ttl=TTL_SHORT, namespace="us_news")
async def get_etf_news(symbols: str = "SPY,QQQ,IWM", limit_per_etf: int = 5) -> dict:
    """获取常见美股 ETF 新闻。

    Args:
        symbols: 逗号分隔 ticker，默认 ``SPY,QQQ,IWM``。
        limit_per_etf: 每个 ETF 最多条数（1–10）。
    """
    limit_per_etf = max(1, min(int(limit_per_etf), 10))
    tickers = [t.strip().upper() for t in symbols.split(",") if t.strip()] or list(_DEFAULT_ETFS)
    tickers = tickers[:8]

    def _call() -> dict[str, Any]:
        buckets = []
        providers: list[str] = []
        for sym in tickers:
            try:
                bundle = _fetch_ticker_news(sym, limit_per_etf)
                news = bundle.get("news") or []
                for p in bundle.get("providers_used") or []:
                    if p not in providers:
                        providers.append(p)
                buckets.append(
                    {
                        "symbol": sym,
                        "news": news,
                        "count": len(news),
                        "providers_used": bundle.get("providers_used") or [],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                buckets.append({"symbol": sym, "error": str(exc), "news": []})
        return {
            "etfs": buckets,
            "source": "+".join(providers) if providers else "yahoo",
            "providers_used": providers,
            "source_url": "https://finance.yahoo.com/etfs/",
        }

    return await _news_call(_call, context=f"get_etf_news({symbols!r})", timeout=50.0)


@mcp.tool()
@cached_tool(ttl=TTL_SHORT, namespace="us_news")
async def get_recent_8k_headlines(identifier: str, limit: int = 8) -> dict:
    """列出公司近期 8-K 标题（SEC EDGAR submissions，官方事件向）。

    正文解析请走 ``us_filing_*``；本工具只返回标题级线索。

    Args:
        identifier: ticker（``AAPL``）或 CIK。
        limit: 条数（1–20）。
    """
    limit = max(1, min(int(limit), 20))
    ident = identifier.strip()
    if not ident:
        return {"error": "identifier 不能为空", "context": "get_recent_8k_headlines()"}

    try:
        if re.fullmatch(r"\d{1,10}", ident):
            cik10 = _pad_cik(ident)
            company = None
            ticker = None
        else:
            mapping = await _load_ticker_map()
            hit = mapping.get(ident.upper())
            if not hit:
                return {
                    "error": f"未找到 ticker {ident.upper()!r} 对应的 CIK",
                    "context": f"get_recent_8k_headlines({identifier!r})",
                }
            cik10 = hit["cik10"]
            company = hit["name"]
            ticker = ident.upper()

        payload = await _http_get_json(SUBMISSIONS_URL.format(cik10=cik10))
        recent = (payload.get("filings") or {}).get("recent") or {}
        accessions = recent.get("accessionNumber") or []
        forms = recent.get("form") or []
        dates = recent.get("filingDate") or []
        primaries = recent.get("primaryDocument") or []
        descriptions = recent.get("primaryDocDescription") or []

        headlines: list[dict[str, Any]] = []
        n = min(len(accessions), len(forms), len(dates), len(primaries))
        for i in range(n):
            form = str(forms[i] or "")
            if not form.upper().startswith("8-K"):
                continue
            accession = str(accessions[i])
            primary = str(primaries[i] or "")
            nodash = accession.replace("-", "")
            doc_url = (
                f"{ARCHIVES_BASE}/{int(cik10)}/{nodash}/{primary.lstrip('/')}" if primary else ""
            )
            headlines.append(
                {
                    "form": form,
                    "filing_date": dates[i],
                    "accession": accession,
                    "title": descriptions[i] if i < len(descriptions) else form,
                    "primary_document": primary,
                    "document_url": doc_url,
                }
            )
            if len(headlines) >= limit:
                break

        return {
            "identifier": ident,
            "cik10": cik10,
            "ticker": ticker or (payload.get("tickers") or [None])[0],
            "company": company or payload.get("name"),
            "headlines": headlines,
            "count": len(headlines),
            "source": "data.sec.gov/submissions",
            "source_url": SUBMISSIONS_URL.format(cik10=cik10),
            "note": "仅标题；完整正文请用 us_filing_search_filings / parse_filing_text",
        }
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context=f"get_recent_8k_headlines({identifier!r})")


def reset_ticker_cache_for_tests() -> None:
    global _TICKER_CACHE
    _TICKER_CACHE = None


if __name__ == "__main__":
    mcp.run(transport="stdio")
