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

# Yahoo 限流时 yfinance 会刷 "possibly delisted"（多为假阳性），压低噪音
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

mcp = FastMCP("UsEquityData")

_ET = ZoneInfo("America/New_York")

# 单次 yfinance 同步调用硬超时，避免 Yahoo 挂起导致研究流永久「处理中」
_YF_CALL_TIMEOUT_SECONDS = 45.0

# 主要指数：显示名（中英） → yfinance 符号
_MAJOR_INDICES: dict[str, str] = {
    "标普500 (S&P 500)": "^GSPC",
    "道琼斯 (Dow 30)": "^DJI",
    "纳斯达克 (Nasdaq)": "^IXIC",
    "纳指100 (Nasdaq 100)": "^NDX",
    "罗素2000 (Russell 2000)": "^RUT",
    "VIX恐慌 (VIX)": "^VIX",
}


async def _yf_call(fn, *, context: str, timeout: float = _YF_CALL_TIMEOUT_SECONDS) -> dict:
    """在线程池跑同步 yfinance，带超时；超时返回 error dict 而非挂死。"""
    try:
        return await asyncio.wait_for(asyncio.to_thread(fn), timeout=timeout)
    except TimeoutError:
        logger.error("[%s] yfinance call timed out after %.0fs", context, timeout)
        return {
            "error": f"TimeoutError: yfinance call exceeded {timeout:.0f}s",
            "context": context,
        }
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context=context)


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


_WEEKDAYS_CN = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def _session_status(*, now: datetime | None = None) -> dict[str, Any]:
    """美东交易时段判定（简化：不区分假日日历，仅周末 + 钟点）。"""
    ts = now or datetime.now(tz=_ET)
    local = ts.astimezone(_ET)
    weekday = local.weekday()  # Mon=0
    t = local.time()
    date_str = local.strftime("%Y-%m-%d")
    clock = local.strftime("%H:%M")
    weekday_cn = _WEEKDAYS_CN[weekday]
    # 如 2026-07-25 周六 02:41
    local_display = f"{date_str} {weekday_cn} {clock}"

    if weekday >= 5:
        return {
            "status": "closed",
            "session": "weekend",
            "local_date": date_str,
            "local_time": clock,
            "local_weekday": weekday_cn,
            "local_display": local_display,
            "timezone": "America/New_York",
            "hint": (
                f"周末休市（美东 {local_display}）。"
                "数据源仍可用：上一交易日收盘价、日线历史、公司概况、SEC 披露与新闻；"
                "请标注为上一交易日数据后正常回答，不要拒绝提问。"
            ),
            "available_off_hours": True,
            "source": "us_session_clock",
        }

    pre_open, regular_open = time(4, 0), time(9, 30)
    regular_close, after_close = time(16, 0), time(20, 0)

    if pre_open <= t < regular_open:
        status, session, hint = (
            "pre_market",
            "pre",
            f"盘前（美东 {local_display}）；报价可能为盘前价，请标注时段。",
        )
    elif regular_open <= t < regular_close:
        status, session, hint = (
            "open",
            "regular",
            f"常规交易中（截至美东 {local_display} 的实时/近实时数据）。",
        )
    elif regular_close <= t < after_close:
        status, session, hint = (
            "after_hours",
            "post",
            f"盘后（美东 {local_display}）；请标注盘后价与今日收盘价的区别。",
        )
    else:
        status, session, hint = (
            "closed",
            "overnight",
            (
                f"已收盘（美东 {local_display}）。"
                "仍可获取最近收盘价、日线历史、概况、披露与新闻；"
                "请写明数据日期后正常回答，不要因休市拒绝提问。"
            ),
        )

    return {
        "status": status,
        "session": session,
        "local_date": date_str,
        "local_time": clock,
        "local_weekday": weekday_cn,
        "local_display": local_display,
        "timezone": "America/New_York",
        "hint": hint,
        "available_off_hours": True,
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


def _num_or_none(val: Any) -> float | None:
    if val is None:
        return None
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return None
    if fval != fval:  # NaN
        return None
    return fval


def _quote_via_yahoo_chart(symbol: str) -> dict[str, Any] | None:
    """Yahoo chart HTTP 快路径（休市/周末也稳定，避开 yfinance 挂起）。"""
    from urllib.parse import quote

    ticker = _normalize_ticker(symbol)
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quote(ticker, safe='')}?interval=1d&range=5d"
    )
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }
    payload: dict[str, Any] | None = None
    try:
        from curl_cffi import requests as curl_requests

        resp = curl_requests.get(url, headers=headers, impersonate="chrome", timeout=8)
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
                resp = sess.get(url, headers=headers, timeout=8)
                if resp.status_code == 200:
                    payload = resp.json()
            finally:
                sess.close()
        except Exception:  # noqa: BLE001
            return None
    try:
        result = ((payload or {}).get("chart") or {}).get("result") or []
        if not result:
            return None
        meta = result[0].get("meta") or {}
        price = _num_or_none(meta.get("regularMarketPrice")) or _num_or_none(
            meta.get("postMarketPrice")
        )
        closes = (((result[0].get("indicators") or {}).get("quote") or [{}])[0]).get("close") or []
        closes = [float(c) for c in closes if c is not None]

        # 昨收优先用日线倒数第二根（与 yfinance fast_info.previous_close / 看板一致）。
        # 勿优先 chartPreviousClose：周末/休市时该字段常偏离上一交易日收盘
        # （例：^GSPC 现价 7411.98，chartPreviousClose=7457.69→-0.61%，
        #  而正确昨收 7408.30→+0.05%，与看板一致）。
        prev: float | None = None
        if price is not None and len(closes) >= 2:
            last_bar = closes[-1]
            if (
                abs(last_bar - float(price)) < 1e-2
                or abs(last_bar - float(price)) / max(abs(float(price)), 1e-9) < 1e-4
            ):
                prev = closes[-2]
            else:
                # 最新价尚未进最后一根 bar：用最后一根当昨收
                prev = last_bar
        if prev is None:
            prev = _num_or_none(meta.get("previousClose")) or _num_or_none(
                meta.get("chartPreviousClose")
            )
        if price is None and closes:
            price = closes[-1]
            if prev is None and len(closes) >= 2:
                prev = closes[-2]
        if price is None:
            return None
        change = (float(price) - float(prev)) if prev is not None else None
        change_pct = None
        if prev not in (None, 0):
            change_pct = round((float(price) - float(prev)) / float(prev) * 100, 4)
        return {
            "symbol": ticker,
            "price": price,
            "previous_close": prev,
            "change": round(change, 4) if change is not None else None,
            "change_percent": change_pct,
            "source": "yahoo_chart",
        }
    except Exception:  # noqa: BLE001
        return None


# 东财美股/指数 secid：Yahoo 在国内常 403/限流时的稳定回退。
# 100=全球指数；105=NASDAQ；106=NYSE；107=AMEX。
# 注意：100.NDX=纳斯达克综合，100.NDX100=纳斯达克100，二者不可混用。
_EM_US_FIXED_SECIDS: dict[str, str] = {
    "^GSPC": "100.SPX",
    "^DJI": "100.DJIA",
    "^IXIC": "100.NDX",
    "^NDX": "100.NDX100",
    "^RUT": "107.IWM",  # 东财无罗素2000指数，用 IWM ETF 代理
    "^VIX": "107.VIXY",  # 东财无 VIX 现货，用 VIXY ETF 代理
}
# 代理标的展示名 / 实际成交代码（避免把 ETF 价标成指数名）
_EM_US_PROXY_LABELS: dict[str, str] = {
    "^RUT": "罗素2000ETF (IWM)",
    "^VIX": "VIX短期期货ETF (VIXY)",
}
_EM_US_PROXY_INSTRUMENTS: dict[str, str] = {
    "^RUT": "IWM",
    "^VIX": "VIXY",
}
_EM_US_SECID_CACHE: dict[str, str] = dict(_EM_US_FIXED_SECIDS)


def _em_us_ulist(secids: list[str]) -> list[dict[str, Any]]:
    """东财 ulist 批量行情；``secids`` 如 ``105.AAPL,107.SPY``。"""
    from urllib.parse import urlencode

    if not secids:
        return []
    params = {
        "fltt": "2",
        "secids": ",".join(secids),
        "fields": "f12,f14,f2,f3,f4,f18",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
    }
    hosts = (
        "https://push2delay.eastmoney.com/api/qt/ulist.np/get",
        "https://push2.eastmoney.com/api/qt/ulist.np/get",
        "https://88.push2.eastmoney.com/api/qt/ulist.np/get",
    )
    try:
        from curl_cffi import requests as curl_requests

        qs = urlencode(params)
        for base in hosts:
            try:
                resp = curl_requests.get(f"{base}?{qs}", impersonate="chrome", timeout=8)
                if resp.status_code != 200:
                    continue
                diff = ((resp.json().get("data") or {}).get("diff")) or []
                if diff:
                    return diff
            except Exception:  # noqa: BLE001
                continue
    except ImportError:
        pass
    try:
        import requests

        sess = requests.Session()
        sess.trust_env = False
        try:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://quote.eastmoney.com/",
            }
            for base in hosts:
                try:
                    r = sess.get(base, params=params, timeout=8, headers=headers)
                    diff = ((r.json().get("data") or {}).get("diff")) or []
                    if diff:
                        return diff
                except Exception:  # noqa: BLE001
                    continue
        finally:
            sess.close()
    except Exception:  # noqa: BLE001
        return []
    return []


def _resolve_eastmoney_us_secid(symbol: str) -> str | None:
    """解析东财美股 secid；命中后写入进程缓存。"""
    ticker = _normalize_ticker(symbol)
    if ticker in _EM_US_SECID_CACHE:
        return _EM_US_SECID_CACHE[ticker]
    bare = ticker.lstrip("^")
    if not bare or not bare.replace(".", "").isalnum():
        return None
    # 常见 ETF 在 AMEX(107)；个股多在 NASDAQ(105)/NYSE(106)
    for mkt in ("105", "107", "106"):
        secid = f"{mkt}.{bare}"
        diff = _em_us_ulist([secid])
        if not diff:
            continue
        price = _num_or_none(diff[0].get("f2"))
        if price is None:
            continue
        _EM_US_SECID_CACHE[ticker] = secid
        return secid
    return None


def _quote_via_eastmoney_us(symbol: str) -> dict[str, Any] | None:
    """东财美股/指数报价（Yahoo 403 或限流时的国内可达回退）。"""
    ticker = _normalize_ticker(symbol)
    secid = _resolve_eastmoney_us_secid(ticker)
    if not secid:
        return None
    diff = _em_us_ulist([secid])
    if not diff:
        return None
    row = diff[0]
    price = _num_or_none(row.get("f2"))
    if price is None:
        return None
    prev = _num_or_none(row.get("f18"))
    change = _num_or_none(row.get("f4"))
    change_pct = _num_or_none(row.get("f3"))
    if change is None and price is not None and prev is not None:
        change = float(price) - float(prev)
    if change_pct is None and price is not None and prev not in (None, 0):
        change_pct = round((float(price) - float(prev)) / float(prev) * 100, 4)
    return {
        "symbol": ticker,
        "price": price,
        "previous_close": prev,
        "change": round(float(change), 4) if change is not None else None,
        "change_percent": round(float(change_pct), 4) if change_pct is not None else None,
        "name_cn": str(row.get("f14") or "").strip() or None,
        "source": "eastmoney_us",
    }


def _quote_from_ticker(symbol: str) -> dict[str, Any]:
    """报价：Yahoo Chart → 东财美股 → yfinance。

    国内网络下 Yahoo 常返回 403 / ``Too Many Requests``；东财 ulist 作稳定回退。
    """
    ticker = _normalize_ticker(symbol)
    display = {
        "^GSPC": "标普500 (S&P 500)",
        "^DJI": "道琼斯 (Dow 30)",
        "^IXIC": "纳斯达克 (Nasdaq)",
        "^NDX": "纳指100 (Nasdaq 100)",
        "^RUT": "罗素2000 (Russell 2000)",
        "^VIX": "VIX恐慌 (VIX)",
    }
    session = _session_status()
    as_of_note = {
        "open": "常规交易时段近实时/延迟报价",
        "pre_market": "可能为盘前价，请与常规收盘价区分",
        "after_hours": "可能为盘后价，请与今日常规收盘价区分",
        "closed": "非交易时段：一般为最近一笔常规收盘价",
    }.get(session.get("status") or "", "最近可得报价")

    def _pack(
        q: dict[str, Any],
        *,
        source_url: str,
        name: str | None = None,
        note: str | None = None,
        proxy: bool = False,
        proxy_of: str | None = None,
        quoted_instrument: str | None = None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "symbol": ticker,
            "name": name or display.get(ticker, ticker),
            "price": _json_safe(q.get("price")),
            "previous_close": _json_safe(q.get("previous_close")),
            "change": _json_safe(q.get("change")),
            "change_percent": q.get("change_percent"),
            "currency": "USD",
            "exchange": "",
            "quote_type": "INDEX" if ticker.startswith("^") else "",
            "market_cap": None,
            "market_status": session.get("status"),
            "session": session.get("session"),
            "as_of_note": note or as_of_note,
            "local_display": session.get("local_display"),
            "available_off_hours": True,
            "source": q.get("source") or "unknown",
            "source_url": source_url,
            "proxy": proxy,
        }
        if proxy:
            out["proxy_of"] = proxy_of or ticker
            out["quoted_instrument"] = quoted_instrument or ""
            out["warning"] = (
                f"非 {ticker} 指数现货：东财无该指数码，当前价为代理 "
                f"{quoted_instrument or 'ETF'} 行情，不可写作「{display.get(ticker, ticker)}」收盘价。"
            )
        return out

    chart = _quote_via_yahoo_chart(ticker)
    if chart and chart.get("price") is not None:
        return _pack(
            chart,
            source_url=f"https://finance.yahoo.com/quote/{ticker.lstrip('^')}",
        )

    em = _quote_via_eastmoney_us(ticker)
    if em and em.get("price") is not None:
        proxy_name = _EM_US_PROXY_LABELS.get(ticker)
        proxy_inst = _EM_US_PROXY_INSTRUMENTS.get(ticker)
        packed = _pack(
            em,
            source_url=(
                f"https://quote.eastmoney.com/us/{proxy_inst}.html"
                if proxy_inst
                else f"https://quote.eastmoney.com/us/{ticker.lstrip('^')}.html"
            ),
            name=proxy_name,
            note=(
                f"{as_of_note}；东财无 {ticker} 指数现货，已用 {proxy_inst} ETF 代理；"
                f"**禁止**称为 {display.get(ticker, ticker)} 收盘价"
                if proxy_name
                else as_of_note
            ),
            proxy=bool(proxy_name),
            proxy_of=ticker if proxy_name else None,
            quoted_instrument=proxy_inst,
        )
        if not ticker.startswith("^") and em.get("name_cn"):
            packed["name_em"] = em["name_cn"]
        return packed

    # 回退 yfinance（可能慢/限流；国内常与 Chart 一同失败）
    import yfinance as yf

    t = yf.Ticker(ticker)
    price: Any = None
    prev: Any = None
    try:
        fi = t.fast_info
        if fi is not None:
            price = _num_or_none(getattr(fi, "last_price", None))
            prev = _num_or_none(getattr(fi, "previous_close", None)) or _num_or_none(
                getattr(fi, "regular_market_previous_close", None)
            )
    except Exception:  # noqa: BLE001
        pass
    if price is None:
        try:
            h1 = t.history(period="1d", auto_adjust=False)
            if h1 is not None and not h1.empty and "Close" in h1.columns:
                c1 = h1["Close"].dropna()
                if len(c1) >= 1:
                    price = float(c1.iloc[-1])
        except Exception:  # noqa: BLE001
            pass
    if prev is None:
        try:
            hist = t.history(period="5d", auto_adjust=False)
            if hist is not None and not hist.empty and "Close" in hist.columns:
                closes = hist["Close"].dropna()
                if len(closes) >= 2:
                    last = float(closes.iloc[-1])
                    prev_candidate = float(closes.iloc[-2])
                    if price is not None and abs(last - float(price)) < 1e-6:
                        prev = prev_candidate
                    else:
                        prev = last
                elif len(closes) == 1 and price is None:
                    price = float(closes.iloc[-1])
        except Exception:  # noqa: BLE001
            pass

    change_pct = None
    change = None
    if price is not None and prev not in (None, 0):
        try:
            change = float(price) - float(prev)
            change_pct = round(change / float(prev) * 100, 4)
        except (TypeError, ValueError, ZeroDivisionError):
            change_pct = None

    return _pack(
        {
            "price": price,
            "previous_close": prev,
            "change": round(change, 4) if change is not None else None,
            "change_percent": change_pct,
            "source": "yfinance",
        },
        source_url=f"https://finance.yahoo.com/quote/{ticker.lstrip('^')}",
    )


# ---------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------


@mcp.tool()
@cached_tool(ttl=TTL_REALTIME, namespace="us")
async def get_market_status() -> dict:
    """返回美股当前交易状态（美东时间）。

    状态：``open`` / ``pre_market`` / ``after_hours`` / ``closed``。
    回答含"今天/实时/收盘"时必须先调本工具，并按 ``hint`` 标注数据时点。
    纯本地时钟计算，**不**进线程池（避免被挂起的 yfinance 占满 executor 导致误超时）。
    """
    try:
        return _session_status()
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

    return await _yf_call(_call, context=f"search_ticker({query!r})")


@mcp.tool()
@cached_tool(ttl=TTL_REALTIME, namespace="us")
async def get_quote(symbol: str) -> dict:
    """获取美股 / ETF / 指数最新报价摘要。

    Args:
        symbol: Yahoo Finance ticker，如 ``AAPL``、``SPY``、``^GSPC``。
    """
    return await _yf_call(
        lambda: _quote_from_ticker(symbol),
        context=f"get_quote({symbol!r})",
    )


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

    return await _yf_call(_call, context=f"get_price_history({symbol!r})")


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

    return await _yf_call(_call, context=f"get_basic_info({symbol!r})", timeout=60.0)


@mcp.tool()
@cached_tool(ttl=TTL_REALTIME, namespace="us")
async def get_index_quotes() -> dict:
    """返回主要美股指数最新报价（标普、道指、纳指、纳斯达克100、罗素2000、VIX）。

    主路径 Yahoo Chart；失败回退东财。若某项 ``proxy=true``（如 VIX→VIXY），展示名与价均为代理 ETF，**不是**指数现货。
    """

    def _call() -> dict[str, Any]:
        indices: list[dict[str, Any]] = []
        sources: set[str] = set()
        proxies: list[str] = []
        for name, sym in _MAJOR_INDICES.items():
            try:
                q = _quote_from_ticker(sym)
                src = str(q.get("source") or "unknown")
                sources.add(src)
                is_proxy = bool(q.get("proxy"))
                if is_proxy:
                    proxies.append(sym)
                indices.append(
                    {
                        # 代理时必须用工具返回名（VIXY/IWM），禁止沿用「VIX恐慌/罗素2000」
                        "name": q.get("name") or name,
                        "symbol": sym,
                        "price": q.get("price"),
                        "previous_close": q.get("previous_close"),
                        "change_percent": q.get("change_percent"),
                        "as_of_note": q.get("as_of_note"),
                        "market_status": q.get("market_status"),
                        "source": src,
                        "proxy": is_proxy,
                        "proxy_of": q.get("proxy_of"),
                        "quoted_instrument": q.get("quoted_instrument"),
                        "warning": q.get("warning"),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                indices.append({"name": name, "symbol": sym, "error": str(exc)})
        ok = sum(1 for x in indices if x.get("price") is not None)
        src_joined = "+".join(sorted(sources)) or "unknown"
        if sources == {"eastmoney_us"}:
            source_url = "https://quote.eastmoney.com/center/gridlist.html#us_stocks"
        elif "eastmoney_us" in sources:
            # 混合来源时只给一个可点击主链（东财美股列表）；Yahoo 另可由个股 source_url 给出
            source_url = "https://quote.eastmoney.com/center/gridlist.html#us_stocks"
        else:
            source_url = "https://finance.yahoo.com/markets/stocks/"
        return {
            "indices": indices,
            "count": len(indices),
            "ok_count": ok,
            "market_status": _session_status(),
            "source": src_joined,
            "source_url": source_url,
            "proxy_symbols": proxies,
            "disclaimer": (
                "含代理 ETF（非指数现货），回答时必须按 name/warning 表述，禁止写成 VIX/罗素指数收盘价。"
                if proxies
                else None
            ),
        }

    # chart HTTP 很快；给足余量但远低于旧的 90s yfinance 并发
    return await _yf_call(_call, context="get_index_quotes()", timeout=40.0)


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

    return await _yf_call(_call, context=f"get_etf_overview({symbol!r})", timeout=60.0)


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

    return await _yf_call(_call, context=f"get_etf_holdings({symbol!r})", timeout=60.0)


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

    return await _yf_call(_call, context=f"get_etf_sector_weights({symbol!r})", timeout=60.0)


if __name__ == "__main__":
    mcp.run(transport="stdio")
