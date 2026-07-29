"""MCP Server — 通过 ``yfinance`` 获取美股（股票 / 指数 / ETF / 共同基金 / 期货 / 期权）数据。

与 ``fin_data_server``（A 股 / akshare）**平行隔离**，禁止混用。

工具
----
1. ``get_market_status`` — 美东时段：盘前 / 开盘 / 盘后 / 收盘 / 非交易日
2. ``search_ticker`` — 名称或模糊串 → ticker 候选
3. ``get_quote`` — 单标的最新报价摘要（含期货 ``CL=F`` 等）
4. ``get_price_history`` — 日线 OHLCV
5. ``get_basic_info`` — 公司 / ETF / 共同基金概况
6. ``get_index_quotes`` — 主要美股指数快照
7. ``get_etf_overview`` — ETF 概况（持仓规模、类别等可得字段）
8. ``get_etf_holdings`` — ETF 重仓股（Yahoo top holdings）
9. ``get_etf_sector_weights`` — ETF 行业权重与大类资产占比
10. ``get_mutual_fund_overview`` — 共同基金概况
11. ``get_mutual_fund_holdings`` — 共同基金重仓
12. ``get_futures_quotes`` — 常用商品/股指期货批量报价
13. ``get_option_expirations`` — 股票期权到期日列表
14. ``get_option_chain`` — 指定到期日 calls/puts 摘要

设计说明
--------
- ``yfinance`` 为同步 I/O，一律 ``asyncio.to_thread``。
- 错误返回 ``{"error": "...", "context": "..."}``，不抛异常以免弄死 stdio。
- 工具结果走 ``cached_tool`` TTL 分层（与 A 股工具缓存同框架，namespace=``us``）。
- ETF / 共同基金深化走 ``Ticker.funds_data``（与 A 股 ``fund_get_fund_holdings`` 平行，不混用）。
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

# 常用美股/商品期货（Yahoo ``=F`` 连续合约）
_DEFAULT_FUTURES: tuple[tuple[str, str], ...] = (
    ("CL=F", "WTI Crude Oil"),
    ("BZ=F", "Brent Crude"),
    ("GC=F", "Gold"),
    ("SI=F", "Silver"),
    ("ES=F", "E-mini S&P 500"),
    ("NQ=F", "E-mini Nasdaq-100"),
    ("YM=F", "E-mini Dow"),
    ("RTY=F", "E-mini Russell 2000"),
)


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
    clock = local.strftime("%H:%M:%S")
    weekday_cn = _WEEKDAYS_CN[weekday]
    # 如 2026-07-25 周六 02:41:03
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


def _history_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    first, last = records[0], records[-1]
    if first.get("close") and last.get("close"):
        try:
            ret = (float(last["close"]) - float(first["close"])) / float(first["close"])
            return {
                "start_close": first["close"],
                "end_close": last["close"],
                "return_percent": round(ret * 100, 4),
                "bars": len(records),
            }
        except (TypeError, ValueError, ZeroDivisionError):
            return {"bars": len(records)}
    return {"bars": len(records)}


_PERIOD_TO_YAHOO_RANGE: dict[str, str] = {
    "5d": "5d",
    "1mo": "1mo",
    "3mo": "3mo",
    "6mo": "6mo",
    "1y": "1y",
    "2y": "2y",
    "5y": "5y",
    "10y": "10y",
    "ytd": "ytd",
    "max": "max",
}

_PERIOD_TO_EM_LIMIT: dict[str, int] = {
    "5d": 8,
    "1mo": 30,
    "3mo": 70,
    "6mo": 140,
    "1y": 260,
    "2y": 520,
    "5y": 1200,
    "10y": 1200,
    "ytd": 260,
    "max": 1200,
}


def _http_get_json(url: str, *, timeout: float = 10.0) -> dict[str, Any] | None:
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    try:
        from curl_cffi import requests as curl_requests

        resp = curl_requests.get(url, headers=headers, impersonate="chrome", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        pass
    try:
        import requests

        sess = requests.Session()
        sess.trust_env = False
        try:
            resp = sess.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                return data if isinstance(data, dict) else None
        finally:
            sess.close()
    except Exception:  # noqa: BLE001
        return None
    return None


def _history_via_yahoo_chart(symbol: str, *, period: str, interval: str) -> dict[str, Any] | None:
    """Yahoo Chart HTTP 日线兜底（yfinance 挂起/失败时仍可能可用）。"""
    from urllib.parse import quote

    if interval != "1d":
        return None
    range_ = _PERIOD_TO_YAHOO_RANGE.get(period)
    if not range_:
        return None
    ticker = _normalize_ticker(symbol)
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quote(ticker, safe='')}?interval=1d&range={range_}"
    )
    payload = _http_get_json(url, timeout=10.0)
    try:
        result = ((payload or {}).get("chart") or {}).get("result") or []
        if not result:
            return None
        ts = result[0].get("timestamp") or []
        quote = ((result[0].get("indicators") or {}).get("quote") or [{}])[0]
        opens = quote.get("open") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []
        records: list[dict[str, Any]] = []
        for i, t in enumerate(ts):
            c = closes[i] if i < len(closes) else None
            if c is None:
                continue
            date_str = datetime.fromtimestamp(int(t), tz=_ET).strftime("%Y-%m-%d")
            records.append(
                {
                    "date": date_str,
                    "open": _num_or_none(opens[i] if i < len(opens) else None),
                    "high": _num_or_none(highs[i] if i < len(highs) else None),
                    "low": _num_or_none(lows[i] if i < len(lows) else None),
                    "close": _num_or_none(c),
                    "volume": _num_or_none(volumes[i] if i < len(volumes) else None),
                }
            )
        if not records:
            return None
        records = records[-120:]
        return {
            "symbol": ticker,
            "period": period,
            "interval": interval,
            "bars": records,
            "summary": _history_summary(records),
            "source": "yahoo_chart",
            "source_url": f"https://finance.yahoo.com/quote/{ticker.lstrip('^')}/history",
        }
    except Exception:  # noqa: BLE001
        return None


def _history_via_eastmoney(symbol: str, *, period: str, interval: str) -> dict[str, Any] | None:
    """东财美股日线 K 线兜底（Yahoo 全挂时的国内可达路径）。"""
    if interval != "1d":
        return None
    ticker = _normalize_ticker(symbol)
    secid = _resolve_eastmoney_us_secid(ticker)
    if not secid:
        return None
    limit = _PERIOD_TO_EM_LIMIT.get(period, 60)
    from urllib.parse import urlencode

    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "1",
        "end": "20500101",
        "lmt": str(limit),
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
    }
    hosts = (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get",
        "https://push2hisdelay.eastmoney.com/api/qt/stock/kline/get",
    )
    payload: dict[str, Any] | None = None
    for base in hosts:
        payload = _http_get_json(f"{base}?{urlencode(params)}", timeout=10.0)
        if payload and (payload.get("data") or {}).get("klines"):
            break
    try:
        klines = ((payload or {}).get("data") or {}).get("klines") or []
        records: list[dict[str, Any]] = []
        for row in klines:
            parts = str(row).split(",")
            if len(parts) < 6:
                continue
            records.append(
                {
                    "date": parts[0],
                    "open": _num_or_none(parts[1]),
                    "close": _num_or_none(parts[2]),
                    "high": _num_or_none(parts[3]),
                    "low": _num_or_none(parts[4]),
                    "volume": _num_or_none(parts[5]),
                }
            )
        if not records:
            return None
        records = records[-120:]
        proxy = _EM_US_PROXY_INSTRUMENTS.get(ticker)
        note = None
        if proxy:
            note = f"东财无该指数直连 K 线，使用代理标的 {proxy} 日线。"
        return {
            "symbol": ticker,
            "period": period,
            "interval": interval,
            "bars": records,
            "summary": _history_summary(records),
            "source": "eastmoney_us_kline",
            "source_url": (f"https://quote.eastmoney.com/us/{proxy or ticker.lstrip('^')}.html"),
            **({"note": note} if note else {}),
        }
    except Exception:  # noqa: BLE001
        return None


def _holdings_via_yahoo_quotesummary(symbol: str, *, top_n: int) -> dict[str, Any] | None:
    """Yahoo quoteSummary topHoldings HTTP（绕过 yfinance.funds_data）。"""
    from urllib.parse import quote

    ticker = _normalize_ticker(symbol)
    url = (
        "https://query1.finance.yahoo.com/v10/finance/quoteSummary/"
        f"{quote(ticker, safe='')}?modules=topHoldings"
    )
    payload = _http_get_json(url, timeout=12.0)
    try:
        result = ((payload or {}).get("quoteSummary") or {}).get("result") or []
        if not result:
            return None
        th = (result[0].get("topHoldings") or {}).get("holdings") or []
        holdings: list[dict[str, Any]] = []
        for row in th[:top_n]:
            sym = (row.get("symbol") or "").strip().upper()
            if not sym:
                continue
            holdings.append(
                {
                    "symbol": sym,
                    "name": row.get("holdingName") or sym,
                    "weight_pct": _pct_display(
                        (row.get("holdingPercent") or {}).get("raw")
                        if isinstance(row.get("holdingPercent"), dict)
                        else row.get("holdingPercent")
                    ),
                }
            )
        if not holdings:
            return None
        return {
            "symbol": ticker,
            "holdings": holdings,
            "count": len(holdings),
            "top_n": top_n,
            "note": "Yahoo quoteSummary topHoldings；通常仅前十大，非实时全持仓。",
            "source": "yahoo_quotesummary",
            "source_url": f"https://finance.yahoo.com/quote/{ticker}/holdings",
        }
    except Exception:  # noqa: BLE001
        return None


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


def _finnhub_api_key() -> str:
    try:
        from research_agent.config import get_settings

        return (get_settings().finnhub_api_key or "").strip()
    except Exception:  # noqa: BLE001
        import os

        return (os.environ.get("FINNHUB_API_KEY") or "").strip()


def _quote_via_finnhub(symbol: str) -> dict[str, Any] | None:
    """Finnhub ``/quote``（需 ``FINNHUB_API_KEY``）。指数/期货/外汇符号跳过。"""
    key = _finnhub_api_key()
    if not key:
        return None
    ticker = _normalize_ticker(symbol)
    # Yahoo 风格指数/期货/外汇留给 Chart/东财；正股与 ETF（含 BRK.B）走 Finnhub
    if ticker.startswith("^") or ticker.endswith(("=F", "=X")):
        return None
    from urllib.parse import urlencode

    qs = urlencode({"symbol": ticker, "token": key})
    url = f"https://finnhub.io/api/v1/quote?{qs}"
    data = _http_get_json(url, timeout=8.0)
    if not isinstance(data, dict):
        return None
    price = _num_or_none(data.get("c"))
    if price is None or price <= 0:
        return None
    prev = _num_or_none(data.get("pc"))
    chg = _num_or_none(data.get("d"))
    chg_pct = _num_or_none(data.get("dp"))
    if chg is None and prev is not None and price is not None:
        chg = price - prev
    if chg_pct is None and prev not in (None, 0) and chg is not None:
        chg_pct = (chg / prev) * 100.0
    return {
        "price": price,
        "previous_close": prev,
        "change": chg,
        "change_percent": round(chg_pct, 4) if chg_pct is not None else None,
        "source": "finnhub",
    }


def _quote_from_ticker(symbol: str) -> dict[str, Any]:
    """报价：Yahoo Chart → Finnhub（若已配 Key）→ 东财美股 → yfinance。

    国内网络下 Yahoo 常返回 403 / ``Too Many Requests``；东财 ulist 作稳定回退。
    配置 ``FINNHUB_API_KEY`` 后自动启用 Finnhub 报价（与新闻第二源共用同一 Key）。
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

    fh = _quote_via_finnhub(ticker)
    if fh and fh.get("price") is not None:
        return _pack(
            fh,
            source_url="https://finnhub.io/docs/api/quote",
            note=f"{as_of_note}；来源 Finnhub quote",
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
        symbol: ticker，如 ``TSLA``、``QQQ``、``^IXIC``。
        period: yfinance period，如 ``5d`` / ``1mo`` / ``3mo`` / ``1y`` / ``5y``。
        interval: ``1d`` / ``1wk`` / ``1mo``（盘中分时可用 ``1h`` / ``5m``，注意延迟）。
    """
    allowed_periods = {"5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
    if period not in allowed_periods:
        return {
            "error": f"period 必须是 {sorted(allowed_periods)} 之一",
            "context": "get_price_history()",
        }

    # 快路径优先：Yahoo Chart → 可选 Finnhub → 东财（秒级、有 HTTP timeout）。
    # yfinance 放最后：国内常挂起，即使有 wait_for 也会占满线程池。
    try:
        chart = await asyncio.wait_for(
            asyncio.to_thread(_history_via_yahoo_chart, symbol, period=period, interval=interval),
            timeout=15.0,
        )
        if chart and (chart.get("bars") or []):
            return chart
    except TimeoutError:
        logger.warning("get_price_history yahoo_chart timed out (%s)", symbol)

    try:
        em = await asyncio.wait_for(
            asyncio.to_thread(_history_via_eastmoney, symbol, period=period, interval=interval),
            timeout=15.0,
        )
        if em and (em.get("bars") or []):
            return em
    except TimeoutError:
        logger.warning("get_price_history eastmoney timed out (%s)", symbol)

    def _call() -> dict[str, Any]:
        import yfinance as yf

        ticker = _normalize_ticker(symbol)
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        records = _history_records(df, limit=120)
        if not records:
            return {
                "error": "empty history from yfinance",
                "context": f"get_price_history({ticker!r})",
                "symbol": ticker,
            }
        return {
            "symbol": ticker,
            "period": period,
            "interval": interval,
            "bars": records,
            "summary": _history_summary(records),
            "source": "yfinance",
            "source_url": f"https://finance.yahoo.com/quote/{ticker.lstrip('^')}/history",
        }

    primary = await _yf_call(_call, context=f"get_price_history({symbol!r})", timeout=30.0)
    if "error" not in primary and (primary.get("bars") or []):
        return primary
    return (
        primary
        if isinstance(primary, dict)
        else {
            "error": "price history unavailable",
            "context": f"get_price_history({symbol!r})",
        }
    )


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
        if not holdings:
            return {
                "error": "empty top holdings",
                "context": f"get_etf_holdings({ticker!r})",
                "symbol": ticker,
            }
        return {
            "symbol": ticker,
            "holdings": holdings,
            "count": len(holdings),
            "top_n": n,
            "note": "Yahoo 通常仅披露前十大持仓；权重为可得快照，非实时全持仓。",
            "source": "yfinance.funds_data",
            "source_url": f"https://finance.yahoo.com/quote/{ticker}/holdings",
        }

    # 提问触发：yfinance → Yahoo quoteSummary；东财无稳定美股 ETF holdings 公开接口
    primary = await _yf_call(_call, context=f"get_etf_holdings({symbol!r})", timeout=60.0)
    if "error" not in primary and (primary.get("holdings") or []):
        return primary
    alt = await asyncio.to_thread(_holdings_via_yahoo_quotesummary, symbol, top_n=n)
    if alt and (alt.get("holdings") or []):
        return alt
    if isinstance(primary, dict):
        primary.setdefault(
            "note",
            "Yahoo 持仓不可用；东财无美股 ETF holdings 公开兜底，请稍后重试或换数据源。",
        )
        return primary
    return {
        "error": "etf holdings unavailable",
        "context": f"get_etf_holdings({symbol!r})",
        "symbol": _normalize_ticker(symbol),
    }


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


def _fund_overview_from_info(ticker: str, info: dict[str, Any]) -> dict[str, Any]:
    quote_type = str(info.get("quoteType") or "")
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
        "five_year_avg_return": _json_safe(info.get("fiveYearAverageReturn")),
        "expense_hint": _json_safe(info.get("annualReportExpenseRatio")),
        "yield": _json_safe(info.get("yield") or info.get("dividendYield")),
        "currency": info.get("currency") or "USD",
    }
    return {k: v for k, v in overview.items() if v is not None}


def _serialize_option_legs(df: Any, *, side: str, limit: int) -> list[dict[str, Any]]:
    if df is None:
        return []
    try:
        if getattr(df, "empty", True):
            return []
        limited = df.head(limit)
    except Exception:  # noqa: BLE001
        return []
    rows: list[dict[str, Any]] = []
    for _, row in limited.iterrows():
        try:
            item = {
                "side": side,
                "contract": _json_safe(row.get("contractSymbol")),
                "strike": _json_safe(row.get("strike")),
                "last": _json_safe(row.get("lastPrice")),
                "bid": _json_safe(row.get("bid")),
                "ask": _json_safe(row.get("ask")),
                "volume": _json_safe(row.get("volume")),
                "open_interest": _json_safe(row.get("openInterest")),
                "implied_volatility": _json_safe(row.get("impliedVolatility")),
                "in_the_money": _json_safe(row.get("inTheMoney")),
            }
            rows.append({k: v for k, v in item.items() if v is not None})
        except Exception:  # noqa: BLE001
            continue
    return rows


@mcp.tool()
@cached_tool(ttl=TTL_DAILY, namespace="us")
async def get_mutual_fund_overview(symbol: str) -> dict:
    """获取美国共同基金概况（NAV、类别、基金公司、费用率等可得字段）。

    Args:
        symbol: 共同基金 ticker，如 ``VTSAX``、``VFIAX``。
    """

    def _call() -> dict[str, Any]:
        import yfinance as yf

        ticker = _normalize_ticker(symbol)
        info = yf.Ticker(ticker).info or {}
        quote_type = str(info.get("quoteType") or "").upper()
        body = _fund_overview_from_info(ticker, info)
        out: dict[str, Any] = {
            "fund": body,
            "source": "yfinance",
            "source_url": f"https://finance.yahoo.com/quote/{ticker}",
        }
        if quote_type and quote_type != "MUTUALFUND":
            out["note"] = (
                f"quoteType={quote_type or 'unknown'}（期望 MUTUALFUND）；"
                "若为 ETF 请改用 get_etf_overview。"
            )
        return out

    return await _yf_call(_call, context=f"get_mutual_fund_overview({symbol!r})", timeout=60.0)


@mcp.tool()
@cached_tool(ttl=TTL_DAILY, namespace="us")
async def get_mutual_fund_holdings(symbol: str, top_n: int = 10) -> dict:
    """获取美国共同基金重仓（Yahoo top holdings；不可用时 quoteSummary 兜底）。

    Args:
        symbol: 共同基金 ticker，如 ``VTSAX``。
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
                "error": "funds_data unavailable (not a fund or yfinance too old)",
                "context": f"get_mutual_fund_holdings({ticker!r})",
                "symbol": ticker,
            }
        try:
            top = funds.top_holdings
        except Exception as exc:  # noqa: BLE001
            return _fmt_error(exc, context=f"get_mutual_fund_holdings({ticker!r}).top_holdings")

        holdings = _serialize_top_holdings(top, top_n=n)
        if not holdings:
            return {
                "error": "empty top holdings",
                "context": f"get_mutual_fund_holdings({ticker!r})",
                "symbol": ticker,
            }
        return {
            "symbol": ticker,
            "holdings": holdings,
            "count": len(holdings),
            "top_n": n,
            "note": "Yahoo 通常仅披露前十大持仓；权重为可得快照。",
            "source": "yfinance.funds_data",
            "source_url": f"https://finance.yahoo.com/quote/{ticker}/holdings",
        }

    primary = await _yf_call(_call, context=f"get_mutual_fund_holdings({symbol!r})", timeout=60.0)
    if "error" not in primary and (primary.get("holdings") or []):
        return primary
    alt = await asyncio.to_thread(_holdings_via_yahoo_quotesummary, symbol, top_n=n)
    if alt and (alt.get("holdings") or []):
        return alt
    if isinstance(primary, dict):
        return primary
    return {
        "error": "mutual fund holdings unavailable",
        "context": f"get_mutual_fund_holdings({symbol!r})",
        "symbol": _normalize_ticker(symbol),
    }


@mcp.tool()
@cached_tool(ttl=TTL_REALTIME, namespace="us")
async def get_futures_quotes(symbols: str = "") -> dict:
    """批量获取常用美股/商品期货报价（Yahoo ``=F`` 连续合约）。

    Args:
        symbols: 可选，逗号分隔期货代码（如 ``CL=F,GC=F``）。空则返回内置常用列表。
    """

    def _parse_symbols(raw: str) -> list[tuple[str, str]]:
        text = (raw or "").strip()
        if not text:
            return list(_DEFAULT_FUTURES)
        out: list[tuple[str, str]] = []
        name_map = {sym: name for sym, name in _DEFAULT_FUTURES}
        for part in text.split(","):
            sym = _normalize_ticker(part)
            if not sym:
                continue
            out.append((sym, name_map.get(sym, sym)))
        return out or list(_DEFAULT_FUTURES)

    wanted = _parse_symbols(symbols)

    def _call() -> dict[str, Any]:
        quotes: list[dict[str, Any]] = []
        sources: set[str] = set()
        for sym, name in wanted:
            try:
                q = _quote_from_ticker(sym)
                src = str(q.get("source") or "unknown")
                sources.add(src)
                quotes.append(
                    {
                        "name": q.get("name") or name,
                        "symbol": sym,
                        "price": q.get("price"),
                        "previous_close": q.get("previous_close"),
                        "change_percent": q.get("change_percent"),
                        "as_of_note": q.get("as_of_note"),
                        "source": src,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                quotes.append({"name": name, "symbol": sym, "error": str(exc)})
        ok = sum(1 for x in quotes if x.get("price") is not None)
        return {
            "futures": quotes,
            "count": len(quotes),
            "ok_count": ok,
            "source": "+".join(sorted(sources)) or "unknown",
            "source_url": "https://finance.yahoo.com/markets/commodities/",
            "note": "Yahoo 连续合约（=F），非交易所具体月份合约。",
        }

    return await _yf_call(_call, context="get_futures_quotes()", timeout=50.0)


@mcp.tool()
@cached_tool(ttl=TTL_DAILY, namespace="us")
async def get_option_expirations(symbol: str) -> dict:
    """返回美股标的的期权到期日列表（Yahoo）。

    Args:
        symbol: 正股 ticker，如 ``AAPL``、``SPY``。
    """

    def _call() -> dict[str, Any]:
        import yfinance as yf

        ticker = _normalize_ticker(symbol)
        try:
            dates = list(yf.Ticker(ticker).options or [])
        except Exception as exc:  # noqa: BLE001
            return _fmt_error(exc, context=f"get_option_expirations({ticker!r})")
        return {
            "symbol": ticker,
            "expirations": dates,
            "count": len(dates),
            "source": "yfinance",
            "source_url": f"https://finance.yahoo.com/quote/{ticker}/options",
        }

    return await _yf_call(_call, context=f"get_option_expirations({symbol!r})", timeout=45.0)


@mcp.tool()
@cached_tool(ttl=TTL_REALTIME, namespace="us")
async def get_option_chain(symbol: str, expiration: str = "", limit_per_side: int = 25) -> dict:
    """返回指定到期日的美股期权链摘要（calls / puts）。

    Args:
        symbol: 正股 ticker，如 ``AAPL``。
        expiration: 到期日 ``YYYY-MM-DD``；空则取最近一个到期日。
        limit_per_side: 每侧最多返回行数（默认 25，上限 80）。
    """
    n = max(1, min(int(limit_per_side), 80))

    def _call() -> dict[str, Any]:
        import yfinance as yf

        ticker = _normalize_ticker(symbol)
        t = yf.Ticker(ticker)
        try:
            dates = list(t.options or [])
        except Exception as exc:  # noqa: BLE001
            return _fmt_error(exc, context=f"get_option_chain({ticker!r}).options")
        if not dates:
            return {
                "error": "no option expirations",
                "context": f"get_option_chain({ticker!r})",
                "symbol": ticker,
            }
        exp = (expiration or "").strip() or dates[0]
        if exp not in dates:
            return {
                "error": f"expiration {exp!r} not in available dates",
                "context": f"get_option_chain({ticker!r})",
                "symbol": ticker,
                "expirations": dates[:12],
            }
        try:
            chain = t.option_chain(exp)
        except Exception as exc:  # noqa: BLE001
            return _fmt_error(exc, context=f"get_option_chain({ticker!r},{exp!r})")
        calls = _serialize_option_legs(getattr(chain, "calls", None), side="call", limit=n)
        puts = _serialize_option_legs(getattr(chain, "puts", None), side="put", limit=n)
        return {
            "symbol": ticker,
            "expiration": exp,
            "calls": calls,
            "puts": puts,
            "call_count": len(calls),
            "put_count": len(puts),
            "limit_per_side": n,
            "source": "yfinance",
            "source_url": f"https://finance.yahoo.com/quote/{ticker}/options?p={ticker}&date={exp}",
            "note": "摘要字段；非全市场扫描，不含自算 Greeks。",
        }

    return await _yf_call(
        _call, context=f"get_option_chain({symbol!r},{expiration!r})", timeout=60.0
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
