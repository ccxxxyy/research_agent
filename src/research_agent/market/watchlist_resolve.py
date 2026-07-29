"""看板自选：宽松搜码解析 + 批量行情。"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_CN_CODE = re.compile(r"^(?:(?:sh|sz|bj)\.? )?(\d{6})(?:\.(SH|SZ|BJ))?$", re.I)
_US_FUTURES = re.compile(r"^[A-Z]{1,3}=F$", re.I)

# 与 derivatives_server 目录对齐的常用期货
_CN_FUTURES_CATALOG: tuple[tuple[str, str, str], ...] = (
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
    ("J", "焦炭", "DCE"),
    ("JM", "焦煤", "DCE"),
    ("RU", "橡胶", "SHFE"),
    ("BU", "沥青", "SHFE"),
    ("FG", "玻璃", "CZCE"),
    ("SA", "纯碱", "CZCE"),
    ("LH", "生猪", "DCE"),
    ("JD", "鸡蛋", "DCE"),
    ("LC", "碳酸锂", "GFEX"),
    ("SI", "工业硅", "GFEX"),
)


def _num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _guess_cn_exchange(code: str) -> str:
    c = code.zfill(6)
    if c.startswith(("5", "6", "9")):
        return "SH"
    if c.startswith(("0", "1", "2", "3")):
        return "SZ"
    if c.startswith(("4", "8")):
        return "BJ"
    return "SH"


def _cn_asset_class(code: str) -> str:
    c = code.zfill(6)
    if c.startswith(("51", "56", "58", "15", "16", "18")):
        return "etf"
    return "equity"


_ASSET_CLASS_ZH = {
    "equity": "股票",
    "etf": "ETF",
    "future": "期货",
    "mutual_fund": "场外基金",
    "fund": "场外基金",
    "index": "指数",
    "forex": "外汇",
    "crypto": "加密货币",
    "unknown": "其他",
}


def _asset_class_zh(asset_class: str) -> str:
    return _ASSET_CLASS_ZH.get((asset_class or "").lower(), asset_class or "其他")


def _candidate(
    *,
    symbol: str,
    name: str,
    asset_class: str,
    market: str,
    exchange: str = "",
    note: str = "",
    industry: str = "",
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "name": name or symbol,
        "asset_class": asset_class,
        "asset_class_zh": _asset_class_zh(asset_class),
        "market": market,
        "exchange": exchange,
        "note": note,
        "industry": industry,
    }


_CN_NAME_CACHE = None
_CN_FUND_CACHE = None


def _ensure_cn_fund_cache():
    global _CN_FUND_CACHE
    if _CN_FUND_CACHE is not None:
        return _CN_FUND_CACHE
    try:
        import akshare as ak

        df = ak.fund_name_em()
        if df is not None and not getattr(df, "empty", True) and "基金代码" in df.columns:
            df = df.copy()
            df["基金代码"] = df["基金代码"].astype(str).str.zfill(6)
            _CN_FUND_CACHE = df
        else:
            _CN_FUND_CACHE = None
    except Exception as exc:  # noqa: BLE001
        logger.debug("fund_name_em cache failed: %s", exc)
        _CN_FUND_CACHE = None
    return _CN_FUND_CACHE


def _cn_fund_hit(code: str) -> dict[str, str] | None:
    df = _ensure_cn_fund_cache()
    if df is None:
        return None
    c = code.zfill(6)
    hits = df[df["基金代码"].astype(str) == c]
    if hits.empty:
        return None
    row = hits.iloc[0]
    name = str(row.get("基金简称") or row.get("基金名称") or c)
    ftype = str(row.get("基金类型") or "")
    # 场内 ETF/LOF 已可由股票/ETF 通道覆盖；其余标为场外基金
    ac = "etf" if ("ETF" in ftype.upper() or "LOF" in ftype.upper()) else "mutual_fund"
    return {"name": name, "asset_class": ac, "fund_type": ftype}


def _search_cn_funds_by_name(q: str, *, limit: int) -> list[dict[str, Any]]:
    df = _ensure_cn_fund_cache()
    if df is None or not q:
        return []
    names = df["基金简称"].astype(str) if "基金简称" in df.columns else df.iloc[:, 1].astype(str)
    codes = df["基金代码"].astype(str)
    mask = names.str.contains(q, na=False, regex=False)
    if q.isdigit():
        mask = mask | codes.str.contains(q, na=False, regex=False)
    hits = df[mask].head(limit)
    out: list[dict[str, Any]] = []
    for _, row in hits.iterrows():
        c = str(row.get("基金代码") or "").zfill(6)
        name = str(row.get("基金简称") or c)
        ftype = str(row.get("基金类型") or "")
        ac = "etf" if ("ETF" in ftype.upper() or "LOF" in ftype.upper()) else "mutual_fund"
        out.append(
            _candidate(
                symbol=c,
                name=name,
                asset_class=ac,
                market="CN_A",
                exchange="",
                note=(
                    "场内 ETF/LOF"
                    if ac == "etf"
                    else "场外开放式基金，按净值申赎（非交易所连续竞价）"
                ),
                industry=ftype,
            )
        )
    return out


def search_cn_watchlist(q: str, *, limit: int = 8) -> list[dict[str, Any]]:
    """宽松搜索 A 股股票/ETF/场外基金/期货候选。"""
    limit = max(1, min(int(limit), 8))
    raw = (q or "").strip()
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(item: dict[str, Any]) -> None:
        sym = item["symbol"]
        if sym in seen:
            return
        seen.add(sym)
        out.append(item)

    # 1) 6 位代码 / 带后缀 — 优先识别场外基金
    stripped = raw.replace(" ", "")
    m = _CN_CODE.match(stripped)
    code6 = None
    exch = ""
    if m:
        code6 = m.group(1)
        exch = (m.group(2) or _guess_cn_exchange(code6)).upper()
    elif re.fullmatch(r"\d{6}", stripped):
        code6 = stripped
        exch = _guess_cn_exchange(code6)

    if code6:
        fund = _cn_fund_hit(code6)
        if fund:
            _add(
                _candidate(
                    symbol=code6,
                    name=fund["name"],
                    asset_class=fund["asset_class"],
                    market="CN_A",
                    exchange=exch or "",
                )
            )
        else:
            name = _cn_name_for_code(code6) or code6
            _add(
                _candidate(
                    symbol=code6,
                    name=name,
                    asset_class=_cn_asset_class(code6),
                    market="CN_A",
                    exchange=exch or _guess_cn_exchange(code6),
                )
            )

    # 2) 期货代码 / 中文名
    up = raw.upper()
    for code, name, exchange in _CN_FUTURES_CATALOG:
        if up == code or up == f"{code}0" or raw in name or name in raw:
            _add(
                _candidate(
                    symbol=f"{code}0",
                    name=f"{name}连续",
                    asset_class="future",
                    market="CN_A",
                    exchange=exchange,
                )
            )
        if len(out) >= limit:
            return out[:limit]

    # 3) 中文名 / 模糊股票
    if any("\u4e00" <= ch <= "\u9fff" for ch in raw) or (not code6 and len(raw) >= 2):
        try:
            import akshare as ak

            global _CN_NAME_CACHE
            if _CN_NAME_CACHE is None:
                _CN_NAME_CACHE = ak.stock_info_a_code_name()
            df = _CN_NAME_CACHE
            if df is not None and not getattr(df, "empty", True):
                mask = df["name"].astype(str).str.contains(raw, na=False, regex=False)
                if raw.isdigit():
                    mask = mask | df["code"].astype(str).str.contains(raw, na=False, regex=False)
                hits = df[mask].head(limit)
                for _, row in hits.iterrows():
                    c = str(row["code"]).zfill(6)
                    _add(
                        _candidate(
                            symbol=c,
                            name=str(row["name"]),
                            asset_class=_cn_asset_class(c),
                            market="CN_A",
                            exchange=_guess_cn_exchange(c),
                        )
                    )
                    if len(out) >= limit:
                        break
        except Exception as exc:  # noqa: BLE001
            logger.debug("cn name search failed: %s", exc)

    # 4) 场外基金名称 / 代码模糊
    if len(out) < limit and (
        any("\u4e00" <= ch <= "\u9fff" for ch in raw)
        or (not code6 and len(raw) >= 2)
        or (code6 is None and raw.isdigit())
    ):
        try:
            for item in _search_cn_funds_by_name(raw, limit=limit):
                _add(item)
                if len(out) >= limit:
                    break
        except Exception as exc:  # noqa: BLE001
            logger.debug("cn fund search failed: %s", exc)

    return out[:limit]


def _cn_name_for_code(code: str) -> str | None:
    global _CN_NAME_CACHE
    try:
        import akshare as ak

        if _CN_NAME_CACHE is None:
            _CN_NAME_CACHE = ak.stock_info_a_code_name()
        df = _CN_NAME_CACHE
        if df is None or getattr(df, "empty", True):
            return None
        hits = df[df["code"].astype(str) == code.zfill(6)]
        if hits.empty:
            return None
        return str(hits.iloc[0]["name"])
    except Exception:  # noqa: BLE001
        return None


def search_us_watchlist(q: str, *, limit: int = 8) -> list[dict[str, Any]]:
    """宽松搜索美股股票/ETF/期货/共同基金候选。

    默认过滤外汇货币对（``=X``）与加密货币，避免搜 ``MU`` 时刷出 MUR/USD 等噪音；
    精确 ticker 命中优先排在最前。
    """
    limit = max(1, min(int(limit), 8))
    raw = (q or "").strip()
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(item: dict[str, Any], *, prefer: bool = False) -> None:
        sym = item["symbol"]
        if not sym or sym in seen:
            return
        seen.add(sym)
        if prefer:
            out.insert(0, item)
        else:
            out.append(item)

    def _us_asset_from_yf(sym: str, qtype: str) -> tuple[str, str]:
        """返回 (asset_class, note)。外汇/加密默认不加入自选，仅作分类说明。"""
        qt = (qtype or "").upper()
        su = sym.upper()
        if su.endswith("=X") or "CURRENCY" in qt or qt == "FX":
            return (
                "forex",
                "外汇货币对汇率（如 MUR/USD），不是上市公司股票，自选看板默认不收录",
            )
        if "CRYPTO" in qt or su.endswith("-USD") and len(su) <= 10 and "BTC" in su:
            return "crypto", "加密货币报价，不是美股股票/ETF"
        if "ETF" in qt:
            return "etf", "交易所交易基金（ETF），跟踪指数或一篮子资产"
        if "OPTION" in qt or qt == "OPT":
            return "option", "期权合约，非正股；自选默认不收录"
        if "MUTUAL" in qt or ("FUND" in qt and "ETF" not in qt):
            return "mutual_fund", "共同基金，按净值交易（非盘中连续竞价）"
        if "FUTURE" in qt or su.endswith("=F"):
            return "future", "期货合约连续报价"
        if "INDEX" in qt or su.startswith("^"):
            return "index", "市场指数，非可交易个股（部分可用对应 ETF 代替）"
        return "equity", "美股上市公司股票"

    up = raw.upper()

    # 期货 =F
    if _US_FUTURES.match(up):
        _add(
            _candidate(
                symbol=up,
                name=up,
                asset_class="future",
                market="US",
                exchange="CME",
                note="期货合约连续报价",
            ),
            prefer=True,
        )

    # 直接当 ticker 试报价（1–5 字母精确匹配优先）
    if re.fullmatch(r"[A-Za-z]{1,5}", raw) or raw.startswith("^") or _US_FUTURES.match(up):
        try:
            from research_agent.mcp_servers.us_data_server import _quote_from_ticker

            sym = up if _US_FUTURES.match(up) else (raw if raw.startswith("^") else up)
            qte = _quote_from_ticker(sym)
            if qte.get("price") is not None:
                qt = "equity"
                note = "美股上市公司股票"
                if sym.endswith("=F"):
                    qt, note = "future", "期货合约连续报价"
                elif sym.startswith("^"):
                    qt, note = "index", "市场指数"
                _add(
                    _candidate(
                        symbol=sym,
                        name=str(qte.get("name") or sym),
                        asset_class=qt,
                        market="US",
                        note=note,
                    ),
                    prefer=True,
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("us direct quote failed: %s", exc)

    # yfinance Search
    try:
        from yfinance import Search

        zh_map = {
            "苹果": "Apple",
            "特斯拉": "Tesla",
            "微软": "Microsoft",
            "英伟达": "NVIDIA",
            "亚马逊": "Amazon",
            "谷歌": "Alphabet",
        }
        search_q = zh_map.get(raw, zh_map.get(raw.lower(), raw))
        s = Search(search_q, max_results=max(limit * 3, 12))
        quotes = getattr(s, "quotes", None) or []
        for item in quotes:
            sym = str(item.get("symbol") or "").strip()
            if not sym:
                continue
            qtype = str(item.get("quoteType") or item.get("typeDisp") or "")
            ac, note = _us_asset_from_yf(sym, qtype)
            # 自选默认不收外汇/加密/期权，避免 MU → MUR/USD、以及一长串期权合约噪音
            if ac in ("forex", "crypto", "option"):
                continue
            # 期权符号形态（含 put/call 日期码）
            if re.search(r"\d{6}[CP]\d{8}$", sym.upper()):
                continue
            prefer = sym.upper() == up
            _add(
                _candidate(
                    symbol=sym,
                    name=str(item.get("shortname") or item.get("longname") or sym),
                    asset_class=ac,
                    market="US",
                    exchange=str(item.get("exchange") or item.get("exchDisp") or ""),
                    note=note,
                ),
                prefer=prefer,
            )
            if len(out) >= limit:
                break
    except Exception as exc:  # noqa: BLE001
        logger.debug("us Search failed: %s", exc)

    return out[:limit]


def search_watchlist(market: str, q: str, *, limit: int = 8) -> list[dict[str, Any]]:
    mkt = (market or "").strip().upper()
    if mkt == "CN_A":
        return search_cn_watchlist(q, limit=limit)
    if mkt == "US":
        return search_us_watchlist(q, limit=limit)
    return []


def _sina_cn_quotes(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """新浪 hq 批量 A 股/ETF 行情。symbols 为 6 位码。"""
    import requests

    if not symbols:
        return {}
    parts: list[str] = []
    for code in symbols:
        c = code.zfill(6)
        ex = _guess_cn_exchange(c).lower()
        prefix = {"sh": "sh", "sz": "sz", "bj": "bj"}.get(ex, "sh")
        parts.append(f"{prefix}{c}")
    url = "https://hq.sinajs.cn/list=" + ",".join(parts)
    headers = {"Referer": "https://finance.sina.com.cn", "User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        resp.encoding = "gbk"
        text = resp.text or ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("sina quotes failed: %s", exc)
        return {}

    out: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        # var hq_str_sh600519="名称,开盘,...";
        if '="' not in line:
            continue
        try:
            left, right = line.split('="', 1)
            key = left.split("_")[-1]  # sh600519
            code = key[2:] if len(key) > 2 else key
            payload = right.rstrip('";')
            fields = payload.split(",")
            if len(fields) < 4:
                continue
            name = fields[0]
            price = _num(fields[3])
            prev = _num(fields[2])
            chg = None
            if price is not None and prev not in (None, 0):
                chg = round((price - prev) / prev * 100, 2)
            out[code.zfill(6)] = {
                "symbol": code.zfill(6),
                "name": name,
                "price": round(price, 2) if price is not None else None,
                "change_pct": chg,
            }
        except Exception:  # noqa: BLE001
            continue
    return out


def _cn_future_quote(symbol: str) -> dict[str, Any] | None:
    """国内期货连续合约行情（RB0）。"""
    try:
        import akshare as ak

        sym = symbol.upper()
        if re.fullmatch(r"[A-Z]{1,2}", sym):
            sym = f"{sym}0"
        df = ak.futures_zh_spot(symbol=sym, market="CF", adjust="0")
        if df is None or getattr(df, "empty", True):
            return None
        row = df.iloc[0]
        price = _num(row.get("current_price"))
        settle = _num(row.get("last_settle_price")) or _num(row.get("last_close"))
        chg = None
        if price is not None and settle not in (None, 0):
            chg = round((price - settle) / settle * 100, 2)
        return {
            "symbol": symbol,
            "name": str(row.get("symbol") or symbol),
            "price": round(price, 2) if price is not None else None,
            "change_pct": chg,
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("cn future quote %s failed: %s", symbol, exc)
        return None


def _cn_industries_em(codes: list[str]) -> dict[str, str]:
    """东财 ulist 批量行业（f100）。"""

    bare = list(dict.fromkeys(c.zfill(6) for c in codes if re.fullmatch(r"\d{6}", c.zfill(6))))
    if not bare:
        return {}

    def _secid(code: str) -> str:
        if code.startswith(("5", "6", "9")) or code.startswith("688"):
            return f"1.{code}"
        return f"0.{code}"

    params = {
        "fltt": "2",
        "secids": ",".join(_secid(c) for c in bare),
        "fields": "f12,f14,f100",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
    }
    hosts = (
        "https://push2delay.eastmoney.com/api/qt/ulist.np/get",
        "https://push2.eastmoney.com/api/qt/ulist.np/get",
    )
    try:
        import requests

        sess = requests.Session()
        sess.trust_env = False
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
        for base in hosts:
            try:
                r = sess.get(base, params=params, timeout=8, headers=headers)
                diff = ((r.json() or {}).get("data") or {}).get("diff") or []
                out: dict[str, str] = {}
                for it in diff:
                    code = str(it.get("f12") or "").zfill(6)
                    ind = str(it.get("f100") or "").strip()
                    if code and ind and ind not in {"-", "—", "null"}:
                        out[code] = ind
                if out:
                    return out
            except Exception:  # noqa: BLE001
                continue
        sess.close()
    except Exception as exc:  # noqa: BLE001
        logger.debug("cn industry batch failed: %s", exc)
    return {}


def _cn_fund_nav_quote(symbol: str) -> dict[str, Any] | None:
    """场外基金最新单位净值（东财）。"""
    try:
        import akshare as ak

        code = symbol.zfill(6)
        df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
        if df is None or getattr(df, "empty", True):
            return None
        row = df.iloc[-1]
        # 列名可能是 净值日期 / 单位净值 / 日增长率
        price = None
        chg = None
        for col in row.index:
            cs = str(col)
            if "单位净值" in cs or cs in ("净值", "nav"):
                price = _num(row[col])
            if "日增长率" in cs or "涨跌" in cs:
                chg = _num(row[col])
        if price is None:
            # 常见列序：日期, 单位净值, 日增长率
            if len(row) >= 2:
                price = _num(row.iloc[1])
            if len(row) >= 3:
                chg = _num(row.iloc[2])
        if price is None:
            return None
        fund = _cn_fund_hit(code)
        return {
            "symbol": code,
            "name": (fund or {}).get("name") or code,
            "price": round(price, 4),
            "change_pct": round(chg, 2) if chg is not None else None,
            "price_kind": "nav",
            "industry": (fund or {}).get("fund_type") or "场外基金",
            "asset_class": "mutual_fund",
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("cn fund nav %s failed: %s", symbol, exc)
        return None


def fetch_watchlist_quotes(market: str, symbols: list[str]) -> list[dict[str, Any]]:
    """批量自选行情（含类型/行业/净值或收盘价）。"""
    mkt = (market or "").strip().upper()
    syms = [str(s).strip() for s in symbols if str(s).strip()]
    if not syms:
        return []

    if mkt == "US":
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FuturesTimeout

        from research_agent.mcp_servers.us_data_server import _quote_from_ticker

        def _one(sym: str) -> dict[str, Any] | None:
            try:
                q = _quote_from_ticker(sym)
                price = _num(q.get("price"))
                if price is None or price <= 0:
                    return None
                chg = _num(q.get("change_percent"))
                ac = "equity"
                if sym.endswith("=F"):
                    ac = "future"
                elif sym.startswith("^"):
                    ac = "index"
                elif sym.endswith("=X"):
                    ac = "forex"
                return {
                    "symbol": sym,
                    "name": str(q.get("name") or sym),
                    "price": round(price, 2),
                    "change_pct": round(chg, 2) if chg is not None else None,
                    "price_kind": "last",
                    "industry": "",
                    "asset_class": ac,
                    "asset_class_zh": _asset_class_zh(ac),
                }
            except Exception as exc:  # noqa: BLE001
                logger.debug("us watchlist quote %s: %s", sym, exc)
                return None

        rows: list[dict[str, Any]] = []
        # 逐只限时，避免脏 ticker 拖死整批；shutdown(wait=False) 不阻塞返回
        pool = ThreadPoolExecutor(max_workers=min(6, max(1, len(syms))))
        try:
            futs = {pool.submit(_one, sym): sym for sym in syms}
            for fut, sym in futs.items():
                try:
                    row = fut.result(timeout=8.0)
                except FuturesTimeout:
                    logger.warning("us watchlist quote timeout: %s", sym)
                    row = None
                if row:
                    rows.append(row)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return rows

    # CN_A：拆期货 / 股票ETF / 可能的场外基金
    equity_codes = [s.zfill(6) for s in syms if re.fullmatch(r"\d{6}", s)]
    future_syms = [s for s in syms if not re.fullmatch(r"\d{6}", s)]
    rows: list[dict[str, Any]] = []
    sina = _sina_cn_quotes(equity_codes)
    industries = _cn_industries_em(equity_codes) if equity_codes else {}
    for c in equity_codes:
        if c in sina:
            row = dict(sina[c])
            fund = _cn_fund_hit(c)
            # 新浪无有效价或明确是场外基金 → 走净值
            if (
                fund
                and fund.get("asset_class") == "mutual_fund"
                and (row.get("price") in (None, 0) or not sina.get(c))
            ):
                nav = _cn_fund_nav_quote(c)
                if nav:
                    rows.append(nav)
                    continue
            ac = _cn_asset_class(c)
            if fund and fund.get("asset_class") == "mutual_fund":
                # 场外基金若新浪误返回价，仍优先净值
                nav = _cn_fund_nav_quote(c)
                if nav:
                    rows.append(nav)
                    continue
                ac = "mutual_fund"
            row["price_kind"] = "nav" if ac == "mutual_fund" else "last"
            row["industry"] = industries.get(c) or (fund.get("fund_type") if fund else "")
            row["asset_class"] = ac
            row["asset_class_zh"] = _asset_class_zh(ac)
            rows.append(row)
        else:
            fund = _cn_fund_hit(c)
            if fund:
                nav = _cn_fund_nav_quote(c)
                if nav:
                    rows.append(nav)
                    continue
            # 无行情也返回元数据占位
            rows.append(
                {
                    "symbol": c,
                    "name": (fund or {}).get("name") or _cn_name_for_code(c) or c,
                    "price": None,
                    "change_pct": None,
                    "price_kind": "nav" if fund else "last",
                    "industry": (fund or {}).get("fund_type") or industries.get(c, ""),
                    "asset_class": (fund or {}).get("asset_class") or _cn_asset_class(c),
                    "asset_class_zh": _asset_class_zh(
                        (fund or {}).get("asset_class") or _cn_asset_class(c)
                    ),
                }
            )
    for fs in future_syms:
        q = _cn_future_quote(fs)
        if q:
            q = dict(q)
            q["price_kind"] = "last"
            q["industry"] = "国内期货"
            q["asset_class"] = "future"
            q["asset_class_zh"] = "期货"
            rows.append(q)
    return rows


__all__ = [
    "fetch_watchlist_quotes",
    "search_cn_watchlist",
    "search_us_watchlist",
    "search_watchlist",
]
