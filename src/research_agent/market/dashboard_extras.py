"""首页看板扩展：国内期货/ETF/QDII + 美股期货/ETF/共同基金动态双榜。

统一返回::

    {"by_volume": [...], "by_change": [...], "limit": N, "source": "..."}

宇宙可为候选池；价格/成交量实时拉取后本地排序。失败返回空双榜，不拖垮整页。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 美股期货候选池（Yahoo 连续合约）；看板按成交量/涨跌幅动态取 Top N
_US_FUTURES_UNIVERSE: tuple[tuple[str, str], ...] = (
    ("CL=F", "WTI原油"),
    ("BZ=F", "布伦特"),
    ("NG=F", "天然气"),
    ("HO=F", "取暖油"),
    ("RB=F", "RBOB汽油"),
    ("GC=F", "黄金"),
    ("SI=F", "白银"),
    ("HG=F", "铜"),
    ("PL=F", "铂金"),
    ("PA=F", "钯金"),
    ("ES=F", "标普期货"),
    ("NQ=F", "纳指期货"),
    ("YM=F", "道指期货"),
    ("RTY=F", "罗素2000期货"),
    ("MES=F", "微型标普"),
    ("MNQ=F", "微型纳指"),
    ("ZB=F", "美债30年"),
    ("ZN=F", "美债10年"),
    ("ZF=F", "美债5年"),
    ("ZT=F", "美债2年"),
    ("ZC=F", "玉米"),
    ("ZS=F", "大豆"),
    ("ZW=F", "小麦"),
    ("ZM=F", "豆粕"),
    ("ZL=F", "豆油"),
    ("LE=F", "活牛"),
    ("HE=F", "瘦肉猪"),
    ("GF=F", "饲牛"),
    ("KC=F", "咖啡"),
    ("CT=F", "棉花"),
    ("SB=F", "糖"),
    ("CC=F", "可可"),
    ("OJ=F", "橙汁"),
    ("LBS=F", "木材"),
    ("BTC=F", "比特币期货"),
)

# 美股高流动性 ETF 候选池 → 动态双榜
_US_ETF_UNIVERSE: tuple[tuple[str, str], ...] = (
    ("SPY", "标普500ETF"),
    ("QQQ", "纳指100ETF"),
    ("IWM", "罗素2000ETF"),
    ("DIA", "道指ETF"),
    ("VOO", "Vanguard标普"),
    ("VTI", "全市场ETF"),
    ("EFA", "发达市场"),
    ("EEM", "新兴市场"),
    ("XLK", "科技扇区"),
    ("XLF", "金融扇区"),
    ("XLE", "能源扇区"),
    ("XLV", "医疗扇区"),
    ("XLI", "工业扇区"),
    ("XLY", "可选消费"),
    ("XLP", "必选消费"),
    ("XLU", "公用事业"),
    ("XLB", "原材料"),
    ("XLRE", "房地产"),
    ("SMH", "半导体"),
    ("XBI", "生物科技"),
    ("ARKK", "ARK创新"),
    ("GLD", "黄金ETF"),
    ("SLV", "白银ETF"),
    ("USO", "原油ETF"),
    ("TLT", "长期国债"),
    ("HYG", "高收益债"),
    ("LQD", "投资级债"),
    ("UNG", "天然气ETF"),
    ("SOXX", "半导体SOXX"),
    ("BOTZ", "机器人AI"),
)

# 美股共同基金候选池 → 按 YTD 动态排序（无可靠日成交量）
_US_MUTUAL_FUNDS_UNIVERSE: tuple[tuple[str, str], ...] = (
    ("VTSAX", "Vanguard全市场"),
    ("VFIAX", "Vanguard标普500"),
    ("FXAIX", "Fidelity标普500"),
    ("SWTSX", "Schwab全市场"),
    ("VWELX", "Vanguard惠灵顿"),
    ("VBTLX", "Vanguard全债"),
    ("VTIAX", "Vanguard国际"),
    ("VIMAX", "Vanguard中盘"),
    ("VSMAX", "Vanguard小盘"),
    ("VGSLX", "Vanguard地产"),
    ("VWILX", "Vanguard国际成长"),
    ("FCNTX", "Fidelity Contra"),
    ("FBGRX", "Fidelity蓝筹成长"),
    ("PRGFX", "T.Rowe成长"),
    ("AGTHX", "美国基金成长"),
    ("AMECX", "美国基金收入"),
    ("DODGX", "Dodge Cox股票"),
    ("TRBCX", "T.Rowe蓝筹"),
    ("RWMFX", "American Washington"),
    ("NWJFX", "Nationwide Janus"),
)


def _num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _empty_rank(*, limit: int, source: str) -> dict[str, Any]:
    return {"by_volume": [], "by_change": [], "limit": limit, "source": source}


def _rank_dual(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    source: str,
) -> dict[str, Any]:
    """从候选行生成成交量榜 + 涨跌幅榜。"""
    limit = max(1, min(int(limit), 40))
    with_vol = [r for r in rows if _num(r.get("volume")) is not None]
    by_volume = sorted(
        with_vol,
        key=lambda r: _num(r.get("volume")) or 0.0,
        reverse=True,
    )[:limit]
    with_chg = [r for r in rows if _num(r.get("change_pct")) is not None]
    by_change = sorted(
        with_chg,
        key=lambda r: _num(r.get("change_pct")) or 0.0,
        reverse=True,
    )[:limit]
    for i, r in enumerate(by_volume, 1):
        r["rank_by"] = "volume"
        r["rank"] = i
    for i, r in enumerate(by_change, 1):
        r["rank_by"] = "change"
        r["rank"] = i
    return {
        "by_volume": by_volume,
        "by_change": by_change,
        "limit": limit,
        "source": source,
    }


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def fetch_cn_futures_panel(*, limit: int = 10) -> dict[str, Any]:
    """国内主力连续合约：批量新浪报价后按成交量/涨跌幅取 Top N。"""
    limit = max(1, min(int(limit), 20))
    empty = _empty_rank(limit=limit, source="sina")
    try:
        import akshare as ak
    except Exception as exc:  # noqa: BLE001
        logger.warning("cn futures panel: akshare unavailable: %s", exc)
        return empty

    try:
        catalog = ak.futures_display_main_sina()
    except Exception as exc:  # noqa: BLE001
        logger.warning("cn futures catalog failed: %s", exc)
        return empty
    if catalog is None or getattr(catalog, "empty", True):
        return empty

    symbols = [str(s).strip() for s in catalog["symbol"].tolist() if str(s).strip()]
    name_by_sym = {
        str(r["symbol"]).strip(): str(r.get("name") or r["symbol"]) for _, r in catalog.iterrows()
    }
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in _chunked(symbols, 18):
        try:
            df = ak.futures_zh_spot(symbol=",".join(chunk), market="CF", adjust="0")
        except Exception as exc:  # noqa: BLE001
            logger.debug("cn futures chunk failed (%s): %s", chunk[:3], exc)
            continue
        if df is None or getattr(df, "empty", True):
            continue
        # 返回行顺序与请求 symbol 列表对齐（symbol 列为中文名）
        for i, (_, row) in enumerate(df.iterrows()):
            code = chunk[i] if i < len(chunk) else str(row.get("symbol") or "")
            if code in seen:
                continue
            seen.add(code)
            display = str(row.get("symbol") or name_by_sym.get(code, code))
            price = _num(row.get("current_price"))
            settle = _num(row.get("last_settle_price")) or _num(row.get("last_close"))
            chg = None
            if price is not None and settle not in (None, 0):
                chg = (price - settle) / settle * 100.0
            vol = _num(row.get("volume"))
            variety = code[:-1] if code.endswith("0") and code[:-1].isalpha() else code
            rows.append(
                {
                    "code": variety,
                    "symbol": code,
                    "name": display,
                    "price": round(price, 2) if price is not None else None,
                    "change_pct": round(chg, 2) if chg is not None else None,
                    "volume": int(vol) if vol is not None else None,
                    "industry": "期货",
                }
            )
    if not rows:
        return empty
    return _rank_dual(rows, limit=limit, source="sina")


def fetch_cn_etf_panel(*, limit: int = 10) -> dict[str, Any]:
    """A 股场内 ETF：成交额榜 + 涨跌幅榜。"""
    limit = max(1, min(int(limit), 20))
    empty = _empty_rank(limit=limit, source="sina")
    try:
        from research_agent.mcp_servers import fund_server as fs

        df = fs._fetch_sina_etf_realtime()
        if df is not None and not getattr(df, "empty", True):
            rows: list[dict[str, Any]] = []
            for _, row in df.iterrows():
                code = str(row.get("代码") or "")
                name = str(row.get("名称") or code)
                vol = _num(row.get("成交额"))
                if vol is None:
                    vol = _num(row.get("成交量"))
                rows.append(
                    {
                        "code": code,
                        "name": name,
                        "price": _num(row.get("最新价")),
                        "change_pct": _num(row.get("涨跌幅")),
                        "volume": vol,
                        "industry": "ETF",
                    }
                )
            if rows:
                return _rank_dual(rows, limit=limit, source="sina")
    except Exception as exc:  # noqa: BLE001
        logger.debug("cn etf sina panel failed: %s", exc)

    try:
        import akshare as ak

        df = ak.fund_open_fund_rank_em(symbol="指数型")
        if df is None or getattr(df, "empty", True):
            return empty
        sort_col = "今年来" if "今年来" in df.columns else None
        rows = []
        for _, row in df.iterrows():
            code = str(row.get("基金代码") or "")
            name = str(row.get("基金简称") or code)
            rows.append(
                {
                    "code": code,
                    "name": name,
                    "price": _num(row.get("单位净值")),
                    "change_pct": _num(row.get(sort_col)) if sort_col else None,
                    "volume": None,
                    "industry": "指数型",
                }
            )
        ranked = _rank_dual(rows, limit=limit, source="eastmoney")
        # 降级无成交量：把 by_change 也填进展示用
        return ranked
    except Exception as exc:  # noqa: BLE001
        logger.warning("cn etf panel failed: %s", exc)
        return empty


def fetch_cn_qdii_panel(*, limit: int = 8) -> dict[str, Any]:
    """QDII：按近一年收益作涨跌榜（无可靠成交量，by_volume 为空）。"""
    limit = max(1, min(int(limit), 15))
    empty = _empty_rank(limit=limit, source="eastmoney")
    try:
        import akshare as ak

        df = ak.fund_open_fund_rank_em(symbol="QDII")
        if df is None or getattr(df, "empty", True):
            return empty
        sort_col = (
            "近1年" if "近1年" in df.columns else ("今年来" if "今年来" in df.columns else None)
        )
        rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            code = str(row.get("基金代码") or "")
            name = str(row.get("基金简称") or code)
            chg = _num(row.get(sort_col)) if sort_col else None
            rows.append(
                {
                    "code": code,
                    "name": name,
                    "price": _num(row.get("单位净值")),
                    "change_pct": chg,
                    "volume": None,
                    "industry": "QDII",
                }
            )
        return _rank_dual(rows, limit=limit, source="eastmoney")
    except Exception as exc:  # noqa: BLE001
        logger.warning("cn qdii panel failed: %s", exc)
        return empty


def _us_pool_quotes(
    universe: tuple[tuple[str, str], ...],
    *,
    industry: str,
) -> list[dict[str, Any]]:
    """对候选池并发拉 Yahoo/东财报价，并尽量补成交量。"""
    try:
        from research_agent.mcp_servers.us_data_server import _quote_from_ticker
    except Exception as exc:  # noqa: BLE001
        logger.warning("us quote import failed: %s", exc)
        return []

    # 批量拉近 5 日成交量
    vol_map: dict[str, float] = {}
    symbols = [s for s, _ in universe]
    try:
        import yfinance as yf

        hist = yf.download(
            symbols,
            period="5d",
            group_by="ticker",
            threads=True,
            progress=False,
            auto_adjust=False,
        )
        if hist is not None and not getattr(hist, "empty", True):
            nlevels = getattr(hist.columns, "nlevels", 1)
            if nlevels > 1:
                for sym in symbols:
                    try:
                        ser = hist[sym]["Volume"].dropna()
                        if len(ser):
                            vol_map[sym] = float(ser.iloc[-1])
                    except Exception:  # noqa: BLE001
                        continue
            elif "Volume" in getattr(hist, "columns", []):
                ser = hist["Volume"].dropna()
                if len(ser) and len(symbols) == 1:
                    vol_map[symbols[0]] = float(ser.iloc[-1])
    except Exception as exc:  # noqa: BLE001
        logger.debug("us volume batch failed: %s", exc)

    name_by = dict(universe)

    def _one(sym: str) -> dict[str, Any] | None:
        try:
            q = _quote_from_ticker(sym)
            price = _num(q.get("price"))
            if price is None:
                return None
            chg = _num(q.get("change_percent"))
            vol = vol_map.get(sym)
            return {
                "symbol": sym,
                "code": sym,
                "name": name_by.get(sym, sym),
                "price": round(price, 2),
                "change_pct": round(chg, 2) if chg is not None else None,
                "volume": int(vol) if vol is not None else None,
                "industry": industry,
                "price_source": str(q.get("source") or "yahoo"),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("us quote %s failed: %s", sym, exc)
            return None

    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(12, max(4, len(symbols)))) as pool:
        futs = [pool.submit(_one, sym) for sym in symbols]
        for fut in as_completed(futs):
            row = fut.result()
            if row:
                rows.append(row)
    return rows


def fetch_us_futures_panel(*, limit: int = 10) -> dict[str, Any]:
    """美股/商品期货：扩大 Yahoo 池报价后按成交量/涨跌幅排。"""
    limit = max(1, min(int(limit), 20))
    rows = _us_pool_quotes(_US_FUTURES_UNIVERSE, industry="期货")
    return _rank_dual(rows, limit=limit, source="yahoo")


def fetch_us_etf_rank_panel(*, limit: int = 10) -> dict[str, Any]:
    """美股 ETF 候选池动态双榜。"""
    limit = max(1, min(int(limit), 20))
    rows = _us_pool_quotes(_US_ETF_UNIVERSE, industry="ETF")
    return _rank_dual(rows, limit=limit, source="yahoo")


def fetch_us_mutual_funds_panel(*, limit: int = 10) -> dict[str, Any]:
    """美国共同基金扩大池：按 YTD 排涨跌榜（无日成交量）。"""
    limit = max(1, min(int(limit), 20))
    empty = _empty_rank(limit=limit, source="yfinance")
    try:
        import yfinance as yf
    except Exception as exc:  # noqa: BLE001
        logger.warning("us mutual funds: yfinance unavailable: %s", exc)
        return empty

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(sym: str, name: str) -> dict[str, Any] | None:
        try:
            info = yf.Ticker(sym).info or {}
            nav = _num(
                info.get("navPrice") or info.get("regularMarketPrice") or info.get("previousClose")
            )
            ytd = _num(info.get("ytdReturn"))
            if ytd is not None and abs(ytd) <= 1.5:
                ytd = ytd * 100.0
            return {
                "symbol": sym,
                "code": sym,
                "name": str(info.get("shortName") or name),
                "price": round(nav, 2) if nav is not None else None,
                "change_pct": round(ytd, 2) if ytd is not None else None,
                "volume": None,
                "industry": "共同基金",
                "price_source": "yfinance",
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("us mutual fund %s failed: %s", sym, exc)
            return None

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_one, sym, name) for sym, name in _US_MUTUAL_FUNDS_UNIVERSE]
        for fut in as_completed(futs):
            row = fut.result()
            if row:
                rows.append(row)
    return _rank_dual(rows, limit=limit, source="yfinance")


__all__ = [
    "fetch_cn_etf_panel",
    "fetch_cn_futures_panel",
    "fetch_cn_qdii_panel",
    "fetch_us_etf_rank_panel",
    "fetch_us_futures_panel",
    "fetch_us_mutual_funds_panel",
]
