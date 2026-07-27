"""美股看板主题面板（由已有榜单聚合，不额外打外网）。

对应 A 股：主线题材 / 盘中异动 / 情绪标杆 / 妖股·庄股近似。
美股无涨停制度，以下均为规则近似，供留意比对。
"""

from __future__ import annotations

from typing import Any

# ETF → 常见龙头成分（用于「主线」命中计数；非完整持仓）
_US_THEME_MEMBERS: dict[str, tuple[str, ...]] = {
    "XLK": ("AAPL", "MSFT", "NVDA", "AVGO", "CRM", "ORCL", "AMD"),
    "XLF": ("JPM", "BAC", "WFC", "GS", "MS", "C"),
    "XLE": ("XOM", "CVX", "COP", "SLB", "EOG"),
    "XLV": ("UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE"),
    "XLI": ("CAT", "GE", "HON", "UPS", "RTX"),
    "XLY": ("AMZN", "TSLA", "HD", "MCD", "NKE"),
    "XLP": ("PG", "KO", "PEP", "WMT", "COST"),
    "XLU": ("NEE", "DUK", "SO", "D"),
    "XLB": ("LIN", "APD", "SHW", "ECL"),
    "XLRE": ("PLD", "AMT", "EQIX", "SPG"),
    "XLC": ("META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA"),
    "QQQ": ("AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "NFLX"),
    "SPY": ("AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"),
    "IWM": ("IWM",),
    "SMH": ("NVDA", "TSM", "AVGO", "AMD", "ASML", "MU", "INTC", "QCOM"),
    "SOXX": ("NVDA", "AVGO", "AMD", "MU", "INTC", "QCOM", "TXN"),
    "BOTZ": ("NVDA", "ISRG", "ABB", "PATH"),
    "ARKK": ("TSLA", "COIN", "ROKU", "HOOD", "PATH"),
    "XBI": ("VRTX", "REGN", "AMGN", "GILD", "BIIB"),
    "IBIT": ("COIN", "MSTR", "MARA", "RIOT"),
    "GLD": ("GLD",),
    "TLT": ("TLT",),
    "HYG": ("HYG",),
}


def _f(val: Any) -> float | None:
    try:
        if val is None:
            return None
        x = float(val)
        if x != x:
            return None
        return x
    except (TypeError, ValueError):
        return None


def _sym(item: dict) -> str:
    return str(item.get("symbol") or item.get("code") or "").strip().upper()


def build_us_mainline_themes(
    sectors: list[dict],
    theme_etfs: list[dict],
    gainers: list[dict],
    mega: list[dict],
    growth: list[dict],
) -> list[dict]:
    """主线题材：行业/主题 ETF 涨幅 + 关联龙头在涨幅/七巨头中的命中。"""
    hot: dict[str, dict] = {}
    for lst in (gainers, mega, growth):
        for it in lst or []:
            s = _sym(it)
            chg = _f(it.get("change_pct")) or 0.0
            if not s or chg <= 0:
                continue
            prev = hot.get(s)
            if prev is None or chg > (_f(prev.get("change_pct")) or 0):
                hot[s] = it

    boards = list(sectors or []) + list(theme_etfs or [])
    themes: list[dict] = []
    for b in boards:
        sym = _sym(b)
        name = str(b.get("name") or sym)
        change = _f(b.get("change_pct")) or 0.0
        members = _US_THEME_MEMBERS.get(sym, ())
        hits = [hot[m] for m in members if m in hot]
        hit_n = len(hits)
        leaders = [str(h.get("name") or _sym(h)) for h in hits[:2]]
        score = change + hit_n * 1.2
        themes.append(
            {
                "code": sym,
                "symbol": sym,
                "name": name,
                "change_pct": change,
                "price": b.get("price"),
                "hit_count": hit_n,
                "score": round(score, 2),
                "leaders": leaders,
                "kind": "sector" if sym.startswith("XL") else "theme",
            }
        )
    themes.sort(key=lambda x: (-(x.get("score") or 0), -(x.get("change_pct") or 0)))
    # 去重名（偶发重复）
    seen: set[str] = set()
    out: list[dict] = []
    for t in themes:
        if t["symbol"] in seen:
            continue
        seen.add(t["symbol"])
        out.append(t)
        if len(out) >= 10:
            break
    return out


def build_us_intraday_moves(
    gainers: list[dict], losers: list[dict], *, min_abs_pct: float = 3.0
) -> list[dict]:
    """日内异动：暴涨 / 暴跌（按 |涨跌幅|）。"""
    items: list[dict] = []
    for it in gainers or []:
        chg = _f(it.get("change_pct"))
        if chg is None or chg < min_abs_pct:
            continue
        items.append(
            {
                **{
                    k: it.get(k)
                    for k in ("code", "symbol", "name", "price", "change_pct", "volume")
                },
                "label": "暴涨",
                "abs_pct": abs(chg),
            }
        )
    for it in losers or []:
        chg = _f(it.get("change_pct"))
        if chg is None or chg > -min_abs_pct:
            continue
        items.append(
            {
                **{
                    k: it.get(k)
                    for k in ("code", "symbol", "name", "price", "change_pct", "volume")
                },
                "label": "暴跌",
                "abs_pct": abs(chg),
            }
        )
    items.sort(key=lambda x: -(x.get("abs_pct") or 0))
    seen: set[str] = set()
    out: list[dict] = []
    for it in items:
        s = _sym(it)
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(it)
        if len(out) >= 10:
            break
    return out


def build_us_sentiment(
    actives: list[dict],
    gainers: list[dict],
    mega: list[dict],
) -> list[dict]:
    """情绪标杆：高换手（活跃榜）+ 强势涨幅；有 52 周高附近则标「近新高」。"""
    out: list[dict] = []
    seen: set[str] = set()

    def _add(it: dict, tag: str) -> None:
        s = _sym(it)
        if not s or s in seen:
            return
        seen.add(s)
        price = _f(it.get("price"))
        high52 = _f(it.get("fifty_two_week_high") or it.get("fiftyTwoWeekHigh"))
        near_high = False
        if price and high52 and high52 > 0 and price >= high52 * 0.98:
            near_high = True
            tag = "近新高" if tag == "强势" else f"{tag}·近新高"
        out.append(
            {
                "code": it.get("code") or s,
                "symbol": s,
                "name": it.get("name") or s,
                "price": it.get("price"),
                "change_pct": it.get("change_pct"),
                "volume": it.get("volume"),
                "tag": tag,
                "near_high": near_high,
            }
        )

    # 活跃榜按成交量优先 → 高换手情绪
    act_sorted = sorted(
        list(actives or []),
        key=lambda x: _f(x.get("volume")) or 0,
        reverse=True,
    )
    for it in act_sorted[:6]:
        chg = _f(it.get("change_pct")) or 0.0
        _add(it, "高换手·活跃" if chg >= 0 else "高换手·承压")

    for it in (gainers or [])[:8]:
        chg = _f(it.get("change_pct")) or 0.0
        if chg >= 5.0:
            _add(it, "强势")

    for it in mega or []:
        chg = _f(it.get("change_pct")) or 0.0
        if chg >= 2.0:
            _add(it, "权重情绪")

    # 近新高优先展示
    out.sort(
        key=lambda x: (
            0 if x.get("near_high") else 1,
            -(abs(_f(x.get("change_pct")) or 0)),
        )
    )
    return out[:10]


def build_us_speculative(
    shorted: list[dict],
    small_gainers: list[dict],
    gainers: list[dict],
) -> list[dict]:
    """投机/拥挤交易近似：高空头关注 + 小盘暴涨（非官方妖股标签）。"""
    out: list[dict] = []
    seen: set[str] = set()

    def _add(it: dict, label: str, reason: str) -> None:
        s = _sym(it)
        if not s or s in seen:
            return
        seen.add(s)
        out.append(
            {
                "code": it.get("code") or s,
                "symbol": s,
                "name": it.get("name") or s,
                "price": it.get("price"),
                "change_pct": it.get("change_pct"),
                "label": label,
                "reason": reason,
            }
        )

    for it in shorted or []:
        chg = _f(it.get("change_pct"))
        chg_s = f"{chg:+.1f}%" if chg is not None else ""
        _add(it, "拥挤空头", f"高空头关注 {chg_s}".strip())

    for it in small_gainers or []:
        chg = _f(it.get("change_pct")) or 0.0
        if chg >= 8.0:
            _add(it, "小盘投机", f"小盘涨幅 {chg:+.1f}%")

    for it in gainers or []:
        chg = _f(it.get("change_pct")) or 0.0
        if chg >= 15.0:
            _add(it, "极端波动", f"日内 {chg:+.1f}%")

    return out[:10]
