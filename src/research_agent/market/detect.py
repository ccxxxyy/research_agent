"""从用户问句 + 偏好解析市场。

优先级（高 → 低）
----------------
1. 请求显式覆盖（API ``market`` 字段）
2. 问句内硬信号（6 位 A 股代码 / 美股 ticker / 市场关键词 / 知名中英文名）
3. 同会话粘性（API ``thread_market`` / ``sticky_market``；跟进句无信号时沿用上一轮）
4. 用户 memory 中的 ``preferred_market``
5. 产品默认 ``CN_A``（与现有 A 股工具全集对齐；无信号时不默认美股）

名字判断
--------
用户通常会说公司/基金中英文名。本模块维护一份**可扩展**的知名映射表
（苹果→AAPL、特斯拉→TSLA、标普500→^GSPC、沪深300 等），用于 PoC；
完整解析可辅以 ``us_search_ticker`` / ``fin_search_stock_by_name``。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from research_agent.market.types import (
    PREFERRED_MARKET_KEY,
    PRODUCT_DEFAULT_MARKET,
    AssetClass,
    Market,
    MarketResolution,
    SymbolRef,
)

if TYPE_CHECKING:
    from research_agent.memory.manager import MemoryManager

# ---------------------------------------------------------------------
# 正则与关键词
# ---------------------------------------------------------------------
_A_SHARE_CODE = re.compile(r"(?<!\d)([0-9]{6})(?!\d)")
_US_TICKER_CANDIDATE = re.compile(r"(?<![A-Za-z])([A-Z]{1,5})(?![A-Za-z])")

_CN_MARKET_KEYWORDS = (
    "a股",
    "ａ股",
    "沪深",
    "上证",
    "深证",
    "创业板",
    "科创板",
    "北向",
    "南向",
    "龙虎榜",
    "涨停",
    "跌停",
    "融资融券",
    "两融",
    "巨潮",
    "沪市",
    "深市",
)

_US_MARKET_KEYWORDS = (
    "美股",
    "纳斯达克",
    "nasdaq",
    "纽交所",
    "nyse",
    "标普",
    "s&p",
    "道琼斯",
    "dow jones",
    "美股市场",
    "盘前",
    "盘后",
    "pre-market",
    "after-hours",
    "华尔街",
)

_MIXED_KEYWORDS = (
    "中美对比",
    "中美比较",
    "ah对比",
    "a/h",
    "adr对应",
    "跨市场",
)

# 知名标的：中文名 / 别名 → (ticker, asset_class, display_name)
_KNOWN_US: dict[str, tuple[str, AssetClass, str]] = {
    "苹果": ("AAPL", AssetClass.EQUITY, "Apple"),
    "apple": ("AAPL", AssetClass.EQUITY, "Apple"),
    "aapl": ("AAPL", AssetClass.EQUITY, "Apple"),
    "特斯拉": ("TSLA", AssetClass.EQUITY, "Tesla"),
    "tesla": ("TSLA", AssetClass.EQUITY, "Tesla"),
    "tsla": ("TSLA", AssetClass.EQUITY, "Tesla"),
    "微软": ("MSFT", AssetClass.EQUITY, "Microsoft"),
    "microsoft": ("MSFT", AssetClass.EQUITY, "Microsoft"),
    "msft": ("MSFT", AssetClass.EQUITY, "Microsoft"),
    "英伟达": ("NVDA", AssetClass.EQUITY, "NVIDIA"),
    "nvidia": ("NVDA", AssetClass.EQUITY, "NVIDIA"),
    "nvda": ("NVDA", AssetClass.EQUITY, "NVIDIA"),
    "亚马逊": ("AMZN", AssetClass.EQUITY, "Amazon"),
    "amazon": ("AMZN", AssetClass.EQUITY, "Amazon"),
    "amzn": ("AMZN", AssetClass.EQUITY, "Amazon"),
    "谷歌": ("GOOGL", AssetClass.EQUITY, "Alphabet"),
    "google": ("GOOGL", AssetClass.EQUITY, "Alphabet"),
    "googl": ("GOOGL", AssetClass.EQUITY, "Alphabet"),
    "goog": ("GOOG", AssetClass.EQUITY, "Alphabet"),
    "meta": ("META", AssetClass.EQUITY, "Meta"),
    "脸书": ("META", AssetClass.EQUITY, "Meta"),
    "facebook": ("META", AssetClass.EQUITY, "Meta"),
    "奈飞": ("NFLX", AssetClass.EQUITY, "Netflix"),
    "netflix": ("NFLX", AssetClass.EQUITY, "Netflix"),
    "amd": ("AMD", AssetClass.EQUITY, "AMD"),
    "博通": ("AVGO", AssetClass.EQUITY, "Broadcom"),
    "costco": ("COST", AssetClass.EQUITY, "Costco"),
    "可口可乐": ("KO", AssetClass.EQUITY, "Coca-Cola"),
    "伯克希尔": ("BRK.B", AssetClass.EQUITY, "Berkshire Hathaway"),
    "spy": ("SPY", AssetClass.ETF, "SPDR S&P 500 ETF"),
    "qqq": ("QQQ", AssetClass.ETF, "Invesco QQQ"),
    "iwm": ("IWM", AssetClass.ETF, "iShares Russell 2000"),
    "voo": ("VOO", AssetClass.ETF, "Vanguard S&P 500"),
    "纳指etf": ("QQQ", AssetClass.ETF, "Invesco QQQ"),
    "标普etf": ("SPY", AssetClass.ETF, "SPDR S&P 500 ETF"),
    "标普500": ("^GSPC", AssetClass.INDEX, "S&P 500"),
    "标普五百": ("^GSPC", AssetClass.INDEX, "S&P 500"),
    "s&p500": ("^GSPC", AssetClass.INDEX, "S&P 500"),
    "spx": ("^GSPC", AssetClass.INDEX, "S&P 500"),
    "纳斯达克100": ("^NDX", AssetClass.INDEX, "Nasdaq-100"),
    "ndx": ("^NDX", AssetClass.INDEX, "Nasdaq-100"),
    "纳斯达克": ("^IXIC", AssetClass.INDEX, "Nasdaq Composite"),
    "纳指": ("^IXIC", AssetClass.INDEX, "Nasdaq Composite"),
    "道琼斯": ("^DJI", AssetClass.INDEX, "Dow Jones"),
    "道指": ("^DJI", AssetClass.INDEX, "Dow Jones"),
}

_KNOWN_CN: dict[str, tuple[str, AssetClass, str]] = {
    "宁德时代": ("300750", AssetClass.EQUITY, "宁德时代"),
    "贵州茅台": ("600519", AssetClass.EQUITY, "贵州茅台"),
    "茅台": ("600519", AssetClass.EQUITY, "贵州茅台"),
    "比亚迪": ("002594", AssetClass.EQUITY, "比亚迪"),
    "招商银行": ("600036", AssetClass.EQUITY, "招商银行"),
    "中国平安": ("601318", AssetClass.EQUITY, "中国平安"),
    "平安银行": ("000001", AssetClass.EQUITY, "平安银行"),
    "沪深300": ("000300", AssetClass.INDEX, "沪深300"),
    "上证指数": ("000001", AssetClass.INDEX, "上证指数"),
    "上证综指": ("000001", AssetClass.INDEX, "上证指数"),
    "创业板指": ("399006", AssetClass.INDEX, "创业板指"),
    "科创50": ("000688", AssetClass.INDEX, "科创50"),
    "中证500": ("000905", AssetClass.INDEX, "中证500"),
    "沪深300etf": ("510300", AssetClass.ETF, "沪深300ETF"),
    "创业板etf": ("159915", AssetClass.ETF, "创业板ETF"),
}

_US_TICKER_STOPWORDS = frozenset(
    {
        "A",
        "I",
        "AM",
        "PM",
        "THE",
        "AND",
        "OR",
        "FOR",
        "TO",
        "IN",
        "ON",
        "OF",
        "ETF",
        "CEO",
        "CFO",
        "IPO",
        "EPS",
        "PE",
        "PB",
        "ROE",
        "USD",
        "USA",
        "AI",
        "IT",
        "GDP",
        "CPI",
        "API",
        "PDF",
        "FAQ",
        "SSE",
        "ALL",
        "NEW",
        "OLD",
    }
)


def parse_preferred_market(raw: str | None) -> Market | None:
    """解析用户偏好字符串为 Market；非法值返回 None。

    偏好仅允许 ``CN_A`` / ``US``（不含 MIXED）。
    """
    if not raw:
        return None
    text = raw.strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "CN": Market.CN_A,
        "CN_A": Market.CN_A,
        "A": Market.CN_A,
        "ASHARE": Market.CN_A,
        "A_SHARE": Market.CN_A,
        "US": Market.US,
        "USA": Market.US,
        "US_STOCK": Market.US,
        "AMERICA": Market.US,
    }
    if text in aliases:
        return aliases[text]
    try:
        m = Market(text)
        if m in (Market.CN_A, Market.US):
            return m
    except ValueError:
        pass
    return None


def parse_market_override(raw: str | None) -> Market | None:
    """解析请求级市场覆盖（允许 ``CN_A`` / ``US`` / ``MIXED``）。"""
    if not raw:
        return None
    text = raw.strip().upper().replace("-", "_").replace(" ", "_")
    if text in {"AUTO", ""}:
        return None
    preferred = parse_preferred_market(text)
    if preferred is not None:
        return preferred
    if text in {"MIXED", "BOTH", "CROSS", "CN_US"}:
        return Market.MIXED
    try:
        m = Market(text)
        if m in (Market.CN_A, Market.US, Market.MIXED):
            return m
    except ValueError:
        pass
    return None


def _find_known_symbols(query: str) -> list[SymbolRef]:
    q_lower = query.lower()
    found: list[SymbolRef] = []
    seen_tickers: set[str] = set()

    for name, (ticker, asset, display) in sorted(
        _KNOWN_US.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        if name in q_lower or name in query:
            if ticker in seen_tickers:
                continue
            seen_tickers.add(ticker)
            found.append(
                SymbolRef(
                    market=Market.US,
                    raw=name,
                    ticker=ticker,
                    asset_class=asset,
                    display_name=display,
                    confidence=0.9,
                )
            )

    for name, (ticker, asset, display) in sorted(
        _KNOWN_CN.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        if name in query:
            if ticker in seen_tickers:
                continue
            seen_tickers.add(ticker)
            found.append(
                SymbolRef(
                    market=Market.CN_A,
                    raw=name,
                    ticker=ticker,
                    asset_class=asset,
                    display_name=display,
                    confidence=0.9,
                )
            )
    return found


def extract_symbols_from_query(query: str) -> list[SymbolRef]:
    """从问句抽取可能的标的引用（代码 + 知名名）。"""
    symbols = _find_known_symbols(query)

    for m in _A_SHARE_CODE.finditer(query):
        code = m.group(1)
        if any(s.ticker == code for s in symbols):
            continue
        symbols.append(
            SymbolRef(
                market=Market.CN_A,
                raw=code,
                ticker=code,
                asset_class=AssetClass.EQUITY,
                confidence=0.95,
            )
        )

    known_us_tickers = {v[0] for v in _KNOWN_US.values()}
    for m in _US_TICKER_CANDIDATE.finditer(query):
        tok = m.group(1)
        if tok in _US_TICKER_STOPWORDS:
            continue
        if tok not in known_us_tickers and tok.lower() not in _KNOWN_US:
            continue
        ticker = tok if tok in known_us_tickers else _KNOWN_US[tok.lower()][0]
        if any(s.ticker == ticker for s in symbols):
            continue
        asset = AssetClass.ETF if ticker in {"SPY", "QQQ", "IWM", "VOO"} else AssetClass.EQUITY
        if ticker.startswith("^"):
            asset = AssetClass.INDEX
        symbols.append(
            SymbolRef(
                market=Market.US,
                raw=tok,
                ticker=ticker,
                asset_class=asset,
                confidence=0.85,
            )
        )
    return symbols


def detect_market_from_query(query: str) -> MarketResolution:
    """仅根据问句内容判定市场（不含用户偏好）。"""
    q = query.strip()
    q_lower = q.lower()
    reasons: list[str] = []
    symbols = extract_symbols_from_query(q)

    if any(k in q_lower or k in q for k in _MIXED_KEYWORDS):
        return MarketResolution(
            market=Market.MIXED,
            source="query_signal",
            confidence=0.85,
            symbols=tuple(symbols),
            reasons=("mixed_keyword",),
        )

    cn_kw = [k for k in _CN_MARKET_KEYWORDS if k in q_lower or k in q]
    us_kw = [k for k in _US_MARKET_KEYWORDS if k in q_lower]
    if cn_kw:
        reasons.append(f"cn_keyword:{','.join(cn_kw[:3])}")
    if us_kw:
        reasons.append(f"us_keyword:{','.join(us_kw[:3])}")

    cn_syms = [s for s in symbols if s.market == Market.CN_A]
    us_syms = [s for s in symbols if s.market == Market.US]
    if cn_syms:
        reasons.append(f"cn_symbol:{cn_syms[0].ticker or cn_syms[0].raw}")
    if us_syms:
        reasons.append(f"us_symbol:{us_syms[0].ticker or us_syms[0].raw}")

    has_cn = bool(cn_kw or cn_syms)
    has_us = bool(us_kw or us_syms)

    if has_cn and has_us:
        return MarketResolution(
            market=Market.MIXED,
            source="query_signal",
            confidence=0.8,
            symbols=tuple(symbols),
            reasons=tuple(reasons) or ("both_cn_us_signals",),
            notes="双边信号；按 MixedOrchestration 分侧路由后综合。",
        )
    if has_us:
        return MarketResolution(
            market=Market.US,
            source="query_signal",
            confidence=0.9 if us_syms else 0.75,
            symbols=tuple(symbols),
            reasons=tuple(reasons),
            notes="问句含美股信号；行情 us_* / 披露 us_filing_* / 新闻 us_news_* / 舆情 us_sentiment_*。",
        )
    if has_cn:
        return MarketResolution(
            market=Market.CN_A,
            source="query_signal",
            confidence=0.9 if cn_syms else 0.75,
            symbols=tuple(symbols),
            reasons=tuple(reasons),
        )
    return MarketResolution(
        market=Market.UNKNOWN,
        source="query_signal",
        confidence=0.0,
        symbols=tuple(symbols),
        reasons=("no_market_signal",),
    )


async def get_user_preferred_market(
    memory: MemoryManager,
    user_id: str,
) -> Market | None:
    from research_agent.memory.manager import MemoryNamespace

    item = await memory.get_memory(
        user_id,
        MemoryNamespace.USER_PREFERENCES,
        PREFERRED_MARKET_KEY,
    )
    if not item:
        return None
    raw = item.get("content") or item.get("market") or item.get("value")
    if isinstance(raw, dict):
        raw = raw.get("market") or raw.get("content")
    return parse_preferred_market(str(raw) if raw is not None else None)


async def set_user_preferred_market(
    memory: MemoryManager,
    user_id: str,
    market: Market | str,
) -> Market:
    from research_agent.memory.manager import MemoryNamespace

    parsed = market if isinstance(market, Market) else parse_preferred_market(str(market))
    if parsed not in (Market.CN_A, Market.US):
        raise ValueError("preferred_market 仅支持 CN_A 或 US")
    await memory.save_memory(
        user_id=user_id,
        namespace=MemoryNamespace.USER_PREFERENCES,
        key=PREFERRED_MARKET_KEY,
        value={"content": parsed.value, "market": parsed.value},
    )
    return parsed


async def resolve_market(
    query: str,
    *,
    memory: MemoryManager | None = None,
    user_id: str = "anonymous",
    override: Market | str | None = None,
    sticky_market: Market | str | None = None,
) -> MarketResolution:
    """完整市场解析：覆盖 → 问句 → 会话粘性 → 偏好 → 产品默认。"""
    if override is not None:
        ov = override if isinstance(override, Market) else parse_market_override(str(override))
        if ov is not None and ov in (Market.CN_A, Market.US, Market.MIXED):
            detected = detect_market_from_query(query)
            notes = ""
            if ov == Market.MIXED:
                notes = "请求强制 MIXED；按双边子任务分别路由（见 MixedOrchestration）。"
            return MarketResolution(
                market=ov,
                source="request_override",
                confidence=1.0,
                symbols=detected.symbols,
                reasons=("request_override", *detected.reasons),
                preferred_market=None,
                notes=notes or detected.notes,
            )

    preferred: Market | None = None
    if memory is not None and user_id != "anonymous":
        preferred = await get_user_preferred_market(memory, user_id)

    detected = detect_market_from_query(query)
    if detected.market != Market.UNKNOWN:
        return MarketResolution(
            market=detected.market,
            source=detected.source,
            confidence=detected.confidence,
            symbols=detected.symbols,
            reasons=detected.reasons,
            preferred_market=preferred,
            notes=detected.notes,
        )

    sticky: Market | None = None
    if sticky_market is not None:
        sticky = (
            sticky_market
            if isinstance(sticky_market, Market)
            else parse_market_override(str(sticky_market))
        )
        if sticky not in (Market.CN_A, Market.US, Market.MIXED):
            sticky = None

    if sticky is not None:
        return MarketResolution(
            market=sticky,
            source="thread_sticky",
            confidence=0.55,
            symbols=detected.symbols,
            reasons=("fallback_thread_sticky",),
            preferred_market=preferred,
            notes=(
                "问句无明确市场信号，沿用同会话上一轮市场。"
                + (" 应继续使用美股专家（us_*）。" if sticky == Market.US else "")
                + (" 跨市场子任务见 MixedOrchestration。" if sticky == Market.MIXED else "")
            ),
        )

    if preferred is not None:
        return MarketResolution(
            market=preferred,
            source="user_preference",
            confidence=0.6,
            symbols=detected.symbols,
            reasons=("fallback_preferred_market",),
            preferred_market=preferred,
            notes=(
                "问句无明确市场信号，使用用户偏好。"
                + (" 应路由至美股行情专家（us_*）。" if preferred == Market.US else "")
            ),
        )

    return MarketResolution(
        market=PRODUCT_DEFAULT_MARKET,
        source="default",
        confidence=0.3,
        symbols=detected.symbols,
        reasons=("product_default_cn_a",),
        preferred_market=None,
        notes="无问句信号且无用户偏好，默认 CN_A（当前已上线工具集）。",
    )


def format_market_preamble(resolution: MarketResolution, *, query: str = "") -> str:
    """写入 supervisor 的 SystemMessage 片段。

    ``query`` 可选：当 ``market=MIXED`` 时用于生成 ``[MixedOrchestration]`` 子任务清单。
    """
    from research_agent.market.orchestrate import build_mixed_orchestration_plan

    lines = [
        f"[MarketResolution] market={resolution.market.value} "
        f"source={resolution.source} confidence={resolution.confidence:.2f}",
    ]
    if resolution.preferred_market:
        lines.append(f"user_preferred_market={resolution.preferred_market.value}")
    if resolution.reasons:
        lines.append("reasons=" + ", ".join(resolution.reasons))
    if resolution.symbols:
        sym_parts = [
            f"{s.display_name or s.raw}({s.market.value}:{s.ticker or '?'}/{s.asset_class.value})"
            for s in resolution.symbols[:5]
        ]
        lines.append("symbols=" + "; ".join(sym_parts))
    if resolution.notes:
        lines.append("notes=" + resolution.notes)

    lines.append(
        "路由约束：CN_A → 使用已挂载的 A 股侧专家；"
        "US → 行情 us_* / 披露 us_filing_* / 新闻 us_news_* / 舆情 us_sentiment_*；"
        "禁止用 fin_* / news_* / sentiment_* / 巨潮 / fund_* 查美股；"
        "MIXED → 必须按下方 MixedOrchestration 分侧移交，最终分侧陈述再综合。"
    )

    plan = build_mixed_orchestration_plan(resolution, query)
    if plan is not None:
        lines.append(plan.format_for_prompt())

    return "\n".join(lines)
