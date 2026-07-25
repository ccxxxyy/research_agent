"""MCP Server — 美股英文新闻情感量化（平行于 ``news_sentiment_server``）。

**不使用 SnowNLP**（中文模型）。PoC 采用可复现的英文金融关键词词典 + 极性规则。

工具
----
1. ``analyze_text_sentiment`` — 任意英文文本批量打分
2. ``get_ticker_sentiment_report`` — 拉取 Yahoo 新闻 → 逐条打分 → 聚合报告

返回结构与 A 股 sentiment 工具同构：``sentiment_score ∈ [-1,1]``、标签、关键词、聚合统计、模型版本。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from fastmcp import FastMCP

from research_agent.cache import TTL_SHORT, cached_tool

logger = logging.getLogger("us_sentiment_server")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

mcp = FastMCP("UsSentimentServer")

_MODEL_VERSION = "en_fin_keywords_v1"
MAX_LIMIT = 40

_POSITIVE_THRESHOLD = 0.15
_NEGATIVE_THRESHOLD = -0.15
_STRONG_POSITIVE = 0.45
_STRONG_NEGATIVE = -0.45

# 英文金融关键词权重（命中即加减分；可后续换 VADER）
_POSITIVE_KEYWORDS: dict[str, float] = {
    "beat": 0.18,
    "beats": 0.18,
    "surge": 0.2,
    "surges": 0.2,
    "rally": 0.18,
    "rallies": 0.18,
    "soar": 0.22,
    "soars": 0.22,
    "record high": 0.2,
    "upgrade": 0.16,
    "upgraded": 0.16,
    "outperform": 0.15,
    "bullish": 0.18,
    "growth": 0.1,
    "profit": 0.12,
    "profits": 0.12,
    "strong demand": 0.15,
    "raised guidance": 0.2,
    "raises guidance": 0.2,
    "dividend hike": 0.15,
    "buyback": 0.12,
    "approved": 0.1,
    "breakthrough": 0.14,
}

_NEGATIVE_KEYWORDS: dict[str, float] = {
    "miss": 0.18,
    "misses": 0.18,
    "plunge": 0.22,
    "plunges": 0.22,
    "crash": 0.25,
    "selloff": 0.18,
    "sell-off": 0.18,
    "downgrade": 0.18,
    "downgraded": 0.18,
    "bearish": 0.18,
    "lawsuit": 0.14,
    "probe": 0.12,
    "investigation": 0.14,
    "fraud": 0.22,
    "bankruptcy": 0.25,
    "layoff": 0.15,
    "layoffs": 0.15,
    "cut guidance": 0.2,
    "cuts guidance": 0.2,
    "warning": 0.12,
    "weak demand": 0.15,
    "recall": 0.14,
    "fine": 0.1,
    "penalty": 0.12,
    "default": 0.2,
}

_ALL_KEYWORDS: dict[str, float] = {
    **{k: v for k, v in _POSITIVE_KEYWORDS.items()},
    **{k: -v for k, v in _NEGATIVE_KEYWORDS.items()},
}

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "from",
        "with",
        "by",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "their",
        "they",
        "them",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "his",
        "her",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "about",
        "into",
        "over",
        "after",
        "before",
        "more",
        "most",
        "than",
        "then",
        "also",
        "just",
        "only",
        "not",
        "no",
        "yes",
        "up",
        "down",
        "out",
        "off",
        "new",
        "says",
        "said",
        "say",
    ]
)


def _fmt_error(exc: Exception, *, context: str) -> dict[str, Any]:
    logger.error("[%s] %s: %s", context, type(exc).__name__, exc)
    return {"error": f"{type(exc).__name__}: {exc}", "context": context}


def _text_fingerprint(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _normalize_ticker(symbol: str) -> str:
    return symbol.strip().upper().lstrip("$")


def _is_valid_us_ticker(symbol: str) -> bool:
    """拒绝 A 股数字代码 / 残缺参数（如 ``000``），避免 Yahoo 空转挂起。"""
    s = _normalize_ticker(symbol)
    if not s or len(s) > 12:
        return False
    if s.isdigit():
        return False
    # 允许 ^GSPC、BRK.B、BRK-B、SPY
    return bool(re.fullmatch(r"\^?[A-Z][A-Z0-9.\-]{0,11}", s))


def _score_single(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "keyword_adjustment": 0.0,
            "keywords_matched": [],
        }

    lower = text.lower()
    matched: list[str] = []
    adjustment = 0.0
    # 长词优先，避免 "miss" 误伤；按长度降序匹配
    for kw, weight in sorted(_ALL_KEYWORDS.items(), key=lambda x: len(x[0]), reverse=True):
        if kw in lower:
            matched.append(kw)
            adjustment += weight

    # 轻度标点/叹号偏置
    if "!" in text and adjustment > 0:
        adjustment += 0.02
    if re.search(r"\b(not|no|never)\b.{0,20}\b(beat|growth|profit)\b", lower):
        adjustment -= 0.08

    final = max(-1.0, min(1.0, adjustment))
    # 无关键词时保持中性，避免空文瞎猜
    if not matched:
        final = 0.0

    if final >= _STRONG_POSITIVE:
        label = "strong_positive"
    elif final >= _POSITIVE_THRESHOLD:
        label = "positive"
    elif final <= _STRONG_NEGATIVE:
        label = "strong_negative"
    elif final <= _NEGATIVE_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"

    return {
        "sentiment_score": round(final, 4),
        "sentiment_label": label,
        "keyword_adjustment": round(adjustment, 4),
        "keywords_matched": matched,
    }


def _aggregate_scores(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "overall_label": "no_data",
            "overall_score": 0.0,
            "positive_count": 0,
            "neutral_count": 0,
            "negative_count": 0,
            "positive_ratio": 0.0,
            "neutral_ratio": 0.0,
            "negative_ratio": 0.0,
            "sample_size": 0,
        }

    scores = [it["sentiment_score"] for it in items]
    avg = sum(scores) / len(scores)
    n = len(scores)
    pos = sum(1 for s in scores if s >= _POSITIVE_THRESHOLD)
    neg = sum(1 for s in scores if s <= _NEGATIVE_THRESHOLD)
    neu = n - pos - neg

    if avg >= _STRONG_POSITIVE:
        overall = "strong_positive"
    elif avg >= _POSITIVE_THRESHOLD:
        overall = "positive"
    elif avg <= _STRONG_NEGATIVE:
        overall = "strong_negative"
    elif avg <= _NEGATIVE_THRESHOLD:
        overall = "negative"
    else:
        overall = "neutral"

    return {
        "overall_label": overall,
        "overall_score": round(avg, 4),
        "positive_count": pos,
        "neutral_count": neu,
        "negative_count": neg,
        "positive_ratio": round(pos / n, 4),
        "neutral_ratio": round(neu / n, 4),
        "negative_ratio": round(neg / n, 4),
        "sample_size": n,
    }


def _extract_hot_words(texts: list[str], top_n: int = 15) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
        for tok in tokens:
            if tok in _STOPWORDS:
                continue
            counter[tok] += 1
    out: list[dict[str, Any]] = []
    for word, count in counter.most_common(top_n):
        weight = _ALL_KEYWORDS.get(word)
        out.append(
            {
                "word": word,
                "count": count,
                "sentiment_weight": round(weight, 3) if weight is not None else None,
            }
        )
    return out


def _normalize_news_item(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    content = raw.get("content") if isinstance(raw.get("content"), dict) else raw
    title = content.get("title") or raw.get("title") or ""
    if not title:
        return None
    summary = content.get("summary") or content.get("description") or ""
    provider = ""
    prov = content.get("provider") or raw.get("provider")
    if isinstance(prov, dict):
        provider = str(prov.get("displayName") or "")
    elif isinstance(prov, str):
        provider = prov
    pub = (
        content.get("pubDate") or content.get("displayTime") or raw.get("providerPublishTime") or ""
    )
    if isinstance(pub, (int, float)):
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
    return {
        "title": str(title),
        "summary": str(summary),
        "publisher": provider,
        "published_at": str(pub),
        "url": link,
    }


def _fetch_scored_news(symbol: str, limit: int) -> tuple[list[dict[str, Any]], list[str]]:
    import logging

    import yfinance as yf

    # 压低 yfinance 在限流时的刷屏，避免干扰排障
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    ticker = _normalize_ticker(symbol)
    logger.info("fetching sentiment news for %s limit=%s", ticker, limit)
    try:
        raw_list = yf.Ticker(ticker).news or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance.news failed for %s: %s", ticker, exc)
        raw_list = []
    items: list[dict[str, Any]] = []
    texts: list[str] = []
    for raw in raw_list:
        news = _normalize_news_item(raw)
        if not news:
            continue
        combined = news["title"]
        if news["summary"]:
            combined = f"{news['title']}. {news['summary']}"
        texts.append(combined)
        scored = _score_single(combined)
        scored.update(
            {
                "title": news["title"],
                "content_preview": (news["summary"] or "")[:200],
                "publish_time": news["published_at"],
                "source_site": news["publisher"],
                "news_url": news["url"],
                "text_fingerprint": _text_fingerprint(combined),
            }
        )
        items.append(scored)
        if len(items) >= limit:
            break
    logger.info("sentiment news ready for %s: %s items", ticker, len(items))
    return items, texts


@mcp.tool()
async def analyze_text_sentiment(texts: list[str]) -> dict:
    """对英文文本列表做金融情感评分（关键词词典，可复现，不走大模型）。

    Args:
        texts: 待打分文本列表。
    """
    if not texts:
        return {
            "model_version": _MODEL_VERSION,
            "items": [],
            "aggregate": _aggregate_scores([]),
            "hot_words": [],
        }

    def _work() -> tuple[list[dict], list[dict]]:
        scored = []
        for text in texts[:MAX_LIMIT]:
            info = _score_single(text)
            info["text_preview"] = text[:120]
            info["text_fingerprint"] = _text_fingerprint(text)
            scored.append(info)
        return scored, _extract_hot_words(texts[:MAX_LIMIT])

    try:
        scored, words = await asyncio.to_thread(_work)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context="analyze_text_sentiment()")

    return {
        "model_version": _MODEL_VERSION,
        "items": scored,
        "aggregate": _aggregate_scores(scored),
        "hot_words": words,
        "language": "en",
    }


@mcp.tool()
@cached_tool(ttl=TTL_SHORT, namespace="us_sentiment")
async def get_ticker_sentiment_report(symbol: str, limit: int = 20) -> dict:
    """一站式美股 ticker 舆情报告：Yahoo 新闻 → 逐条打分 → 聚合。

    Args:
        symbol: 美股 ticker，如 ``AAPL``、``TSLA``、``SPY``、``QQQ``。
            **禁止** A 股数字代码或残缺参数（如 ``000``、``000001``）。
        limit: 新闻条数上限（1–40）。
    """
    limit = max(1, min(int(limit), MAX_LIMIT))
    ticker = _normalize_ticker(symbol)
    if not ticker:
        return {"error": "symbol 不能为空", "context": "get_ticker_sentiment_report()"}
    if not _is_valid_us_ticker(ticker):
        return {
            "error": (
                f"无效美股 ticker: {symbol!r}。请传入如 SPY/QQQ/AAPL；不要传 A 股代码或残缺数字。"
            ),
            "context": "get_ticker_sentiment_report()",
            "symbol": ticker,
        }

    def _work() -> dict[str, Any]:
        items, texts = _fetch_scored_news(ticker, limit)
        return {
            "symbol": ticker,
            "model_version": _MODEL_VERSION,
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "items": items,
            "aggregate": _aggregate_scores(items),
            "hot_words": _extract_hot_words(texts),
            "source": "yfinance",
            "source_url": f"https://finance.yahoo.com/quote/{ticker.lstrip('^')}/news",
            "language": "en",
        }

    try:
        return await asyncio.wait_for(asyncio.to_thread(_work), timeout=45.0)
    except TimeoutError:
        return {
            "error": "TimeoutError: Yahoo 舆情拉取超过 45s",
            "context": f"get_ticker_sentiment_report(symbol={symbol!r})",
            "symbol": ticker,
        }
    except Exception as e:  # noqa: BLE001
        return _fmt_error(
            e,
            context=f"get_ticker_sentiment_report(symbol={symbol!r}, limit={limit})",
        )


if __name__ == "__main__":
    mcp.run(transport="stdio")
