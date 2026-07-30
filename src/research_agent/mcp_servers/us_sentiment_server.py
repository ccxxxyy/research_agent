"""MCP Server — 美股英文新闻情感量化（平行于 ``news_sentiment_server``）。

**不使用 SnowNLP**（中文模型）。主路径为 **VADER** compound 分 + 金融关键词增强，可复现、无 GPU。

打分文本优先 ``标题 + 摘要``；摘要过短且有 URL 时，再抓取页面 meta/正文前段（非全文）。

工具
----
1. ``analyze_text_sentiment`` — 任意英文文本批量打分
2. ``get_ticker_sentiment_report`` — 拉取 Yahoo 新闻 →（可选正文片段）→ 逐条打分 → 聚合报告

返回结构与 A 股 sentiment 工具同构：``sentiment_score ∈ [-1,1]``、标签、关键词、聚合统计、模型版本。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any

from fastmcp import FastMCP
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from research_agent.cache import TTL_SHORT, cached_tool

logger = logging.getLogger("us_sentiment_server")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

mcp = FastMCP("UsSentimentServer")

_MODEL_VERSION = "en_vader_finlex_v2"
MAX_LIMIT = 60
# 摘要短于此则尝试抓取页面片段，减轻「标题党」偏差
_THIN_SUMMARY_CHARS = 80
_BODY_SNIPPET_CHARS = 900
_BODY_FETCH_TIMEOUT = 4.0
_BODY_FETCH_WORKERS = 4
_AUX_SIGNAL_TIMEOUT_S = 10.0

_US_SIGNAL_WHAT = {
    "news": "Yahoo/Finnhub 新闻标题与摘要：VADER 打分的主样本（文本情绪）。",
    "social": "美股社交讨论（Reddit/Stocktwits）：本仓库尚未接入，故社交源通常为空。",
    "fund_flow": "报价涨跌与成交额（盘面代理）：资金情绪的间接信号，不是文本舆情分。",
    "analyst": "Yahoo 分析师评级/目标价：机构观点，不是散户讨论情绪。",
}

_POSITIVE_THRESHOLD = 0.15
_NEGATIVE_THRESHOLD = -0.15
_STRONG_POSITIVE = 0.45
_STRONG_NEGATIVE = -0.45

# 合成权重：VADER 为主，金融词表为增强。
# 0.4×clip(lex,0.45) 在 VADER≈0 时仍可达 positive 阈值 0.15。
_VADER_WEIGHT = 0.6
_LEXICON_WEIGHT = 0.4
_LEXICON_CLIP = 0.45

_VADER_ANALYZER: SentimentIntensityAnalyzer | None = None

# 英文金融关键词权重（命中即加减分；与 VADER 加权合成）
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


def _get_vader() -> SentimentIntensityAnalyzer:
    """懒加载单例，避免每条新闻重复构造 Analyzer。"""
    global _VADER_ANALYZER
    if _VADER_ANALYZER is None:
        _VADER_ANALYZER = SentimentIntensityAnalyzer()
    return _VADER_ANALYZER


def _lexicon_adjustment(text: str) -> tuple[float, list[str]]:
    """金融词表加减分；钳制到 ±_LEXICON_CLIP，避免淹没 VADER。"""
    lower = text.lower()
    matched: list[str] = []
    adjustment = 0.0
    # 长词优先，避免 "miss" 误伤；按长度降序匹配
    for kw, weight in sorted(_ALL_KEYWORDS.items(), key=lambda x: len(x[0]), reverse=True):
        if kw in lower:
            matched.append(kw)
            adjustment += weight

    if "!" in text and adjustment > 0:
        adjustment += 0.02
    if re.search(r"\b(not|no|never)\b.{0,20}\b(beat|growth|profit)\b", lower):
        adjustment -= 0.08

    adjustment = max(-_LEXICON_CLIP, min(_LEXICON_CLIP, adjustment))
    return adjustment, matched


def _label_from_score(final: float) -> str:
    if final >= _STRONG_POSITIVE:
        return "strong_positive"
    if final >= _POSITIVE_THRESHOLD:
        return "positive"
    if final <= _STRONG_NEGATIVE:
        return "strong_negative"
    if final <= _NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def _score_single(text: str) -> dict[str, Any]:
    """VADER compound（0.6）+ 金融词表（0.4）合成到 [-1, 1]。"""
    if not text or not text.strip():
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "keyword_adjustment": 0.0,
            "keywords_matched": [],
            "vader_compound": 0.0,
        }

    vader_compound = float(_get_vader().polarity_scores(text)["compound"])
    adjustment, matched = _lexicon_adjustment(text)
    final = max(
        -1.0,
        min(1.0, _VADER_WEIGHT * vader_compound + _LEXICON_WEIGHT * adjustment),
    )

    return {
        "sentiment_score": round(final, 4),
        "sentiment_label": _label_from_score(final),
        "keyword_adjustment": round(adjustment, 4),
        "keywords_matched": matched,
        "vader_compound": round(vader_compound, 4),
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
    summary = (
        content.get("summary")
        or content.get("description")
        or content.get("content")
        or raw.get("summary")
        or raw.get("description")
        or ""
    )
    if isinstance(summary, dict):
        summary = summary.get("description") or summary.get("summary") or ""
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
    from research_agent.text.urls import sanitize_http_url

    return {
        "title": str(title),
        "summary": str(summary).strip(),
        "publisher": provider,
        "published_at": str(pub),
        "url": sanitize_http_url(link),
    }


def _http_get_bytes(url: str, *, timeout: float = _BODY_FETCH_TIMEOUT) -> bytes | None:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    }
    try:
        from curl_cffi import requests as curl_requests

        resp = curl_requests.get(url, headers=headers, impersonate="chrome", timeout=timeout)
        if resp.status_code == 200 and resp.content:
            return bytes(resp.content)
    except Exception:  # noqa: BLE001
        pass
    try:
        import requests

        sess = requests.Session()
        sess.trust_env = False
        try:
            resp = sess.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200 and resp.content:
                return resp.content
        finally:
            sess.close()
    except Exception:  # noqa: BLE001
        return None
    return None


def _extract_meta_description(html: str) -> str:
    patterns = [
        r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:description["\']',
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']description["\']',
    ]
    for pat in patterns:
        m = re.search(pat, html, flags=re.IGNORECASE)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


def _html_to_text_snippet(html: str, *, max_chars: int = _BODY_SNIPPET_CHARS) -> str:
    """粗提取正文前段：去 script/style/标签，不做完整 DOM 解析。"""
    cleaned = re.sub(
        r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>",
        " ",
        html,
    )
    cleaned = re.sub(r"(?is)<!--.*?-->", " ", cleaned)
    cleaned = re.sub(r"(?is)<br\s*/?>", "\n", cleaned)
    cleaned = re.sub(r"(?is)</p\s*>", "\n", cleaned)
    cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"&nbsp;|&amp;|&lt;|&gt;|&quot;|&#39;", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_chars]


def _fetch_article_snippet(url: str) -> str:
    """抓取新闻页 meta description 或正文前段；失败返回空串。"""
    if not url or not url.startswith("http"):
        return ""
    raw = _http_get_bytes(url)
    if not raw:
        return ""
    try:
        html = raw.decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return ""
    meta = _extract_meta_description(html)
    if len(meta) >= 40:
        return meta[:_BODY_SNIPPET_CHARS]
    body = _html_to_text_snippet(html)
    # 过滤过短/导航垃圾
    if len(body) < 40:
        return meta or ""
    return body


def _build_score_text(title: str, summary: str, body: str) -> tuple[str, str]:
    """组装打分文本，并返回依据标签。"""
    parts = [title.strip()] if title.strip() else []
    basis = ["title"]
    if summary and summary.strip():
        parts.append(summary.strip())
        basis.append("summary")
    # body 与 summary 重复时不再叠一段
    if body and body.strip():
        b = body.strip()
        if not summary or b[:80].lower() not in summary.lower():
            parts.append(b)
            basis.append("body")
    text = ". ".join(parts)
    return text, "+".join(basis) if basis else "title"


def _enrich_thin_summaries(news_list: list[dict[str, Any]]) -> None:
    """对摘要过短且带 URL 的条目并行抓取页面片段，写入 ``body_snippet``。"""
    need: list[tuple[int, str]] = []
    for i, news in enumerate(news_list):
        summary = (news.get("summary") or "").strip()
        url = (news.get("url") or "").strip()
        if url and len(summary) < _THIN_SUMMARY_CHARS:
            need.append((i, url))
        if len(need) >= 5:  # 最多补 5 条，避免拖垮整单舆情超时
            break
    if not need:
        return

    def _one(url: str) -> str:
        try:
            return _fetch_article_snippet(url)
        except Exception:  # noqa: BLE001
            return ""

    with ThreadPoolExecutor(max_workers=_BODY_FETCH_WORKERS) as pool:
        futures = {pool.submit(_one, url): idx for idx, url in need}
        try:
            for fut in as_completed(futures, timeout=12.0):
                idx = futures[fut]
                try:
                    snippet = fut.result(timeout=0.1)
                except Exception:  # noqa: BLE001
                    snippet = ""
                if snippet:
                    news_list[idx]["body_snippet"] = snippet
        except TimeoutError:
            logger.warning("body enrichment truncated after 12s (%s urls)", len(need))


def _fetch_news_via_yahoo_search(symbol: str, limit: int) -> list[dict[str, Any]]:
    """Yahoo search news HTTP（休市也可用），避开 yfinance.news 挂起。"""
    from urllib.parse import quote

    ticker = _normalize_ticker(symbol)
    url = (
        "https://query1.finance.yahoo.com/v1/finance/search"
        f"?q={quote(ticker)}&quotesCount=0&newsCount={max(limit, 20)}&listsCount=0"
    )
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    payload: dict[str, Any] | None = None
    try:
        from curl_cffi import requests as curl_requests

        resp = curl_requests.get(url, headers=headers, impersonate="chrome", timeout=20)
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
                resp = sess.get(url, headers=headers, timeout=20)
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
        provider = ""
        if isinstance(raw.get("publisher"), str):
            provider = raw["publisher"]
        link = ""
        for key in ("link", "url"):
            if isinstance(raw.get(key), str) and raw[key]:
                link = raw[key]
                break
        summary = ""
        for key in ("summary", "description", "body", "content"):
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                summary = val.strip()
                break
            if isinstance(val, dict):
                inner = val.get("description") or val.get("summary") or ""
                if isinstance(inner, str) and inner.strip():
                    summary = inner.strip()
                    break
        from research_agent.text.urls import sanitize_http_url

        out.append(
            {
                "title": title,
                "summary": summary,
                "publisher": provider,
                "published_at": str(raw.get("publishTime") or raw.get("providerPublishTime") or ""),
                "url": sanitize_http_url(link),
            }
        )
        if len(out) >= limit:
            break
    return out


def _fetch_scored_news(symbol: str, limit: int) -> tuple[list[dict[str, Any]], list[str]]:
    from research_agent.mcp_servers.us_news_pipeline import collect_us_news

    ticker = _normalize_ticker(symbol)
    logger.info("fetching sentiment news for %s limit=%s", ticker, limit)

    pull = min(40, max(limit * 2, limit + 5))
    # 复用 us_news 分源限时拉取，避免本处再串行挂死 yfinance
    from research_agent.mcp_servers.us_news_server import _fetch_yahoo_news_raw

    yahoo = _fetch_yahoo_news_raw(ticker, pull)
    source = "yahoo"

    bundle = collect_us_news(ticker, yahoo_items=yahoo, limit=limit)
    news_list = bundle.get("news") or []
    providers = bundle.get("providers_used") or []
    if providers:
        source = "+".join(providers)
    # 摘要过短时补抓页面 meta/正文前段，缓解标题夸大
    _enrich_thin_summaries(news_list)
    body_hits = sum(1 for n in news_list if n.get("body_snippet"))
    if body_hits:
        logger.info(
            "enriched %s/%s news items with article snippets for %s",
            body_hits,
            len(news_list),
            ticker,
        )

    items: list[dict[str, Any]] = []
    texts: list[str] = []
    for news in news_list:
        summary = (news.get("summary") or "").strip()
        body = (news.get("body_snippet") or "").strip()
        combined, basis = _build_score_text(news["title"], summary, body)
        texts.append(combined)
        scored = _score_single(combined)
        preview = summary or body
        scored.update(
            {
                "title": news["title"],
                "content_preview": preview[:240],
                "publish_time": news.get("published_at") or "",
                "source_site": news.get("publisher") or "",
                "news_url": news.get("url") or "",
                "text_fingerprint": _text_fingerprint(combined),
                "fetch_source": source,
                "score_text_basis": basis,
                "event_type": news.get("event_type") or "other",
                "event_label_zh": news.get("event_label_zh") or "其他",
                "cluster_size": news.get("cluster_size") or 1,
                "provider": news.get("provider") or "",
            }
        )
        items.append(scored)
    logger.info("sentiment news ready for %s: %s items via %s", ticker, len(items), source)
    return items, texts


def _call_aux_timeout(fn, *, timeout: float, default, label: str = ""):
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        fut = pool.submit(fn)
        try:
            return fut.result(timeout=max(0.1, float(timeout)))
        except FuturesTimeoutError:
            logger.warning("%s timed out after %.1fs", label or "aux", timeout)
            fut.cancel()
            return default
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s failed: %s", label or "aux", exc)
            return default
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _fetch_board_proxy(symbol: str) -> dict[str, Any]:
    """用近价涨跌/成交额作盘面代理（非 Level-2 资金流）。"""
    import yfinance as yf

    t = yf.Ticker(_normalize_ticker(symbol))
    fi = getattr(t, "fast_info", None)
    price = getattr(fi, "last_price", None) if fi is not None else None
    prev = getattr(fi, "previous_close", None) if fi is not None else None
    volume = getattr(fi, "last_volume", None) if fi is not None else None
    chg_pct = None
    if price is not None and prev not in (None, 0):
        try:
            chg_pct = (float(price) - float(prev)) / float(prev) * 100.0
        except Exception:  # noqa: BLE001
            chg_pct = None
    if price is None and prev is None:
        return {"available": False, "reason": "empty_quote"}
    return {
        "available": True,
        "price": price,
        "previous_close": prev,
        "change_percent": round(chg_pct, 4) if chg_pct is not None else None,
        "volume": volume,
        "source": "yfinance_fast_info",
        "source_url": f"https://finance.yahoo.com/quote/{_normalize_ticker(symbol)}",
    }


def _fetch_analyst_us(symbol: str) -> dict[str, Any]:
    """Yahoo 分析师摘要 / 目标价（旁路）。"""
    import yfinance as yf

    ticker = _normalize_ticker(symbol)
    t = yf.Ticker(ticker)
    summary = None
    targets = None
    try:
        summary = t.recommendations_summary
    except Exception:  # noqa: BLE001
        summary = None
    try:
        targets = t.analyst_price_targets
    except Exception:  # noqa: BLE001
        targets = None

    summary_rows: list[dict[str, Any]] = []
    if summary is not None and hasattr(summary, "empty") and not summary.empty:
        for _, row in summary.head(6).iterrows():
            summary_rows.append({str(k): v for k, v in row.items()})

    target_info: dict[str, Any] = {}
    if isinstance(targets, dict):
        for k in ("current", "low", "high", "mean", "median"):
            if k in targets:
                target_info[k] = targets.get(k)
    elif targets is not None and hasattr(targets, "to_dict"):
        try:
            target_info = {str(k): v for k, v in dict(targets).items()}
        except Exception:  # noqa: BLE001
            target_info = {}

    if not summary_rows and not target_info:
        return {"available": False, "reason": "empty", "recommendations": [], "price_targets": {}}
    return {
        "available": True,
        "recommendations": summary_rows,
        "price_targets": target_info,
        "source": "yfinance_analyst",
        "source_url": f"https://finance.yahoo.com/quote/{ticker}/analysis",
    }


def _build_us_aux_signals(
    *,
    board: dict[str, Any] | None,
    analyst: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    social = {
        "what": _US_SIGNAL_WHAT["social"],
        "used": False,
        "available": False,
        "skipped": True,
        "reason": "reddit_stocktwits_not_wired",
    }

    b = board or {"available": False}
    board_block = {"what": _US_SIGNAL_WHAT["fund_flow"], **b, "used": bool(b.get("available"))}
    if board_block["used"]:
        notes.append("已纳入盘面代理（近价涨跌/成交额）：属资金情绪间接信号，不是文本舆情分。")

    a = analyst or {"available": False}
    analyst_block = {"what": _US_SIGNAL_WHAT["analyst"], **a, "used": bool(a.get("available"))}
    if analyst_block["used"]:
        notes.append("已纳入分析师信号（Yahoo 评级/目标价）：表示机构观点，不是散户舆情。")

    return (
        {
            "news_what": _US_SIGNAL_WHAT["news"],
            "social": social,
            "fund_flow": board_block,
            "analyst": analyst_block,
        },
        notes,
    )


@mcp.tool()
async def analyze_text_sentiment(texts: list[str]) -> dict:
    """对英文文本列表做金融情感评分（VADER + 金融词表，可复现，不走大模型）。

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
async def get_ticker_sentiment_report(symbol: str, limit: int = 30) -> dict:
    """一站式美股 ticker 舆情报告：Yahoo 新闻 → 逐条打分 → 聚合。

    Args:
        symbol: 美股 ticker，如 ``AAPL``、``TSLA``、``SPY``、``QQQ``。
            **禁止** A 股数字代码或残缺参数（如 ``000``、``000001``）。
        limit: 新闻条数上限（1–60，默认 30；样本越多正/中/负占比越稳）。
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
        fetch_src = (items[0].get("fetch_source") if items else None) or "yahoo_search"
        board = _call_aux_timeout(
            lambda: _fetch_board_proxy(ticker),
            timeout=_AUX_SIGNAL_TIMEOUT_S,
            default={"available": False, "skipped": True, "reason": "timeout_or_error"},
            label=f"board_proxy({ticker})",
        )
        analyst = _call_aux_timeout(
            lambda: _fetch_analyst_us(ticker),
            timeout=_AUX_SIGNAL_TIMEOUT_S,
            default={
                "available": False,
                "skipped": True,
                "reason": "timeout_or_error",
                "recommendations": [],
                "price_targets": {},
            },
            label=f"analyst_us({ticker})",
        )
        aux_signals, signal_notes = _build_us_aux_signals(board=board, analyst=analyst)
        return {
            "symbol": ticker,
            "model_version": _MODEL_VERSION,
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "items": items,
            "aggregate": _aggregate_scores(items),
            "hot_words": _extract_hot_words(texts),
            "source": fetch_src,
            "source_url": f"https://finance.yahoo.com/quote/{ticker.lstrip('^')}/news",
            "language": "en",
            "available_off_hours": True,
            "aux_signals": aux_signals,
            "signal_notes": signal_notes,
        }

    try:
        return await asyncio.wait_for(asyncio.to_thread(_work), timeout=90.0)
    except TimeoutError:
        return {
            "error": "TimeoutError: Yahoo 舆情拉取超过 90s",
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
