"""美股新闻管道：双源合并、垃圾过滤、相似度聚类、事件标签。

供 ``us_news_server`` / ``us_sentiment_server`` 共用。
不做 LLM 新闻内容抽取；Finnhub 无 key 时优雅降级为 Yahoo-only。
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger("us_news_pipeline")

_CLUSTER_THRESHOLD = 0.72

# publisher / 域名（小写子串匹配）
_JUNK_PUBLISHERS: frozenset[str] = frozenset(
    {
        "motley fool advertorial",
        "investorplace tipranks",
        "penny stock",
        "stockstotrade",
        "benzinga promos",
    }
)
_JUNK_HOST_FRAGMENTS: tuple[str, ...] = (
    "pennystock",
    "stockpromoter",
    "hotstock",
    "getrich",
    "surefire",
)

_TRUSTED_PUBLISHERS: tuple[str, ...] = (
    "reuters",
    "bloomberg",
    "wall street journal",
    "wsj",
    "associated press",
    "ap news",
    "financial times",
    "ft.com",
    "cnbc",
    "marketwatch",
    "barron",
    "sec.gov",
    "pr newswire",
    "business wire",
)

_EVENT_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "earnings",
        "财报",
        ("earnings", "guidance", "eps", "quarterly results", "q1 ", "q2 ", "q3 ", "q4 "),
    ),
    ("m_and_a", "并购", ("acquire", "acquisition", "merger", "buyout", "takeover")),
    ("sec_8k", "SEC披露", ("8-k", "form 8k", "sec filing", "edgar")),
    (
        "analyst",
        "分析师",
        ("upgrade", "downgrade", "price target", "initiates coverage", "analyst"),
    ),
    (
        "legal",
        "诉讼监管",
        ("lawsuit", "litigation", "sec charges", "probe", "antitrust", "investigation"),
    ),
)


def finnhub_api_key() -> str:
    try:
        from research_agent.config import get_settings

        return (get_settings().finnhub_api_key or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _norm_title(title: str) -> str:
    t = (title or "").lower()
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _host(url: str) -> str:
    try:
        return (urlparse(url or "").netloc or "").lower()
    except Exception:  # noqa: BLE001
        return ""


def is_junk_item(item: dict[str, Any]) -> bool:
    """垃圾源 / 促销标题启发式。"""
    pub = str(item.get("publisher") or "").lower()
    title = str(item.get("title") or "")
    host = _host(str(item.get("url") or ""))

    for bad in _JUNK_PUBLISHERS:
        if bad in pub:
            return True
    for frag in _JUNK_HOST_FRAGMENTS:
        if frag in host or frag in pub:
            return True

    if title.count("!") >= 3:
        return True
    if re.search(r"\bpenny stocks?\b", title, re.I):
        return True
    if re.search(r"\b(guaranteed|get rich|secret stock)\b", title, re.I):
        return True
    # 过度全大写短标题
    letters = re.sub(r"[^A-Za-z]", "", title)
    return bool(len(letters) >= 12 and letters.isupper())


def tag_event(item: dict[str, Any]) -> tuple[str, str]:
    text = f"{item.get('title') or ''} {item.get('summary') or ''}".lower()
    provider = str(item.get("provider") or item.get("source") or "").lower()
    if "sec" in provider or "edgar" in provider or item.get("form") == "8-K":
        return "sec_8k", "SEC披露"
    for etype, label_zh, kws in _EVENT_RULES:
        if any(k in text for k in kws):
            return etype, label_zh
    return "other", "其他"


def _trust_rank(item: dict[str, Any]) -> tuple[int, int, str]:
    pub = str(item.get("publisher") or "").lower()
    for i, name in enumerate(_TRUSTED_PUBLISHERS):
        if name in pub:
            return (0, i, str(item.get("published_at") or ""))
    return (1, 99, str(item.get("published_at") or ""))


def _title_similar(a: str, b: str) -> float:
    na, nb = _norm_title(a), _norm_title(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def _same_cluster(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ua, ub = str(a.get("url") or ""), str(b.get("url") or "")
    if ua and ub and ua == ub:
        return True
    return (
        _title_similar(str(a.get("title") or ""), str(b.get("title") or "")) >= _CLUSTER_THRESHOLD
    )


def cluster_news_items(items: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    """标题/URL 相似度聚类；每簇选代表条。"""
    clusters: list[list[dict[str, Any]]] = []
    for item in items:
        placed = False
        for cluster in clusters:
            if _same_cluster(item, cluster[0]):
                cluster.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])

    out: list[dict[str, Any]] = []
    for cluster in clusters:
        cluster.sort(key=_trust_rank)
        rep = dict(cluster[0])
        urls = []
        providers: list[str] = []
        for it in cluster:
            u = str(it.get("url") or "")
            if u and u != rep.get("url") and u not in urls:
                urls.append(u)
            p = str(it.get("provider") or it.get("source") or "")
            if p and p not in providers:
                providers.append(p)
        etype, label = tag_event(rep)
        rep["cluster_size"] = len(cluster)
        rep["cluster_urls"] = urls[:3]
        rep["providers_in_cluster"] = providers
        rep["event_type"] = etype
        rep["event_label_zh"] = label
        out.append(rep)
        if len(out) >= limit:
            break
    return out


def filter_and_cluster(items: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    cleaned = [it for it in items if it.get("title") and not is_junk_item(it)]
    return cluster_news_items(cleaned, limit=limit)


def fetch_finnhub_company_news(
    symbol: str,
    *,
    limit: int = 30,
    api_key: str | None = None,
    days: int = 7,
) -> list[dict[str, Any]]:
    """Finnhub company-news → 统一条目（provider=finnhub）。"""
    key = (api_key if api_key is not None else finnhub_api_key()).strip()
    if not key:
        return []
    ticker = symbol.strip().upper().lstrip("^")
    if not ticker:
        return []
    end = date.today()
    start = end - timedelta(days=max(1, int(days)))
    from urllib.parse import urlencode

    params = urlencode(
        {
            "symbol": ticker,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "token": key,
        }
    )
    url = f"https://finnhub.io/api/v1/company-news?{params}"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    payload: Any = None
    try:
        from curl_cffi import requests as curl_requests

        resp = curl_requests.get(url, headers=headers, impersonate="chrome", timeout=12)
        if resp.status_code == 200:
            payload = resp.json()
        else:
            logger.warning("finnhub company-news HTTP %s for %s", resp.status_code, ticker)
    except Exception as exc:  # noqa: BLE001
        logger.warning("finnhub company-news failed for %s: %s", ticker, exc)
        payload = None
    if payload is None:
        try:
            import requests

            sess = requests.Session()
            sess.trust_env = False
            try:
                resp = sess.get(url, headers=headers, timeout=12)
                if resp.status_code == 200:
                    payload = resp.json()
            finally:
                sess.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("finnhub company-news requests failed for %s: %s", ticker, exc)
            return []

    if not isinstance(payload, list):
        return []

    from research_agent.text.urls import sanitize_http_url

    out: list[dict[str, Any]] = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("headline") or raw.get("title") or "").strip()
        if not title:
            continue
        ts = raw.get("datetime")
        published = ""
        if isinstance(ts, (int, float)) and ts > 0:
            try:
                from datetime import UTC, datetime

                published = datetime.fromtimestamp(float(ts), tz=UTC).isoformat()
            except Exception:  # noqa: BLE001
                published = str(ts)
        out.append(
            {
                "title": title,
                "summary": str(raw.get("summary") or "")[:500],
                "publisher": str(raw.get("source") or "Finnhub"),
                "published_at": published,
                "url": sanitize_http_url(str(raw.get("url") or "")),
                "provider": "finnhub",
                "source": "finnhub",
            }
        )
        if len(out) >= limit:
            break
    return out


def merge_news_sources(
    *batches: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """合并多源条目；返回 (items, providers_used)。"""
    merged: list[dict[str, Any]] = []
    providers: list[str] = []
    for batch in batches:
        for it in batch or []:
            item = dict(it)
            prov = str(item.get("provider") or item.get("source") or "unknown")
            item.setdefault("provider", prov)
            if prov not in providers and batch:
                # 仅当该 batch 非空时记入
                pass
            merged.append(item)
        if batch:
            # 取 batch 内第一个 provider 标记
            p0 = str((batch[0] or {}).get("provider") or (batch[0] or {}).get("source") or "")
            if p0 and p0 not in providers:
                providers.append(p0)
            elif not p0:
                # yahoo_search / yfinance
                for it in batch:
                    p = str(it.get("source") or it.get("provider") or "")
                    if p and p not in providers:
                        providers.append(p)
                        break
    return merged, providers


def collect_us_news(
    symbol: str,
    *,
    yahoo_items: list[dict[str, Any]],
    limit: int = 15,
    fetch_finnhub: bool = True,
    finnhub_key: str | None = None,
) -> dict[str, Any]:
    """Yahoo 条目 + 可选 Finnhub → 过滤 → 聚类 → 标签。

    Returns:
        ``{"news": [...], "providers_used": [...], "note": str|None, "raw_count": int}``
    """
    limit = max(1, min(int(limit), 40))
    # 拉多一点再聚类，避免误伤后条数不足
    pull_n = min(40, max(limit * 2, limit + 5))

    yahoo = list(yahoo_items or [])[:pull_n]
    for it in yahoo:
        it.setdefault("provider", str(it.get("source") or "yahoo"))

    fh: list[dict[str, Any]] = []
    note: str | None = None
    key = finnhub_key if finnhub_key is not None else finnhub_api_key()
    if fetch_finnhub:
        if key:
            fh = fetch_finnhub_company_news(symbol, limit=pull_n, api_key=key)
        else:
            note = "未配置 FINNHUB_API_KEY：新闻仅 Yahoo；报价跳过 Finnhub。配置后新闻与行情第二源同时启用。"

    merged, providers = merge_news_sources(yahoo, fh)
    # 规范化 provider 列表
    providers_used: list[str] = []
    for it in merged:
        p = str(it.get("provider") or it.get("source") or "")
        if p.startswith("yahoo"):
            p = "yahoo"
        if p and p not in providers_used:
            providers_used.append(p)

    clustered = filter_and_cluster(merged, limit=limit)
    return {
        "news": clustered,
        "providers_used": providers_used,
        "note": note,
        "raw_count": len(merged),
        "count": len(clustered),
    }
