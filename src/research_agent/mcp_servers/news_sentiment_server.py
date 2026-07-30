"""MCP Server — 结构化新闻情感分析（可计量、可对账）。

定位
----
``news_server`` 提供原始新闻文本流（东财 / 财联社 / 百度 / 雪球），
本服务在其上叠加 结构化情感评分 + 高频话题提取 + 多源融合，输出可量化、可审计的个股舆情报告。

数据源
------
1. 东方财富个股新闻 — 标题 + 摘要，逐条打分。
2. 雪球讨论热度榜 — 标的讨论量排名（stock_hot_tweet_xq），如果目标股票出现在榜上，融合其讨论量作为"热度信号"。
3. 东方财富热搜关键词 — 个股关联的热搜词（stock_hot_keyword_em），作为"市场关注点"直接呈现。

输出维度
--------
1. 逐条新闻打分 — score / label / 命中关键词 / 文本指纹
2. 聚合统计 — 正负比例、均分、样本量
3. 高频讨论词 — 从所有新闻标题+摘要中提取 top-N 高频词
4. 话题聚类 — 按正/负/中性分组的代表性标题
5. 雪球热度 — 讨论量排名位次（如有）
6. 东财热搜词 — 当前关联热搜话题
7. 审计元数据 — model_version / timestamp / fingerprint


雪球返回的是个股讨论热度排行榜——一张表，每行是一只股票，包含：股票代码、股票简称、讨论量（有多少人在讨论这只票）、最新股价。
不是词汇、不是帖子内容、不是新闻——是"哪些股票被讨论得最多"的排名。
具体例子：假设查宁德时代（300750）的舆情报告，雪球这一步做的是：
拉取"最热门"讨论榜（比如榜上 500 只股票） 在里面找 300750
如果找到了，返回：
{
  "on_list": true,        ← 宁德时代在榜上
  "rank": 8,              ← 排第 8 名
  "total_ranked": 500,    ← 一共 500 只票入榜
  "discussion_volume": 12345,  ← 讨论量 12345
  "stock_name": "宁德时代",
  "latest_price": 215.30
}
如果不在榜上，返回 {"on_list": false, "total_ranked": 500}
在整个舆情报告里的角色：东财新闻告诉"大家在说什么"（内容），雪球告诉"有多少人在讨论这只票"（热度数字），东财热搜词告诉"和这只票关联的概念是什么"（话题标签）。
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, TypeVar

from fastmcp import FastMCP

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("news_sentiment_server")
_T = TypeVar("_T")

# 雪球「最热门」榜常因全量分页卡住；旁路限时且并行，超时则跳过。
_XUEQIU_HEAT_TIMEOUT_S = 5.0
_HOT_KEYWORDS_TIMEOUT_S = 10.0
_NEWS_FETCH_TIMEOUT_S = 35.0
# 外层预算：新闻 + 旁路并行 + 打分；留余量给 MCP 冷启动
_FULL_REPORT_TIMEOUT_S = 120.0
# 资金流 / 研报旁路（研报已改为单页直连，通常 <3s）
_AUX_SIGNAL_TIMEOUT_S = 18.0

_SIGNAL_WHAT = {
    "news": "东财个股新闻标题/摘要：本地 SnowNLP 打分的主样本（文本情绪）。",
    "social": "雪球讨论热度 + 东财热搜词：社交关注度与关联概念，不是逐条情绪分。",
    "fund_flow": "个股主力资金流向（东财）：盘面资金进退的间接情绪代理，不是社交媒体舆情。",
    "analyst": "券商研报评级（东财研报）：机构观点/评级，不是散户讨论情绪。",
}


def _call_with_timeout[T](fn: Callable[[], _T], *, timeout: float, default: _T) -> _T:
    """在独立线程跑同步函数；超时返回 default。

    ``shutdown(wait=False)``：超时后不阻塞等待雪球等慢 IO，主流程可带着 partial 结果继续。
    """
    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(fn)
    try:
        return fut.result(timeout=timeout)
    except FuturesTimeout:
        logger.warning("timed out after %.1fs: %s", timeout, getattr(fn, "__name__", repr(fn)))
        return default
    except Exception:  # noqa: BLE001
        logger.exception("call failed: %s", getattr(fn, "__name__", repr(fn)))
        return default
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


for _proxy_key in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
):
    os.environ.pop(_proxy_key, None)
os.environ["NO_PROXY"] = "*"

mcp = FastMCP("NewsSentiment")

# ---------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------
MAX_LIMIT = 50
_SHANGHAI_TZ = timezone(timedelta(hours=8))

_POSITIVE_THRESHOLD = 0.25
_NEGATIVE_THRESHOLD = -0.25
_STRONG_POSITIVE = 0.50
_STRONG_NEGATIVE = -0.50

_MODEL_VERSION = "snownlp-0.12.3+fin_keywords_v2"

# ---------------------------------------------------------------------
# 金融情感关键词词典
# ---------------------------------------------------------------------
_POSITIVE_KEYWORDS: dict[str, float] = {
    # ── 业绩类 ──
    "业绩预增": 0.25,
    "业绩大增": 0.28,
    "超预期": 0.22,
    "净利润增长": 0.20,
    "净利润增": 0.20,
    "净利润上涨": 0.25,
    "净利上涨": 0.25,
    "净利润为": 0.10,
    "营收增长": 0.18,
    "营收增": 0.15,
    "扭亏为盈": 0.28,
    "盈利": 0.10,
    "利润增": 0.12,
    "同比增长": 0.12,
    "同比增": 0.10,
    "环比增长": 0.10,
    "环比增": 0.08,
    "创新高": 0.15,
    "历史新高": 0.18,
    "上涨": 0.08,
    "增速": 0.08,
    "高增长": 0.18,
    "翻倍": 0.22,
    "预喜": 0.18,
    "大幅增长": 0.22,
    "快速增长": 0.15,
    "稳步增长": 0.10,
    "复合增长": 0.12,
    "毛利率提升": 0.12,
    "净利率提升": 0.12,
    # ── 市场类 ──
    "涨停": 0.18,
    "大涨": 0.15,
    "突破": 0.08,
    "放量上涨": 0.15,
    "主力净流入": 0.12,
    "北向资金": 0.06,
    "增持": 0.15,
    "回购": 0.15,
    "分红": 0.12,
    "派息": 0.10,
    "送转": 0.08,
    "抢筹": 0.12,
    "净流入": 0.10,
    "连阳": 0.10,
    "新高": 0.12,
    "反弹": 0.06,
    "底部放量": 0.08,
    "强势": 0.08,
    "领涨": 0.10,
    "拉升": 0.08,
    # ── 政策/行业 ──
    "利好": 0.18,
    "政策支持": 0.15,
    "补贴": 0.12,
    "获批": 0.12,
    "中标": 0.15,
    "签约": 0.10,
    "合作": 0.05,
    "战略投资": 0.12,
    "产能扩张": 0.10,
    "订单": 0.10,
    "新产品": 0.10,
    "技术突破": 0.15,
    "募资": 0.06,
    "投资": 0.04,
    "成立": 0.03,
    "落地": 0.06,
    "量产": 0.10,
    "出海": 0.08,
    "全球化": 0.06,
    "国产替代": 0.12,
    "自主可控": 0.10,
    "降本增效": 0.10,
    "市占率提升": 0.12,
}

_NEGATIVE_KEYWORDS: dict[str, float] = {
    # ── 业绩类 ──
    "业绩预减": -0.22,
    "业绩下滑": -0.22,
    "业绩变脸": -0.25,
    "亏损": -0.20,
    "净利润下降": -0.18,
    "净利润下滑": -0.18,
    "净利润下": -0.15,
    "营收下降": -0.15,
    "营收下滑": -0.15,
    "同比下降": -0.12,
    "同比下滑": -0.10,
    "环比下降": -0.10,
    "毛利率下滑": -0.15,
    "计提减值": -0.18,
    "商誉减值": -0.20,
    "大幅下降": -0.22,
    "持续亏损": -0.22,
    "增收不增利": -0.12,
    # ── 市场类 ──
    "跌停": -0.18,
    "大跌": -0.15,
    "闪崩": -0.22,
    "暴跌": -0.25,
    "主力净流出": -0.12,
    "减持": -0.18,
    "高管减持": -0.22,
    "质押": -0.12,
    "爆仓": -0.25,
    "破位": -0.10,
    "缩量下跌": -0.12,
    "阴跌": -0.08,
    "套牢": -0.10,
    "割肉": -0.08,
    "破发": -0.12,
    "连跌": -0.10,
    # ── 风险/事件 ──
    "利空": -0.18,
    "处罚": -0.20,
    "立案调查": -0.28,
    "违规": -0.20,
    "退市": -0.28,
    "ST": -0.22,
    "*ST": -0.28,
    "风险提示": -0.15,
    "诉讼": -0.12,
    "被告": -0.15,
    "暂停上市": -0.25,
    "停牌": -0.10,
    "监管": -0.10,
    "警告": -0.15,
    "罚款": -0.18,
    "约谈": -0.12,
    "洗钱": -0.25,
    "失联": -0.22,
    "跑路": -0.28,
    "造假": -0.28,
    "虚增": -0.25,
    "起诉": -0.15,
    "被起诉": -0.18,
    "公诉": -0.20,
    "做空": -0.15,
    "谣言": -0.05,
    "扰动": -0.05,
}

_ALL_KEYWORDS: dict[str, float] = {**_POSITIVE_KEYWORDS, **_NEGATIVE_KEYWORDS}


def _fmt_error(exc: Exception, *, context: str) -> dict[str, Any]:
    return {"error": f"{type(exc).__name__}: {exc}", "context": context}


def _coerce_limit(limit: int) -> int:
    return max(1, min(int(limit), MAX_LIMIT))


def _text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------
# 中文分词 + 停用词（用 SnowNLP 内置分词，无额外依赖）
# ---------------------------------------------------------------------
_STOPWORDS: frozenset[str] = frozenset(
    [
        "的",
        "了",
        "在",
        "是",
        "我",
        "有",
        "和",
        "就",
        "不",
        "人",
        "都",
        "一",
        "一个",
        "上",
        "也",
        "很",
        "到",
        "说",
        "要",
        "去",
        "你",
        "会",
        "着",
        "没有",
        "看",
        "好",
        "自己",
        "这",
        "他",
        "她",
        "它",
        "们",
        "那",
        "被",
        "从",
        "把",
        "让",
        "用",
        "对",
        "为",
        "与",
        "及",
        "等",
        "但",
        "而",
        "或",
        "之",
        "其",
        "且",
        "因",
        "如",
        "所",
        "能",
        "更",
        "将",
        "已",
        "由",
        "于",
        "可",
        "以",
        "中",
        "个",
        "年",
        "月",
        "日",
        "时",
        "分",
        "万",
        "亿",
        "元",
        "股",
        "公司",
        "发布",
        "公告",
        "表示",
        "记者",
        "截至",
        "目前",
        "同时",
        "以及",
        "相关",
        "根据",
        "显示",
        "数据",
        "通过",
        "进行",
        "报告",
        "方面",
        "市场",
        "行业",
        "中国",
        "以来",
        "来源",
        "编辑",
        "责任",
        "证券",
        "财经",
        "新闻",
        "消息",
        "据悉",
        "获悉",
        "上述",
        "下述",
        "其中",
        "此外",
        "此前",
        "针对",
        "关于",
        "这是",
        "该股",
        "SZ",
        "SH",
    ]
)

_CN_CHAR_RE = re.compile(r"[\u4e00-\u9fff]+")


def _extract_keywords_from_texts(texts: list[str], top_n: int = 15) -> list[dict[str, Any]]:
    """从一批文本中提取高频有意义的词。

    用 SnowNLP 分词 + 停用词过滤 + 词频统计。
    返回 [{word, count, sentiment_weight}] 按 count 降序。
    """
    from snownlp import SnowNLP

    counter: dict[str, int] = collections.Counter()
    for text in texts:
        if not text.strip():
            continue
        cn_text = "".join(_CN_CHAR_RE.findall(text))
        if len(cn_text) < 2:
            continue
        try:
            words = SnowNLP(cn_text).words
        except Exception:  # noqa: BLE001
            continue
        for w in words:
            if len(w) >= 2 and w not in _STOPWORDS:
                counter[w] += 1

    result = []
    for word, count in counter.most_common(top_n):
        weight = _ALL_KEYWORDS.get(word, 0.0)
        result.append(
            {
                "word": word,
                "count": count,
                "sentiment_weight": round(weight, 3) if weight else None,
            }
        )
    return result


def _build_topic_clusters(items: list[dict[str, Any]]) -> dict[str, list[str]]:
    """将已打分的新闻按情感分组，每组取最多 3 条代表性标题。"""
    pos_titles = []
    neg_titles = []
    neu_titles = []

    for it in items:
        s = it.get("sentiment_score", 0)
        title = it.get("title", "")[:60]
        if not title:
            continue
        if s >= _POSITIVE_THRESHOLD:
            pos_titles.append(title)
        elif s <= _NEGATIVE_THRESHOLD:
            neg_titles.append(title)
        else:
            neu_titles.append(title)

    return {
        "positive_headlines": pos_titles[:3],
        "negative_headlines": neg_titles[:3],
        "neutral_headlines": neu_titles[:3],
    }


# ---------------------------------------------------------------------
# 核心评分引擎
# ---------------------------------------------------------------------
def _score_single(text: str) -> dict[str, Any]:
    from snownlp import SnowNLP

    if not text or not text.strip():
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "中性",
            "snownlp_raw": 0.5,
            "keyword_adjustment": 0.0,
            "keywords_matched": [],
        }

    s = SnowNLP(text)
    raw_prob = s.sentiments
    base_score = 2.0 * raw_prob - 1.0

    matched: list[str] = []
    adjustment = 0.0
    text_lower = text.lower()
    for kw, weight in _ALL_KEYWORDS.items():
        if kw.lower() in text_lower:
            matched.append(kw)
            adjustment += weight

    final = max(-1.0, min(1.0, base_score + adjustment))

    if final >= _STRONG_POSITIVE:
        label = "强正面"
    elif final >= _POSITIVE_THRESHOLD:
        label = "正面"
    elif final <= _STRONG_NEGATIVE:
        label = "强负面"
    elif final <= _NEGATIVE_THRESHOLD:
        label = "负面"
    else:
        label = "中性"

    return {
        "sentiment_score": round(final, 4),
        "sentiment_label": label,
        "snownlp_raw": round(raw_prob, 4),
        "keyword_adjustment": round(adjustment, 4),
        "keywords_matched": matched,
    }


def _aggregate_scores(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "overall_label": "无数据",
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
        overall = "强正面"
    elif avg >= _POSITIVE_THRESHOLD:
        overall = "偏正面"
    elif avg <= _STRONG_NEGATIVE:
        overall = "强负面"
    elif avg <= _NEGATIVE_THRESHOLD:
        overall = "偏负面"
    else:
        overall = "中性"

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


# ---------------------------------------------------------------------
# 雪球热度查询
# ---------------------------------------------------------------------
def _fetch_xueqiu_heat(symbol: str) -> dict[str, Any] | None:
    """查询雪球讨论热度榜，看目标股票是否在榜上。"""
    import akshare as ak

    try:
        df = ak.stock_hot_tweet_xq(symbol="最热门")
        if df is None or df.empty:
            return None
        if "关注" in df.columns:
            df = df.rename(columns={"关注": "讨论量"})
        code_col = "股票代码" if "股票代码" in df.columns else "代码"
        if code_col not in df.columns:
            for c in df.columns:
                if "代码" in c:
                    code_col = c
                    break
        match = df[df[code_col].astype(str).str.contains(symbol)]
        if match.empty:
            total = len(df)
            return {"on_list": False, "total_ranked": total}
        row = match.iloc[0]
        rank = int(match.index[0]) + 1
        return {
            "on_list": True,
            "rank": rank,
            "total_ranked": len(df),
            "discussion_volume": int(row.get("讨论量", 0)),
            "stock_name": str(row.get("股票简称", row.get("简称", ""))),
            "latest_price": float(row.get("最新价", 0)),
        }
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------
# 东财热搜关键词
# ---------------------------------------------------------------------
def _fetch_hot_keywords(symbol: str) -> list[dict[str, Any]]:
    """获取个股关联的东财热搜词。"""
    import akshare as ak

    bare = symbol.strip().upper()
    if not bare.startswith(("SH", "SZ")):
        prefix = "SH" if bare.startswith("6") else "SZ"
        bare = f"{prefix}{bare}"
    try:
        df = ak.stock_hot_keyword_em(symbol=bare)
        if df is None or df.empty:
            return []
        records = []
        for _, row in df.head(10).iterrows():
            records.append(
                {
                    "keyword": str(row.get("概念名称", "")),
                    "hot_value": str(row.get("热度", "")),
                    "time": str(row.get("时间", "")),
                }
            )
        return records
    except Exception:  # noqa: BLE001
        return []


def _exchange_prefix_simple(symbol: str) -> str:
    return "sh" if str(symbol).startswith("6") else "sz"


def _is_na(v: Any) -> bool:
    try:
        import pandas as pd

        return bool(pd.isna(v))
    except Exception:  # noqa: BLE001
        return v is None


def _fetch_fund_flow_signal(symbol: str, *, days: int = 5) -> dict[str, Any]:
    """个股近期资金流向摘要（旁路）。"""
    import akshare as ak

    try:
        df = ak.stock_individual_fund_flow(stock=symbol, market=_exchange_prefix_simple(symbol))
    except Exception as exc:  # noqa: BLE001 — 东财偶发断连，旁路失败即可
        return {"available": False, "reason": f"error:{type(exc).__name__}"}
    if df is None or getattr(df, "empty", True):
        return {"available": False, "reason": "empty"}
    tail = df.tail(max(1, min(int(days), 10)))
    records: list[dict[str, Any]] = []
    for _, row in tail.iterrows():
        records.append({str(k): (None if _is_na(v) else v) for k, v in row.items()})
    latest = records[-1] if records else {}
    main_net = None
    for key in ("主力净流入-净额", "主力净流入净额", "主力净流入"):
        if key in latest and latest[key] is not None:
            main_net = latest[key]
            break
    return {
        "available": True,
        "days": len(records),
        "latest_main_net_inflow": main_net,
        "recent": records[-3:],
        "source": "eastmoney_fund_flow",
        "source_url": f"https://data.eastmoney.com/zjlx/{symbol}.html",
    }


def _http_get_json(url: str, *, params: dict[str, str], timeout: float = 10.0) -> dict[str, Any]:
    """东财 JSON GET：优先 curl_cffi（Chrome 指纹），回退 requests。"""
    try:
        from curl_cffi import requests as curl_requests

        resp = curl_requests.get(url, params=params, impersonate="chrome", timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return data
    except Exception:  # noqa: BLE001
        logger.debug("curl_cffi get failed for %s; fallback requests", url, exc_info=True)

    import requests

    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise TypeError(f"expected JSON object from {url}")
    return data


def _fetch_analyst_reports(symbol: str, *, limit: int = 8) -> dict[str, Any]:
    """东财个股研报列表（旁路，机构评级）。

    注意：不要用 ``ak.stock_research_report_em``——它会按 TotalPage 翻完全部历史页
    （热门股可达数十页），轻易超过旁路超时。这里只请求第 1 页、``pageSize=limit``。
    """
    page_size = max(1, min(int(limit), 15))
    end_year = datetime.now(tz=_SHANGHAI_TZ).year + 1
    params = {
        "industryCode": "*",
        "pageSize": str(page_size),
        "industry": "*",
        "rating": "*",
        "ratingChange": "*",
        "beginTime": "2018-01-01",
        "endTime": f"{end_year}-01-01",
        "pageNo": "1",
        "fields": "",
        "qType": "0",
        "orgCode": "",
        "code": str(symbol).zfill(6),
        "rcode": "",
        "p": "1",
        "pageNum": "1",
        "pageNumber": "1",
    }
    try:
        data_json = _http_get_json(
            "https://reportapi.eastmoney.com/report/list",
            params=params,
            timeout=10.0,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "reason": f"http_error:{type(exc).__name__}",
            "reports": [],
        }

    raw_rows = data_json.get("data") or []
    if not isinstance(raw_rows, list) or not raw_rows:
        return {"available": False, "reason": "empty", "reports": []}

    rows: list[dict[str, Any]] = []
    for item in raw_rows[:page_size]:
        if not isinstance(item, dict):
            continue
        info_code = str(item.get("infoCode") or "")
        rows.append(
            {
                "title": str(item.get("title") or ""),
                "rating": str(item.get("emRatingName") or item.get("sRatingName") or ""),
                "institution": str(item.get("orgSName") or item.get("orgName") or ""),
                "date": str(item.get("publishDate") or ""),
                "industry": str(item.get("indvInduName") or item.get("industryName") or ""),
                "pdf_url": (f"https://pdf.dfcfw.com/pdf/H3_{info_code}_1.pdf" if info_code else ""),
            }
        )
    ratings = [
        r["rating"] for r in rows if r.get("rating") and r["rating"] not in ("", "nan", "None")
    ]
    return {
        "available": bool(rows),
        "count": len(rows),
        "ratings_sample": ratings[:8],
        "reports": rows,
        "source": "eastmoney_research_report",
        "source_url": f"https://data.eastmoney.com/report/stock.jshtml?stockcode={symbol}",
    }


def _build_aux_signals(
    *,
    xueqiu_heat: dict[str, Any],
    eastmoney_keywords: list[dict[str, Any]],
    fund_flow: dict[str, Any] | None,
    analyst: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    """组装旁路信号 + 白话说明（用到才提示）。"""
    notes: list[str] = []
    social_used = bool(
        (isinstance(xueqiu_heat, dict) and xueqiu_heat.get("on_list")) or eastmoney_keywords
    )
    social = {
        "what": _SIGNAL_WHAT["social"],
        "xueqiu_heat": xueqiu_heat,
        "eastmoney_trending_keywords": eastmoney_keywords[:10],
        "used": social_used,
    }
    if social_used:
        notes.append(
            "已纳入社交关注信号（雪球热度/东财热搜）：表示讨论热度与关联概念，不是新闻正文情绪分。"
        )

    ff = fund_flow or {"available": False}
    fund_block = {"what": _SIGNAL_WHAT["fund_flow"], **ff, "used": bool(ff.get("available"))}
    if fund_block["used"]:
        notes.append("已纳入资金/盘面信号（个股主力资金流向）：表示资金进退，属间接情绪代理。")

    an = analyst or {"available": False}
    analyst_block = {"what": _SIGNAL_WHAT["analyst"], **an, "used": bool(an.get("available"))}
    if analyst_block["used"]:
        notes.append("已纳入分析师/研报信号（东财研报评级）：表示机构观点，不是散户舆情。")

    return (
        {
            "news_what": _SIGNAL_WHAT["news"],
            "social": social,
            "fund_flow": fund_block,
            "analyst": analyst_block,
        },
        notes,
    )


# ---------------------------------------------------------------------
# Tool 1: 纯文本情感打分
# ---------------------------------------------------------------------
@mcp.tool()
async def analyze_text_sentiment(texts: list[str]) -> dict:
    """对一组文本逐条做金融情感评分 + 高频词提取。

    不依赖外部数据源。评分模型：SnowNLP + 金融关键词词典（v2）。

    Args:
        texts: 待打分的文本列表（每条建议 < 2000 字符）。

    Returns:
        items（逐条分数）+ aggregate（聚合）+ hot_words（高频词）。
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
        words = _extract_keywords_from_texts(texts[:MAX_LIMIT])
        return scored, words

    try:
        scored, words = await asyncio.to_thread(_work)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context="analyze_text_sentiment()")

    return {
        "model_version": _MODEL_VERSION,
        "items": scored,
        "aggregate": _aggregate_scores(scored),
        "hot_words": words,
    }


# ---------------------------------------------------------------------
# Tool 2: 一站式个股舆情报告（东财新闻 + 雪球热度 + 热搜词 + 高频词）
# ---------------------------------------------------------------------
def _fetch_eastmoney_news_df(symbol: str):
    """东财个股新闻 DataFrame；失败返回 None。"""
    import akshare as ak

    try:
        return ak.stock_news_em(symbol=symbol)
    except Exception:  # noqa: BLE001
        return None


def _full_report(symbol: str, limit: int) -> dict[str, Any]:
    """同步：拉新闻 + 打分 + 旁路并行（雪球/热搜/资金/研报）。"""
    timestamp = datetime.now(tz=_SHANGHAI_TZ).isoformat()
    notes: list[str] = []

    # ── 1. 东财新闻 + 打分（核心路径，限时）──
    df = _call_with_timeout(
        lambda: _fetch_eastmoney_news_df(symbol),
        timeout=_NEWS_FETCH_TIMEOUT_S,
        default=None,
    )
    if df is None:
        notes.append(f"eastmoney_news_timeout_or_empty>{_NEWS_FETCH_TIMEOUT_S:.0f}s")

    items: list[dict[str, Any]] = []
    all_texts: list[str] = []

    if df is not None and not getattr(df, "empty", True):
        for _, row in df.head(limit).iterrows():
            title = str(row.get("新闻标题", ""))
            content = str(row.get("新闻内容", ""))
            combined = f"{title}。{content}" if content else title
            all_texts.append(combined)

            info = _score_single(combined)
            info["title"] = title
            info["content_preview"] = content[:200] if content else ""
            info["publish_time"] = str(row.get("发布时间", ""))
            info["source_site"] = str(row.get("文章来源", ""))
            info["news_url"] = str(row.get("新闻链接", ""))
            info["text_fingerprint"] = _text_fingerprint(combined)
            items.append(info)

    # ── 2. 高频讨论词 / 话题聚类（本地，快）──
    hot_words = _extract_keywords_from_texts(all_texts) if all_texts else []
    topic_clusters = _build_topic_clusters(items)

    # ── 3. 旁路并行（总耗时约等于最慢一路）
    # 注意：不可用 ``with ThreadPoolExecutor()``（退出时 shutdown(wait=True)），
    # 否则已超时仍在跑的雪球/东财线程会把整份报告堵死数十分钟。
    xueqiu_heat: dict[str, Any] = {
        "on_list": False,
        "skipped": True,
        "reason": "timeout_or_error",
    }
    eastmoney_keywords: list[dict[str, Any]] = []
    fund_flow: dict[str, Any] = {"available": False, "skipped": True, "reason": "timeout_or_error"}
    analyst: dict[str, Any] = {
        "available": False,
        "skipped": True,
        "reason": "timeout_or_error",
        "reports": [],
    }

    pool = ThreadPoolExecutor(max_workers=4)
    try:
        fut_map = {
            "xueqiu": pool.submit(_fetch_xueqiu_heat, symbol),
            "keywords": pool.submit(_fetch_hot_keywords, symbol),
            "fund": pool.submit(_fetch_fund_flow_signal, symbol),
            "analyst": pool.submit(_fetch_analyst_reports, symbol),
        }
        try:
            got = fut_map["xueqiu"].result(timeout=_XUEQIU_HEAT_TIMEOUT_S)
            if got is not None:
                xueqiu_heat = got
            else:
                notes.append(f"xueqiu_heat_skipped_timeout>{_XUEQIU_HEAT_TIMEOUT_S:.0f}s")
        except FuturesTimeout:
            notes.append(f"xueqiu_heat_skipped_timeout>{_XUEQIU_HEAT_TIMEOUT_S:.0f}s")
            fut_map["xueqiu"].cancel()
        except Exception:  # noqa: BLE001
            notes.append("xueqiu_heat_error")
            logger.exception("xueqiu_heat failed for %s", symbol)

        try:
            got_kw = fut_map["keywords"].result(timeout=_HOT_KEYWORDS_TIMEOUT_S)
            eastmoney_keywords = got_kw if isinstance(got_kw, list) else []
            if not eastmoney_keywords:
                notes.append(f"eastmoney_keywords_empty_or_timeout>{_HOT_KEYWORDS_TIMEOUT_S:.0f}s")
        except FuturesTimeout:
            notes.append(f"eastmoney_keywords_empty_or_timeout>{_HOT_KEYWORDS_TIMEOUT_S:.0f}s")
            fut_map["keywords"].cancel()
        except Exception:  # noqa: BLE001
            notes.append("eastmoney_keywords_error")
            logger.exception("hot_keywords failed for %s", symbol)

        try:
            got_ff = fut_map["fund"].result(timeout=_AUX_SIGNAL_TIMEOUT_S)
            if isinstance(got_ff, dict):
                fund_flow = got_ff
            else:
                notes.append(f"fund_flow_skipped_timeout>{_AUX_SIGNAL_TIMEOUT_S:.0f}s")
        except FuturesTimeout:
            notes.append(f"fund_flow_skipped_timeout>{_AUX_SIGNAL_TIMEOUT_S:.0f}s")
            fut_map["fund"].cancel()
        except Exception:  # noqa: BLE001
            notes.append("fund_flow_error")
            logger.exception("fund_flow failed for %s", symbol)

        try:
            got_an = fut_map["analyst"].result(timeout=_AUX_SIGNAL_TIMEOUT_S)
            if isinstance(got_an, dict):
                analyst = got_an
            else:
                notes.append(f"analyst_skipped_timeout>{_AUX_SIGNAL_TIMEOUT_S:.0f}s")
        except FuturesTimeout:
            notes.append(f"analyst_skipped_timeout>{_AUX_SIGNAL_TIMEOUT_S:.0f}s")
            fut_map["analyst"].cancel()
        except Exception:  # noqa: BLE001
            notes.append("analyst_error")
            logger.exception("analyst_reports failed for %s", symbol)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    aux_signals, signal_notes = _build_aux_signals(
        xueqiu_heat=xueqiu_heat,
        eastmoney_keywords=eastmoney_keywords or [],
        fund_flow=fund_flow,
        analyst=analyst,
    )

    out: dict[str, Any] = {
        "symbol": symbol,
        "source": "eastmoney+xueqiu+fund_flow+analyst",
        "source_url": f"https://so.eastmoney.com/news/s?keyword={symbol}",
        "model_version": _MODEL_VERSION,
        "timestamp": timestamp,
        "items": items,
        "aggregate": _aggregate_scores(items),
        "hot_words": hot_words,
        "topic_clusters": topic_clusters,
        "xueqiu_heat": xueqiu_heat,
        "eastmoney_trending_keywords": eastmoney_keywords,
        "aux_signals": aux_signals,
        "signal_notes": signal_notes,
    }
    if notes:
        out["partial_notes"] = notes
    return out


def _news_only_report(symbol: str, limit: int) -> dict[str, Any]:
    """外层超时兜底：只保留东财新闻打分，旁路一律 skipped。"""
    timestamp = datetime.now(tz=_SHANGHAI_TZ).isoformat()
    df = _fetch_eastmoney_news_df(symbol)
    items: list[dict[str, Any]] = []
    all_texts: list[str] = []
    if df is not None and not getattr(df, "empty", True):
        for _, row in df.head(limit).iterrows():
            title = str(row.get("新闻标题", ""))
            content = str(row.get("新闻内容", ""))
            combined = f"{title}。{content}" if content else title
            all_texts.append(combined)
            info = _score_single(combined)
            info["title"] = title
            info["content_preview"] = content[:200] if content else ""
            info["publish_time"] = str(row.get("发布时间", ""))
            info["source_site"] = str(row.get("文章来源", ""))
            info["news_url"] = str(row.get("新闻链接", ""))
            info["text_fingerprint"] = _text_fingerprint(combined)
            items.append(info)
    empty_heat = {"on_list": False, "skipped": True, "reason": "outer_timeout_fallback"}
    empty_ff = {"available": False, "skipped": True, "reason": "outer_timeout_fallback"}
    empty_an = {
        "available": False,
        "skipped": True,
        "reason": "outer_timeout_fallback",
        "reports": [],
    }
    aux_signals, signal_notes = _build_aux_signals(
        xueqiu_heat=empty_heat,
        eastmoney_keywords=[],
        fund_flow=empty_ff,
        analyst=empty_an,
    )
    return {
        "symbol": symbol,
        "source": "eastmoney_news_only_fallback",
        "source_url": f"https://so.eastmoney.com/news/s?keyword={symbol}",
        "model_version": _MODEL_VERSION,
        "timestamp": timestamp,
        "items": items,
        "aggregate": _aggregate_scores(items),
        "hot_words": _extract_keywords_from_texts(all_texts) if all_texts else [],
        "topic_clusters": _build_topic_clusters(items),
        "xueqiu_heat": empty_heat,
        "eastmoney_trending_keywords": [],
        "aux_signals": aux_signals,
        "signal_notes": signal_notes,
        "partial_notes": [
            f"full_report_outer_timeout>{_FULL_REPORT_TIMEOUT_S:.0f}s_fallback_news_only"
        ],
    }


@mcp.tool()
async def get_stock_sentiment_report(symbol: str, limit: int = 20) -> dict:
    """一站式个股舆情报告（多源融合）。

    拉东财新闻 → 逐条打分 → 提取高频词 → 话题分组 → 查雪球热度 → 拉东财热搜词
    →（旁路）个股资金流向 →（旁路）东财研报评级 → 汇总。
    雪球/热搜/资金流/研报为旁路：超时跳过并在 ``partial_notes`` 说明，仍返回新闻打分主结果。
    旁路有数据时 ``signal_notes`` / ``aux_signals.*.what`` 会说明该信号含义。

    返回内容：
    - ``items``: 逐条新闻（标题/摘要/时间/分数/标签/关键词/指纹）
    - ``aggregate``: 聚合统计（正负比例/均分/样本量/总体标签）
    - ``hot_words``: 高频讨论词 top-15（词 + 出现次数 + 情感权重）
    - ``topic_clusters``: 按正/负/中性分组的代表性标题（各 3 条）
    - ``xueqiu_heat``: 雪球讨论热度（排名/讨论量，如在榜；超时则 skipped）
    - ``eastmoney_trending_keywords``: 东财热搜关联词 top-10
    - ``aux_signals``: 社交 / 资金盘面 / 分析师旁路（含 ``what`` 说明）
    - ``signal_notes``: 本次实际用到的旁路白话提示
    - 审计：model_version / timestamp / text_fingerprint

    Args:
        symbol: 6 位 A 股代码，如 ``"300750"``。
        limit: 分析新闻条数上限（默认 20，上限 50）。多标的时建议 15–20。
    """
    limit = _coerce_limit(limit)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_full_report, symbol, limit),
            timeout=_FULL_REPORT_TIMEOUT_S,
        )
    except TimeoutError:
        logger.warning(
            "get_stock_sentiment_report outer timeout %.0fs; fallback news-only for %s",
            _FULL_REPORT_TIMEOUT_S,
            symbol,
        )
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_news_only_report, symbol, limit),
                timeout=_NEWS_FETCH_TIMEOUT_S + 15.0,
            )
        except Exception:  # noqa: BLE001
            return {
                "error": f"TimeoutError: 舆情报告超过 {_FULL_REPORT_TIMEOUT_S:.0f}s",
                "context": f"get_stock_sentiment_report(symbol={symbol!r}, limit={limit})",
                "symbol": symbol,
            }
    except Exception as e:  # noqa: BLE001
        return _fmt_error(
            e,
            context=f"get_stock_sentiment_report(symbol={symbol!r}, limit={limit})",
        )


if __name__ == "__main__":
    mcp.run(transport="stdio")
