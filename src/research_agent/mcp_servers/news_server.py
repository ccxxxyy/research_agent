"""MCP Server — 通过 ``akshare`` 获取 A 股金融新闻 / 情感数据。

本服务器是金融研究管道的 新闻层。
``fin_data_server`` 提供结构化的行情 / 基本面数据，
``pdf_report_server`` 提供官方披露 PDF，
而本服务器提供及时的 文本 信号：公司新闻、实时市场快讯、热搜话题和宏观摘要。

提供的工具
----------
1. ``get_stock_news`` — 特定 A 股代码的近期新闻文章（东方财富个股新闻流）。  东方财富 / 某只股票的近期新闻（标题+摘要+链接） / "宁德时代最近有什么新闻"
2. ``get_market_telegraph`` — 实时全市场新闻快讯（财联社电报，每隔数分钟刷新）。 财联社 / 全市场实时快讯（短消息） / "今天 A 股有什么大事"
3. ``get_hot_keywords`` — 围绕某个代码的热搜关键词/话题（东方财富热搜词端点），可作为快速情感信号代理。 东方财富 / 某只股票的热搜概念词（如"固态电池"） / "蔚来目前大家在讨论什么"
4. ``get_economic_news`` — 来自百度财经的每日宏观/经济新闻摘要（财经早晚报格式）。 百度财经 / 当日宏观/政策新闻摘要 / "最近有什么宏观政策"
5. ``get_xueqiu_discussion_hot_rank`` — 雪球沪深「讨论」热度排行榜（个股维度）。 雪球 / 讨论最火的股票排行榜 / "雪球上哪些票讨论最多"
     封装 ``akshare.stock_hot_tweet_xq``（来自``stock_feature/stock_hot_xq.py``）。
     返回按 xueqiu.com/hq 上讨论活跃度指标排序的个股 — 不是论坛帖子的标题/正文。

akshare 是一个统一的数据接口。它把各个网站的 API 都封装成了 Python 函数。news_server 通过调 akshare 来间接调各个网站。

akshare（Python 第三方库，统一封装） 和各数据源的关系
  ├── 东方财富 API → 个股新闻、热搜关键词
  ├── 财联社 API   → 实时市场快讯
  ├── 百度财经 API → 宏观经济摘要
  └── 雪球 API     → 讨论热度排行榜

为什么单独成服务器（而不扩展 ``fin_data_server``）？
----------------------------------------------------
``fin_data_server`` 的工具全部返回数值 / 表格数据。
此处的新闻工具返回约 10 倍大小的自由文本负载，具有不同的延迟特征（不需要缓存 — 新闻天然是新鲜的）和不同的失败模式（空新闻流在无事发生时是正常的，而空 K 线是个 bug）。
LLM 对新闻与行情数据也需要不同的提示规则 — 见``agents/specialists.py`` 中的 ``NEWS_EXPERT_PROMPT``。

多源 / 回退策略
----------------
每个工具主要与一个提供方通信；不像行情端点那样级联备份。理由：新闻负载是定性的，因此缺失的数据源最好如实告知 Agent（"提供方 X 宕机，当前无新闻可用"），
而非用回退源的陈旧数据掩盖。Agent 可自行判断用户问题是否必须依赖新闻，还是可以从数据 / 研报层回答。

情感分析是 Agent 的工作，而非工具的
------------------------------------
刻意不提供 ``analyze_sentiment`` 工具。基于关键词的轻量情感分析对金融领域来说过于粗糙（如"毛利率下滑但费用率改善"无法归为单一正/负标签），
而基于 LLM 的情感工具会与 ``news_expert`` Agent 在原始新闻流上已做的工作重复。Agent 阅读新闻内容并在其综合分析中推理情感 — 这使工具接口保持精简，复现性保持高水平。

设计说明
--------
- ``akshare`` 是同步且 I/O 密集的。每个工具用 ``asyncio.to_thread``包装，确保上游慢响应不会阻塞 MCP stdio 事件循环。
- 错误以 ``{"error": "...", "context": "..."}`` 形式返回 —抛出异常会终止 stdio 子进程。
- 所有上游列名保持中文。目标的 LLM 层（DeepSeek / Qwen 等）能流畅阅读中文；翻译为英文会丢失 Agent 可能逐字引用的信息。
- 通过显式 ``limit`` 参数（默认值适中）限制返回行数，确保单个工具响应在 LLM 上下文窗口内。``limit`` 上限为 ``MAX_LIMIT=100``。
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from fastmcp import FastMCP

# akshare 通过 requests 发起 HTTP 请求，requests 会自动读取系统代理。
# 国内数据源（东方财富/财联社/百度/雪球）走代理反而不通，在子进程启动时清除。
for _proxy_key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_proxy_key, None)
os.environ["NO_PROXY"] = "*"

mcp = FastMCP("FinNewsAShare")

# ---------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------
MAX_LIMIT = 100
"""任何工具的 ``limit`` 硬性上限。超过此值 LLM 上下文窗口会开始受影响（单条 ``stock_news_em`` 记录含标题+摘要可达约 500 字符）。
"""

_SHANGHAI_TZ = timezone(timedelta(hours=8))


def _fmt_error(exc: Exception, *, context: str) -> dict[str, Any]:
    """标准错误格式 — LLM 可读，无堆栈跟踪。"""
    return {"error": f"{type(exc).__name__}: {exc}", "context": context}


def _df_to_records(df: pd.DataFrame, *, limit: int | None = None) -> list[dict[str, Any]]:
    """将 DataFrame 转换为 JSON 安全的字典列表。

    与 ``fin_data_server`` 中的辅助函数保持一致的线上格式（中文键名、NaN 为 ``None``、ISO 格式日期）— Agent 可在数据 / 新闻层之间切换而无需学习两种响应约定。
    """
    if limit is not None:
        df = df.head(limit)
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rec: dict[str, Any] = {}
        for col, val in row.items():
            if pd.isna(val):
                rec[str(col)] = None
            elif isinstance(val, (pd.Timestamp, datetime)):
                rec[str(col)] = val.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(val, (int, float, str, bool)):
                rec[str(col)] = val
            else:
                rec[str(col)] = str(val)
        records.append(rec)
    return records


def _today_shanghai_yyyymmdd() -> str:
    """返回亚洲/上海时区今日日期，格式为 ``YYYYMMDD``。

    新闻摘要按上海本地时间发布，因此以上海时间为基准计算"今天"。
    """
    return datetime.now(tz=_SHANGHAI_TZ).strftime("%Y%m%d")


def _coerce_limit(limit: int) -> int:
    """将 ``limit`` 规范化到 ``[1, MAX_LIMIT]`` 范围。"""
    return max(1, min(int(limit), MAX_LIMIT))


# ---------------------------------------------------------------------
# 工具 0（模块文档中列为第五个）: 雪球讨论热度榜（个股）
# ---------------------------------------------------------------------
XUEQIU_DISCUSSION_RANKINGS = frozenset({"最热门", "本周新增"})
"""原样传递给 ``akshare.stock_hot_tweet_xq(symbol=...)``。

* ``最热门`` — 总讨论强度排名（API ``order_by=tweet``）。
* ``本周新增`` — 最近 7 天讨论排名（API ``order_by=tweet7d``）。

akshare 从雪球拉回来的 DataFrame，有一列叫"关注"——但这列实际存的不是关注数，而是讨论量； akshare 将 ``tweet``/``tweet7d`` 映射到其中。
在返回的 JSON 中重命名为 ``讨论量``。
"""


def _xueqiu_discussion_hot_rank(ranking: str, limit: int) -> dict[str, Any]:
    """``get_xueqiu_discussion_hot_rank`` 的同步执行体。"""
    import akshare as ak

    df = ak.stock_hot_tweet_xq(symbol=ranking)
    if df is None or df.empty:
        return {
            "ranking": ranking,
            "count": 0,
            "stocks": [],
            "source": "xueqiu",
            "warning": "stock_hot_tweet_xq 未返回数据",
        }
    # akshare 对 tweet / tweet7d 计数复用了列名 ``关注``。
    if "关注" in df.columns:
        df = df.rename(columns={"关注": "讨论量"})
    return {
        "ranking": ranking,
        "count": min(int(len(df)), limit),
        "stocks": _df_to_records(df, limit=limit),
        "source": "xueqiu",
    }


@mcp.tool()
async def get_xueqiu_discussion_hot_rank(ranking: str = "最热门", limit: int = 30) -> dict:
    """雪球沪深「讨论」热度排行榜 — 个股按讨论活跃度排序。

    对 ``akshare.stock_hot_tweet_xq``（见 ``akshare/stock_feature/stock_hot_xq.py``）的轻量封装。
    每行是一只上市个股（代码 / 简称 / **讨论量** / 最新价），而非用户帖子的标题和链接。
    当用户想了解"雪球上哪些票讨论最火"/"讨论榜"时使用此工具；
    东方财富新闻请用 ``get_stock_news``；财联社快讯请用 ``get_market_telegraph``。

    性能说明： 上游实现会分页遍历完整的筛选结果集 — 首次调用可能耗时数十秒。
    同一 MCP 子进程内的后续调用复用 akshare 内部预热的 ``requests`` 会话，但仍需重新拉取所有分页。

    Args:
        ranking: 仅接受 ``"最热门"``（总讨论量排名）或``"本周新增"``（近 7 天讨论排名）。
        其他值直接返回 ``{"error": ...}``，不发起网络请求。
        limit: 排序后返回的最大股票数（默认 30，上限 ``MAX_LIMIT``=100）。

    Returns:
        包含 ``ranking``、``count``、``stocks``（含中文键名的记录列表，包括 ``讨论量``）、``source``（``"xueqiu"``）的字典。
        失败时返回 ``{"error": ..., "context": ...}``。
    """
    limit = _coerce_limit(limit)
    if ranking not in XUEQIU_DISCUSSION_RANKINGS:
        return _fmt_error(
            ValueError(
                f"ranking 必须是 {sorted(XUEQIU_DISCUSSION_RANKINGS)} 之一，收到 {ranking!r}"
            ),
            context=f"get_xueqiu_discussion_hot_rank(ranking={ranking!r})",
        )
    try:
        return await asyncio.to_thread(_xueqiu_discussion_hot_rank, ranking, limit)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(
            e,
            context=(f"get_xueqiu_discussion_hot_rank(ranking={ranking!r}, limit={limit})"),
        )


# ---------------------------------------------------------------------
# 工具 1: 个股新闻（东方财富）
# ---------------------------------------------------------------------
def _stock_news_em(symbol: str, limit: int) -> dict[str, Any]:
    import akshare as ak

    df = ak.stock_news_em(symbol=symbol)
    if df is None or df.empty:
        return {
            "symbol": symbol,
            "count": 0,
            "news": [],
            "source": "eastmoney",
            "warning": "该代码近期无新闻",
        }
    return {
        "symbol": symbol,
        "count": min(int(len(df)), limit),
        "news": _df_to_records(df, limit=limit),
        "source": "eastmoney",
    }


@mcp.tool()
async def get_stock_news(symbol: str, limit: int = 20) -> dict:
    """获取特定 A 股代码的近期新闻文章。

    由东方财富个股新闻流支撑。每行通常包含：``关键词``（搜索的代码）、 ``新闻标题``、``新闻内容``（简短摘要）、``发布时间``、``文章来源``、 ``新闻链接``。

    Args:
        symbol: 6 位代码，如 ``"300750"`` 代表宁德时代。请勿包含 ``sh`` 或 ``sz`` 等交易所前缀。
        limit: 最大返回新闻行数（默认 20，上限 ``MAX_LIMIT``=100）。大多数代码在新闻流中有 50-200 条；东方财富的新闻 API 一次可能返回 200 条新闻。akshare 内部会自动分页把所有新闻都拉回来。然后再用 limit 参数从这 200 条里截取前 20 条返回。

    Returns:
        包含 ``symbol``、``count``、``news``（新闻记录列表）、``source``（始终为 ``"eastmoney"``）的字典。
        失败时返回``{"error": ..., "context": ...}``。
    """
    limit = _coerce_limit(limit)
    try:
        return await asyncio.to_thread(_stock_news_em, symbol, limit)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(
            e,
            context=f"get_stock_news(symbol={symbol!r}, limit={limit})",
        )


# ---------------------------------------------------------------------
# 工具 2: 实时市场电报（财联社）
# ---------------------------------------------------------------------
TELEGRAPH_CATEGORIES = {"全部", "重点"}
"""``get_market_telegraph(category=...)`` 的允许值。

以前（1.18 之前）有一个函数 stock_telegraph_cls，支持按"A股"、"宏观"等类别过滤。但 1.18 版本之后这个函数被废弃了，新函数 stock_info_global_cls 只支持"全部"和"重点"两个选项。
如果 LLM 生成了 get_market_telegraph(category="A股")（用了不支持的类别），立刻返回错误告诉 LLM "只能用全部或重点"。如果不做这个检查，akshare 会返回一个空的 DataFrame，LLM 就会以为"今天没有新闻"——但实际上是参数写错了。快速失败会比静默返回空数据好，因为 LLM 能看到错误信息并修正调用。
上游 ``akshare.stock_info_global_cls`` 端点仅支持两个过滤器 — ``全部``（全量）和 ``重点``（标记为重要的）。
旧版 akshare 曾以``stock_telegraph_cls`` 名称暴露更丰富的分类集，但在 1.18+ 中已废弃；
在此显式约束，使 LLM 生成的 ``A股`` / ``宏观`` 调用能快速失败并给出有用错误，而非静默返回空数据帧。
"""


def _telegraph_cls(symbol: str, limit: int) -> dict[str, Any]:
    import akshare as ak

    df = ak.stock_info_global_cls(symbol=symbol)
    if df is None or df.empty:
        return {
            "category": symbol,
            "count": 0,
            "telegraph": [],
            "source": "cls",
            "warning": f"类别 {symbol!r} 近期无快讯",
        }
    return {
        "category": symbol,
        "count": min(int(len(df)), limit),
        "telegraph": _df_to_records(df, limit=limit),
        "source": "cls",
    }


@mcp.tool()
async def get_market_telegraph(category: str = "全部", limit: int = 30) -> dict:
    """从财联社获取实时市场新闻快讯。

    财联社是中国市场版的彭博"FIRST WORD"终端 — 关于市场动向事件的短时间戳快讯（每条约 50-300 字符）。交易时段内每隔数分钟更新。

    Args:
        category: 上游数据流的过滤器。调用的 akshare 端点仅支持
            两个值：
              - ``"全部"``（默认） — 所有快讯（全量）
              - ``"重点"``         — 仅标记为重要的
            其他值返回 ``{"error": ...}``。
        limit: 最大返回快讯数（默认 30，上限 ``MAX_LIMIT``=100）。
            超出 ``limit`` 的旧条目被静默丢弃。

    Returns:
        包含 ``category``、``count``、``telegraph``（快讯记录列表，每条通常含 ``标题``、``内容``、``发布日期``、``发布时间``）和
        ``source``（始终为 ``"cls"``）的字典。
        失败时返回``{"error": ..., "context": ...}``。
    """
    limit = _coerce_limit(limit)
    if category not in TELEGRAPH_CATEGORIES:
        return _fmt_error(
            ValueError(f"category 必须是 {sorted(TELEGRAPH_CATEGORIES)} 之一，收到 {category!r}"),
            context=f"get_market_telegraph(category={category!r})",
        )
    try:
        return await asyncio.to_thread(_telegraph_cls, category, limit)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(
            e,
            context=f"get_market_telegraph(category={category!r}, limit={limit})",
        )


# ---------------------------------------------------------------------
# 工具 3: 热搜关键词 / 热门话题（东方财富）
# ---------------------------------------------------------------------
def _hot_keywords_em(symbol: str, limit: int) -> dict[str, Any]:
    import akshare as ak

    df = ak.stock_hot_keyword_em(symbol=symbol)
    if df is None or df.empty:
        return {
            "symbol": symbol,
            "count": 0,
            "keywords": [],
            "source": "eastmoney",
            "warning": "该代码无热搜关键词",
        }
    return {
        "symbol": symbol,
        "count": min(int(len(df)), limit),
        "keywords": _df_to_records(df, limit=limit),
        "source": "eastmoney",
    }


@mcp.tool()
async def get_hot_keywords(symbol: str, limit: int = 10) -> dict:
    """获取某个 A 股代码周围的热搜关键词 / 话题。

    由东方财富的 stock_hot_keyword 端点支撑。关键词列表是快速的情感 / 讨论话题代理：
    当前有哪些主题（如 ``"碳中和"``、``"业绩预增"``、 ``"高管减持"``）正与该代码在散户论坛和分析师动态中共现。

    使用场景
    --------
    - "蔚来目前在讨论什么？" → 先调用此工具，然后对突出的关键词用 ``get_stock_news`` 深入查看。
    - "中芯国际的芯片短缺叙事是否降温？" → 比较跨时段的关键词频率（需要多次调用）。

    Args:
        symbol: 带交易所前缀的代码。
            与本服务器其他工具不同，``stock_hot_keyword_em`` 需要 ``SH``/``SZ`` 前缀的大写形式，如 ``"SZ300750"``。
            在此进行规范化，调用者仍可传入纯 6 位代码 — 前缀会自动添加。
        limit: 最大关键词行数（默认 10，上限 ``MAX_LIMIT``=100）。 每行通常包含 ``时间``、``概念名称``、``概念代码``、``热度``。

    Returns:
        包含 ``symbol``、``count``、``keywords`` 和 ``source`` 的字典。
        失败时返回 ``{"error": ..., "context": ...}``。
    """
    limit = _coerce_limit(limit)
    bare = symbol.strip().upper()
    if bare.startswith(("SH", "SZ")):
        prefixed = bare
    else:
        prefix = "SH" if bare.startswith("6") else "SZ"
        prefixed = f"{prefix}{bare}"
    try:
        return await asyncio.to_thread(_hot_keywords_em, prefixed, limit)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(
            e,
            context=f"get_hot_keywords(symbol={symbol!r}, limit={limit})",
        )


# ---------------------------------------------------------------------
# 工具 4: 经济新闻摘要（百度财经早晚报）
# ---------------------------------------------------------------------
def _economic_news_baidu(date: str, limit: int) -> dict[str, Any]:
    import akshare as ak

    df = ak.news_economic_baidu(date=date)
    if df is None or df.empty:
        return {
            "date": date,
            "count": 0,
            "news": [],
            "source": "baidu",
            "warning": f"{date} 无经济新闻摘要",
        }
    return {
        "date": date,
        "count": min(int(len(df)), limit),
        "news": _df_to_records(df, limit=limit),
        "source": "baidu",
    }


@mcp.tool()
async def get_economic_news(date: str = "", limit: int = 30) -> dict:
    """获取每日宏观 / 经济新闻摘要（百度财经早晚报）。

    早晚报格式是百度财经每日两次发布的精选摘要，涵盖宏观政策、央行、 GDP、CPI、汇率和大公司公告。
    与 ``get_market_telegraph`` 相比，实时性较低但编辑性更强（每条都是人工挑选而非全量推送）。

    Args:
        date: ``YYYYMMDD`` 字符串，如 ``"20260508"``。留空（默认）→今天（亚洲/上海）。最近约 30 天的数据可靠可用；更早日期可能返回空。
        limit: 最大新闻行数（默认 30，上限 ``MAX_LIMIT``=100）。

    Returns:
        包含 ``date``（实际查询的日期）、``count``、``news``（每条通常含 ``发布日期``、``发布时间``、``内容``）和 ``source``（``"baidu"``）的字典。
        失败时返回 ``{"error": ..., "context": ...}``。
    """
    limit = _coerce_limit(limit)
    use_date = date.strip() or _today_shanghai_yyyymmdd()
    if not use_date.isdigit() or len(use_date) != 8:
        return _fmt_error(
            ValueError(
                f"date 必须是 YYYYMMDD 格式（8 位数字），收到 {date!r}；传入空字符串表示今天"
            ),
            context=f"get_economic_news(date={date!r})",
        )
    try:
        return await asyncio.to_thread(_economic_news_baidu, use_date, limit)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(
            e,
            context=f"get_economic_news(date={use_date!r}, limit={limit})",
        )


if __name__ == "__main__":
    mcp.run(transport="stdio")
