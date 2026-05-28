"""MCP Server — 通过 ``akshare`` 获取中国 A 股金融数据。

本服务器是金融研究管道的据层。它暴露由 ``akshare`` 支撑的真实、免费、无需 API 密钥的金融端点，
``akshare`` 进而聚合来自 东方财富 / 新浪财经 / 巨潮资讯 的数据。

提供的工具
----------
1. ``get_stock_basic_info`` — 公司概况（行业、市值、上市日期、最新价）。
2. ``get_stock_price_history`` — 日级 OHLCV 及简单汇总统计。
3. ``get_financial_abstract`` — 按报告期的营收 / 净利润 / 现金流 / EPS。
4. ``get_financial_indicators`` — 按报告期的 ROE / ROA / 利润率 / 杠杆比率。
5. ``search_stock_by_name`` — 模糊匹配公司名称到 A 股代码（使用一次性内存缓存，避免每次调用都请求全市场名单）。

设计说明
--------
- ``akshare`` 是同步且 I/O 密集的。每个工具用 ``asyncio.to_thread``包装，确保单个慢请求不会阻塞 MCP stdio 事件循环中的其他请求。
- ``akshare`` 在上游 HTML / JSON 结构变化时偶尔抛出 ``KeyError`` / ``AttributeError`` / ``ValueError``，网络故障则表现为 ``ConnectionError``
  / ``ProxyError``。在工具边界捕获 ``Exception``，因为抛出异常的 MCP 工具会导致子进程崩溃；取而代之返回结构化的 ``{"error": "..."}`` 负载，LLM 可据此推理。
- 多源容灾：位于 ``push2*.eastmoney.com`` 的两个端点（实时报价 +日K线）出了名地不稳定——它们会被 Windows 注册表代理探测阻断、被限流、偶尔从中国境外返回 451。
  对于这两个工具，级联备选提供方（雪球、新浪），使用户获得可用结果而非 ProxyError。
  每个响应携带 ``source`` 字段，让调用者知道实际由哪个提供方响应。
- 所有股票代码必须是 6 位数字（如宁德时代为 ``300750``）。不接受带交易所前缀的形式（``sh300750``）。内部按各上游需求添加/去除前缀。
- 列名保持中文，因为 ``akshare`` 就是这样返回的，且下游 LLM（DeepSeek / Qwen）能流畅阅读中文。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from fastmcp import FastMCP

mcp = FastMCP("FinDataAShare")


_ALL_STOCKS_CACHE: pd.DataFrame | None = None
"""``ak.stock_info_a_code_name()`` 的模块级缓存。

该调用耗时约 6 秒，因为它抓取了完整的 A 股名单。只需在每个 MCP子进程生命周期内支付一次该开销；后续的 ``search_stock_by_name`` 调用是纯 pandas 过滤操作。
"""


def _fmt_error(exc: Exception, *, context: str) -> dict[str, Any]:
    """标准错误格式 — LLM 可读，无堆栈跟踪。"""
    return {"error": f"{type(exc).__name__}: {exc}", "context": context}


def _exchange_prefix(symbol: str, *, upper: bool = False) -> str:
    """返回交易所代码 — 6 开头为 ``sh``，其他为 ``sz``。

    ``akshare`` 不一致：雪球需要大写（``SZ300750``），新浪/腾讯需要小写（``sz300750``），东财使用数字市场代码。此处仅构建字符串前缀形式。
    """
    prefix = "sh" if symbol.startswith("6") else "sz"
    return prefix.upper() if upper else prefix


def _prefixed_symbol(symbol: str, *, upper: bool = False) -> str:
    """返回 ``sh300750`` / ``SH300750`` / ``sz300750`` / ``SZ300750``。"""
    return f"{_exchange_prefix(symbol, upper=upper)}{symbol}"


def _df_to_records(df: pd.DataFrame, *, limit: int | None = None) -> list[dict[str, Any]]:
    """将 DataFrame 转换为 JSON 安全的字典列表。"""
    if limit is not None:
        df = df.head(limit)
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rec: dict[str, Any] = {}
        for col, val in row.items():
            if pd.isna(val):
                rec[str(col)] = None
            elif isinstance(val, (pd.Timestamp, datetime)):
                rec[str(col)] = val.strftime("%Y-%m-%d")
            elif isinstance(val, (int, float, str, bool)):
                rec[str(col)] = val
            else:
                rec[str(col)] = str(val)
        records.append(rec)
    return records


# ---------------------------------------------------------------------
# 工具 1: 股票基本信息（公司概况）— 多源
# ---------------------------------------------------------------------
def _basic_info_from_eastmoney(symbol: str) -> dict[str, Any]:
    import akshare as ak
    df = ak.stock_individual_info_em(symbol=symbol)
    info = dict(zip(df["item"].astype(str), df["value"].tolist(), strict=False))
    return {"symbol": symbol, "info": info, "source": "eastmoney"}


def _basic_info_from_xueqiu(symbol: str) -> dict[str, Any]:
    import akshare as ak
    df = ak.stock_individual_basic_info_xq(symbol=_prefixed_symbol(symbol, upper=True))
    info = dict(zip(df["item"].astype(str), df["value"].astype(str).tolist(), strict=False))
    return {"symbol": symbol, "info": info, "source": "xueqiu"}


@mcp.tool()
async def get_stock_basic_info(symbol: str) -> dict:
    """返回 A 股代码的公司概况。

    优先尝试东方财富（行业/市值/流通股数据），若主端点不可达则回退到雪球（返回更丰富的 39 字段概况，含英文名、注册信息和经营范围）。

    典型字段包括：最新价、总股本、流通股、总市值、流通市值、行业、上市时间、股票简称、股票代码（eastmoney）或 org_name_cn、org_short_name_cn、
    established_date、main_business、reg_asset、listed_date（xueqiu）。

    Args:
        symbol: 6 位代码，如 ``"300750"`` 代表宁德时代。勿包含``sh`` 或 ``sz`` 等交易所前缀。

    Returns:
        包含 ``symbol``、``info`` 和 ``source`` 的字典 — ``source`` 为``"eastmoney"`` 或 ``"xueqiu"``，取决于哪个提供方响应了请求。
        若两个源都失败，返回 ``{"error": ...}``。
    """
    errors: list[str] = []
    for label, fn in (
        ("eastmoney", _basic_info_from_eastmoney),
        ("xueqiu", _basic_info_from_xueqiu),
    ):
        try:
            return await asyncio.to_thread(fn, symbol)
        except Exception as e:
            errors.append(f"{label}: {type(e).__name__}: {str(e)[:100]}")
    return {
        "error": "all providers failed",
        "context": f"get_stock_basic_info(symbol={symbol!r})",
        "attempts": errors,
    }


# ---------------------------------------------------------------------
# 工具 2: 价格历史及汇总统计 — 多源
# ---------------------------------------------------------------------
def _summarize_bars(
    df: pd.DataFrame,
    *,
    date_col: str,
    close_col: str,
    high_col: str,
    low_col: str,
) -> dict[str, Any]:
    """不同提供方数据结构的通用汇总统计构建器。"""
    first_close = float(df[close_col].iloc[0])
    last_close = float(df[close_col].iloc[-1])
    pct_change = (last_close - first_close) / first_close * 100 if first_close else 0.0
    return {
        "period_start": str(df[date_col].iloc[0]),
        "period_end": str(df[date_col].iloc[-1]),
        "sessions": int(len(df)),
        "high": float(df[high_col].max()),
        "low": float(df[low_col].min()),
        "pct_change": round(pct_change, 2),
    }


def _price_history_from_eastmoney(symbol: str, days: int, adjust: str) -> dict[str, Any]:
    import akshare as ak
    end = datetime.now()
    start = end - timedelta(days=days)
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        adjust=adjust,
    )
    if df is None or df.empty:
        return {"symbol": symbol, "bars": [], "summary": {"sessions": 0}, "source": "eastmoney"}
    bars = _df_to_records(df)
    summary = _summarize_bars(
        df, date_col="日期", close_col="收盘", high_col="最高", low_col="最低"
    )
    return {"symbol": symbol, "bars": bars, "summary": summary, "source": "eastmoney"}


def _price_history_from_sina(symbol: str, days: int, adjust: str) -> dict[str, Any]:
    import akshare as ak
    end = datetime.now()
    start = end - timedelta(days=days)
    df = ak.stock_zh_a_daily(
        symbol=_prefixed_symbol(symbol),
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        adjust=adjust,
    )
    if df is None or df.empty:
        return {"symbol": symbol, "bars": [], "summary": {"sessions": 0}, "source": "sina"}
    bars = _df_to_records(df)
    summary = _summarize_bars(
        df, date_col="date", close_col="close", high_col="high", low_col="low"
    )
    return {"symbol": symbol, "bars": bars, "summary": summary, "source": "sina"}


@mcp.tool()
async def get_stock_price_history(
    symbol: str,
    days: int = 30,
    adjust: str = "qfq",
) -> dict:
    """返回最近 ``days`` 个交易日的日级 OHLCV K线。

    优先尝试东方财富（最丰富的字段：成交量/成交额/振幅/换手率），
    回退到新浪（较简字段：date/open/high/low/close/volume/amount）。
    周末和市场假期由两个提供方自动跳过 — ``days`` 是日历窗口，因此 30 个日历日约产生 20 个交易日。

    Args:
        symbol: 6 位代码，如 ``"300750"``。
        days: 回看窗口（日历天数，默认 30，最大 365）。
        adjust: 复权模式。``"qfq"`` = 前复权（推荐用于收益分析），``"hfq"`` = 后复权，``""`` = 不复权。

    Returns:
        包含 ``symbol``、``bars``（日级记录列表 — 东方财富为中文键名， 新浪为英文键名）、
        ``summary``（含 ``{period_start, period_end, sessions, high, low, pct_change}``）以及 ``source``（指示哪个提供方响应）的字典。
    """
    if days < 1 or days > 365:
        return _fmt_error(
            ValueError(f"days must be in [1, 365], got {days}"),
            context=f"get_stock_price_history(symbol={symbol!r}, days={days})",
        )

    errors: list[str] = []
    for label, fn in (
        ("eastmoney", _price_history_from_eastmoney),
        ("sina", _price_history_from_sina),
    ):
        try:
            return await asyncio.to_thread(fn, symbol, days, adjust)
        except Exception as e:
            errors.append(f"{label}: {type(e).__name__}: {str(e)[:100]}")
    return {
        "error": "all providers failed",
        "context": f"get_stock_price_history(symbol={symbol!r}, days={days}, adjust={adjust!r})",
        "attempts": errors,
    }


# ---------------------------------------------------------------------
# 工具 3: 财务摘要（核心三表摘要）
# ---------------------------------------------------------------------
_ABSTRACT_KEY_METRICS: tuple[str, ...] = (
    "归母净利润",
    "营业总收入",
    "营业成本",
    "净利润",
    "扣非净利润",
    "股东权益合计(净资产)",
    "商誉",
    "经营现金流量净额",
    "基本每股收益",
    "每股净资产",
)
"""从 ``stock_financial_abstract`` 中呈现的行项白名单。

``akshare`` 返回约 50 行覆盖所有明细项；对于研报用例来说 90% 是噪音。仅呈现分析师会实际引用的行项。
"""


@mcp.tool()
async def get_financial_abstract(symbol: str, last_n_periods: int = 4) -> dict:
    """返回最近报告期的核心财务报表项目。

    涵盖分析师在研报中引用的项目：营收、净利润、经营现金流、EPS及若干资产负债表锚定指标。每列为一个报告期（季度或年度）。

    Args:
        symbol: 6 位代码。
        last_n_periods: 包含的最近报告期数量（默认 4 ≈ 一年的季报，最大 12）。

    Returns:
        包含 ``symbol``、``periods``（报告期代码列表，如 ``"20241231"``）
        和 ``metrics``（字典，形如``{metric_name: [value_period_1, value_period_2, ...]}``）的字典。
    """
    if last_n_periods < 1 or last_n_periods > 12:
        return _fmt_error(
            ValueError(f"last_n_periods must be in [1, 12], got {last_n_periods}"),
            context=f"get_financial_abstract(symbol={symbol!r})",
        )

    def _call() -> dict[str, Any]:
        import akshare as ak
        df = ak.stock_financial_abstract(symbol=symbol)
        if df is None or df.empty:
            return {"symbol": symbol, "periods": [], "metrics": {}}

        period_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 8]
        period_cols.sort(reverse=True)  # newest first
        period_cols = period_cols[:last_n_periods]

        metrics: dict[str, list[Any]] = {}
        indicator_col = "指标" if "指标" in df.columns else df.columns[1]

        for metric in _ABSTRACT_KEY_METRICS:
            mask = df[indicator_col].astype(str).str.contains(
                metric, na=False, regex=False
            )
            if not mask.any():
                continue
            row = df[mask].iloc[0]
            values: list[Any] = []
            for pc in period_cols:
                val = row.get(pc)
                if pd.isna(val):
                    values.append(None)
                else:
                    try:
                        values.append(float(val))
                    except (TypeError, ValueError):
                        values.append(str(val))
            metrics[metric] = values

        return {
            "symbol": symbol,
            "periods": period_cols,
            "metrics": metrics,
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(
            e,
            context=f"get_financial_abstract(symbol={symbol!r}, last_n_periods={last_n_periods})",
        )


# ---------------------------------------------------------------------
# 工具 4: 财务比率（ROE/ROA/利润率/杠杆）
# ---------------------------------------------------------------------
_RATIO_KEY_METRICS: tuple[str, ...] = (
    "净资产收益率",
    "总资产净利润率",
    "销售毛利率",
    "销售净利率",
    "资产负债率",
    "流动比率",
    "速动比率",
    "存货周转率",
    "应收账款周转率",
    "营业利润率",
)


@mcp.tool()
async def get_financial_indicators(symbol: str, start_year: str = "2023") -> dict:
    """返回核心财务比率（ROE、ROA、利润率、杠杆）。

    Args:
        symbol: 6 位代码。
        start_year: 4 位年份字符串，如 ``"2023"``。akshare 返回从该年起的所有报告期。

    Returns:
        包含 ``symbol``、``periods``（报告日期列表，如 ``"2024-09-30"``）
        和 ``ratios``（字典，形如``{ratio_name: [value_period_1, value_period_2, ...]}``）的字典。
        上游源留空的位置值为 ``None``。
    """
    if len(start_year) != 4 or not start_year.isdigit():
        return _fmt_error(
            ValueError(f"start_year must be a 4-digit year, got {start_year!r}"),
            context=f"get_financial_indicators(symbol={symbol!r})",
        )

    def _call() -> dict[str, Any]:
        import akshare as ak
        df = ak.stock_financial_analysis_indicator(symbol=symbol, start_year=start_year)
        if df is None or df.empty:
            return {"symbol": symbol, "periods": [], "ratios": {}}

        date_col = "日期" if "日期" in df.columns else df.columns[0]
        df = df.sort_values(date_col, ascending=False)
        periods = [str(d) for d in df[date_col].tolist()]

        ratios: dict[str, list[Any]] = {}
        for metric in _RATIO_KEY_METRICS:
            matched_cols = [c for c in df.columns if metric in str(c)]
            if not matched_cols:
                continue
            col = matched_cols[0]
            values: list[Any] = []
            for val in df[col].tolist():
                if pd.isna(val):
                    values.append(None)
                else:
                    try:
                        values.append(round(float(val), 4))
                    except (TypeError, ValueError):
                        values.append(str(val))
            ratios[str(col)] = values

        return {"symbol": symbol, "periods": periods, "ratios": ratios}

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(
            e,
            context=f"get_financial_indicators(symbol={symbol!r}, start_year={start_year!r})",
        )


# ---------------------------------------------------------------------
# 工具 5: 按公司名称模糊搜索股票
# ---------------------------------------------------------------------
@mcp.tool()
async def search_stock_by_name(keyword: str, limit: int = 10) -> dict:
    """模糊匹配公司名称到 A 股代码。

    首次调用会预热一个涵盖全部 A 股名单的内存缓存（约 5k 个代码，一次性耗时约 6 秒）。后续调用为亚毫秒级的 pandas 过滤。

    Args:
        keyword: 部分公司名称，如 ``"宁德"`` 或 ``"平安"``。
        limit: 最大返回匹配数（默认 10，上限 50）。

    Returns:
        包含 ``keyword`` 和 ``matches``（列表，形如``{"code": "300750", "name": "宁德时代"}``）的字典，
        或失败时返回 ``{"error": ...}``。
    """
    if not keyword.strip():
        return _fmt_error(
            ValueError("keyword must be non-empty"),
            context="search_stock_by_name()",
        )
    limit = max(1, min(limit, 50))

    def _call() -> dict[str, Any]:
        global _ALL_STOCKS_CACHE
        if _ALL_STOCKS_CACHE is None:
            import akshare as ak
            _ALL_STOCKS_CACHE = ak.stock_info_a_code_name()

        df = _ALL_STOCKS_CACHE
        if "name" not in df.columns or "code" not in df.columns:
            raise RuntimeError(
                f"unexpected schema from stock_info_a_code_name: {list(df.columns)}"
            )
        mask = df["name"].astype(str).str.contains(keyword, na=False, regex=False)
        hits = df[mask].head(limit)
        matches = [
            {"code": str(r["code"]), "name": str(r["name"])}
            for _, r in hits.iterrows()
        ]
        return {"keyword": keyword, "matches": matches}

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(
            e,
            context=f"search_stock_by_name(keyword={keyword!r}, limit={limit})",
        )


if __name__ == "__main__":
    mcp.run(transport="stdio")
