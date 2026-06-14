"""MCP Server — 通过 ``akshare`` 获取中国 A 股金融数据。

本服务器是金融研究管道的据层。它暴露由 ``akshare`` 支撑的真实、免费、无需 API 密钥的金融端点，
``akshare`` 进而聚合来自 东方财富 / 新浪财经 / 巨潮资讯 的数据。

提供的工具
----------
1. ``get_stock_basic_info`` — 公司概况（行业、市值、上市日期、最新价）。
2. ``get_stock_price_history`` — 日级 OHLCV 及简单汇总统计。
3. ``get_financial_abstract`` — 按报告期的营收 / 净利润 / 现金流 / EPS。
4. ``get_financial_indicators`` — 按报告期的 ROE / ROA / 利润率 / 杠杆比率。
5. ``search_stock_by_name`` — 模糊匹配公司名称到 A 股代码。
6. ``get_index_quotes`` — A 股主要指数实时行情。
7. ``get_sector_fund_flow`` — 板块资金流向排行。
8. ``get_stock_rank`` — 今日涨跌幅排行。
9. ``get_intraday`` — 分时 K 线（1/5/15/30/60 分钟）。
10. ``get_lhb_detail`` — 龙虎榜详情。
11. ``get_margin_detail`` — 个股融资融券数据。
12. ``get_top_holders`` — 十大流通股东。
13. ``get_etf_spot`` — ETF 基金实时行情排行。
14. ``get_macro_china`` — 宏观经济指标（GDP/CPI/PMI/M2/社融）。
15. ``get_concept_board`` — 概念板块行情 + 成分股。
16. ``get_industry_board`` — 行业板块行情 + 成分股。
17. ``get_individual_fund_flow`` — 个股资金流向。
18. ``get_hsgt_flow`` — 沪深港通资金流向。
19. ``get_market_status`` — 市场交易状态（开盘/收盘/午休/非交易日）。

设计说明
--------
- ``akshare`` 是同步且 I/O 密集的。每个工具用 ``asyncio.to_thread``包装，确保单个慢请求不会阻塞 MCP stdio 事件循环中的其他请求。
- ``akshare`` 在上游 HTML / JSON 结构变化时偶尔抛出 ``KeyError`` / ``AttributeError`` / ``ValueError``，网络故障则表现为 ``ConnectionError``
  / ``ProxyError``。在工具边界捕获 ``Exception``，因为抛出异常的 MCP 工具会导致子进程崩溃；取而代之返回结构化的 ``{"error": "..."}`` 负载，LLM 可据此推理。
- 多源容灾：``push2*.eastmoney.com`` 端点不稳定（TLS 指纹检测、代理阻断、限流）。
  所有工具采用"先尝试再说"策略——直接调用 akshare，如果失败则在工具边界捕获异常返回 error dict，
  而不是预先检测 push2 连通性后拒绝服务。涨跌榜使用新浪/腾讯双层兜底，指数行情使用腾讯兜底。
  每个响应携带 ``source`` 字段，让调用者知道实际由哪个提供方响应。
- 所有股票代码必须是 6 位数字（如宁德时代为 ``300750``）。不接受带交易所前缀的形式（``sh300750``）。内部按各上游需求添加/去除前缀。
- 列名保持中文，因为 ``akshare`` 就是这样返回的，且下游 LLM（DeepSeek / Qwen）能流畅阅读中文。
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from fastmcp import FastMCP

logger = logging.getLogger("fin_data_server")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# akshare 通过 requests 发起 HTTP 请求，requests 会自动读取系统代理。
# 国内数据源（东方财富/新浪/雪球）走代理反而不通，在子进程启动时清除。
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

# requests 在 Windows 上会从注册表读取系统代理（即使环境变量已清除），
# 导致 push2 请求被代理转发后断开。必须禁用 trust_env。
import requests as _requests  # noqa: E402
import urllib3 as _urllib3  # noqa: E402

_urllib3.disable_warnings(_urllib3.exceptions.InsecureRequestWarning)

_orig_session_init = _requests.Session.__init__


def _patched_session_init(self: _requests.Session, *args, **kwargs):  # type: ignore[no-untyped-def]
    _orig_session_init(self, *args, **kwargs)
    self.trust_env = False
    self.verify = False


_requests.Session.__init__ = _patched_session_init  # type: ignore[method-assign]

mcp = FastMCP("FinDataAShare")

# ---------------------------------------------------------------------------
# push2 连通性检测 + curl_cffi 直连助手
# ---------------------------------------------------------------------------
import time as _time  # noqa: E402

try:
    from curl_cffi import requests as _curl_requests  # noqa: E402

    _HAS_CURL_CFFI = True
except ImportError:
    _HAS_CURL_CFFI = False


def _curl_get_json(url: str, *, timeout: int = 10) -> dict | None:
    """用 curl_cffi（Chrome TLS 指纹）请求 JSON，返回 dict 或 None。"""
    if not _HAS_CURL_CFFI:
        return None
    try:
        resp = _curl_requests.get(url, impersonate="chrome", timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.debug("curl_cffi 请求失败 (%s): %s", url, e)
    return None


def _probe_push2_connectivity() -> bool:
    """探测 push2 实时端点是否可达（先 curl_cffi 再 requests）。"""
    url = "https://88.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1&fs=b:MK0021&fields=f12"
    data = _curl_get_json(url, timeout=8)
    if data and data.get("data"):
        logger.info("push2 实时端点探测成功 (curl_cffi)")
        return True
    try:
        resp = _requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200 and resp.text:
            logger.info("push2 实时端点探测成功 (requests)")
            return True
    except Exception:
        pass
    return False


_PUSH2_AVAILABLE: bool | None = None
_PROBE_TS: float = 0.0
_PROBE_TTL: float = 300.0


def _is_push2_available() -> bool:
    """检测 push2 实时端点连通性（带 5 分钟 TTL 缓存）。"""
    global _PUSH2_AVAILABLE, _PROBE_TS  # noqa: PLW0603
    now = _time.time()
    if _PUSH2_AVAILABLE is None or (now - _PROBE_TS > _PROBE_TTL):
        _PROBE_TS = now
        _PUSH2_AVAILABLE = _probe_push2_connectivity()
        if not _PUSH2_AVAILABLE:
            logger.warning(
                "push2.eastmoney.com 不可达——实时行情/分时/板块/资金流等工具将降级。"
                "历史 K 线将走新浪（sina）源；财务数据不受影响。"
            )
        else:
            logger.info("push2.eastmoney.com 连通性正常")
    return _PUSH2_AVAILABLE


def _fetch_stock_kline_via_curl(
    symbol: str, period: str = "daily", adjust: str = "qfq", limit: int = 120
) -> pd.DataFrame | None:
    """通过 curl_cffi 直连 push2his 获取股票/ETF K 线，绕过 TLS 指纹检测。"""
    if not _HAS_CURL_CFFI:
        return None

    period_map = {"daily": "101", "weekly": "102", "monthly": "103"}
    adjust_map = {"qfq": "1", "hfq": "2", "": "0"}
    market_id = 1 if symbol.startswith(("5", "6")) else 0

    url = (
        f"https://push2his.eastmoney.com/api/qt/stock/kline/get?"
        f"secid={market_id}.{symbol}&fields1=f1,f2,f3,f4,f5,f6"
        f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
        f"&klt={period_map.get(period, '101')}"
        f"&fqt={adjust_map.get(adjust, '1')}"
        f"&end=20500101&lmt={limit}"
        f"&ut=7eea3edcaed734bea9cbfc24409ed989"
    )
    data = _curl_get_json(url, timeout=12)
    if not data or not data.get("data") or not data["data"].get("klines"):
        return None

    rows = [row.split(",") for row in data["data"]["klines"]]
    cols = [
        "日期",
        "开盘",
        "收盘",
        "最高",
        "最低",
        "成交量",
        "成交额",
        "振幅",
        "涨跌幅",
        "涨跌额",
        "换手率",
    ]
    df = pd.DataFrame(rows, columns=cols[: len(rows[0])])
    for c in df.columns:
        if c != "日期":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _fetch_tencent_realtime(codes: list[str]) -> pd.DataFrame | None:
    """通过腾讯行情 qt.gtimg.cn 批量获取实时行情，绕过东方财富 TLS 限制。"""
    if not _HAS_CURL_CFFI or not codes:
        return None
    tencent_codes = []
    for c in codes:
        if c.startswith(("5", "6")):
            tencent_codes.append(f"sh{c}")
        else:
            tencent_codes.append(f"sz{c}")

    all_records: list[dict[str, Any]] = []
    batch_size = 80
    for i in range(0, len(tencent_codes), batch_size):
        batch = tencent_codes[i : i + batch_size]
        url = f"https://qt.gtimg.cn/q={','.join(batch)}"
        try:
            resp = _curl_requests.get(url, impersonate="chrome", timeout=10)
            for line in resp.text.strip().split("\n"):
                line = line.strip().rstrip(";")
                if not line or "=" not in line:
                    continue
                parts = line.split("~")
                if len(parts) < 45:
                    continue
                try:
                    rec: dict[str, Any] = {
                        "代码": parts[2],
                        "名称": parts[1],
                        "最新价": float(parts[3]) if parts[3] else None,
                        "涨跌幅": float(parts[32]) if parts[32] else None,
                        "涨跌额": float(parts[31]) if parts[31] else None,
                        "成交额": float(parts[37]) * 10000 if parts[37] else None,
                        "换手率": float(parts[38]) if parts[38] else None,
                    }
                    if rec["最新价"] and rec["最新价"] > 0:
                        all_records.append(rec)
                except (ValueError, IndexError):
                    continue
        except Exception as e:
            logger.debug("腾讯行情批次查询失败: %s", e)
            continue

    if not all_records:
        return None
    return pd.DataFrame(all_records)


def _fetch_realtime_quotes_via_curl(fs: str, *, limit: int = 50) -> pd.DataFrame | None:
    """通过 curl_cffi 直连 push2 获取实时排行（涨跌幅/资金流等）。"""
    if not _HAS_CURL_CFFI:
        return None

    url = (
        f"https://88.push2.eastmoney.com/api/qt/clist/get?pn=1&pz={limit}&po=1&np=1"
        f"&fltt=2&invt=2&fid=f3&fs={fs}"
        f"&fields=f2,f3,f4,f5,f6,f7,f12,f14,f15,f16,f17,f18,f20,f21"
    )
    data = _curl_get_json(url, timeout=10)
    if not data or not data.get("data") or not data["data"].get("diff"):
        return None

    col_map = {
        "f12": "代码",
        "f14": "名称",
        "f2": "最新价",
        "f3": "涨跌幅",
        "f4": "涨跌额",
        "f5": "成交量",
        "f6": "成交额",
        "f7": "振幅",
        "f15": "最高",
        "f16": "最低",
        "f17": "今开",
        "f18": "昨收",
        "f20": "总市值",
        "f21": "流通市值",
    }
    records = []
    for r in data["data"]["diff"]:
        rec = {}
        for fk, cn in col_map.items():
            val = r.get(fk, "-")
            rec[cn] = val if val != "-" else None
        records.append(rec)
    return pd.DataFrame(records)


_ALL_STOCKS_CACHE: pd.DataFrame | None = None
"""``ak.stock_info_a_code_name()`` 的模块级缓存。

该调用耗时约 6 秒，因为它抓取了完整的 A 股名单。只需在每个 MCP子进程生命周期内支付一次该开销；后续的 ``search_stock_by_name`` 调用是纯 pandas 过滤操作。
"""


def _fmt_error(exc: Exception, *, context: str) -> dict[str, Any]:
    """标准错误格式 — LLM 可读，无堆栈跟踪。同时记录日志便于终端排查。"""
    logger.error("[%s] %s: %s", context, type(exc).__name__, exc)
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
            elif isinstance(val, pd.Timestamp | datetime):
                rec[str(col)] = val.strftime("%Y-%m-%d")
            elif isinstance(val, int | float | str | bool):
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

    def _basic_info_from_name_cache(sym: str) -> dict[str, Any]:
        """从 A 股名单缓存获取基本名称信息（无网络请求）。"""
        global _ALL_STOCKS_CACHE
        if _ALL_STOCKS_CACHE is None:
            import akshare as ak

            _ALL_STOCKS_CACHE = ak.stock_info_a_code_name()
        df = _ALL_STOCKS_CACHE
        match = df[df["code"].astype(str) == sym]
        if match.empty:
            raise ValueError(f"未在 A 股名单中找到代码 {sym}")
        name = str(match.iloc[0]["name"])
        return {
            "symbol": sym,
            "info": {"股票代码": sym, "股票简称": name},
            "source": "local_cache",
            "note": "仅基本名称信息，实时行情端点当前不可达",
        }

    errors: list[str] = []
    for label, fn in (
        ("eastmoney", _basic_info_from_eastmoney),
        ("xueqiu", _basic_info_from_xueqiu),
        ("local_cache", _basic_info_from_name_cache),
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
    # 新浪优先：东财 push2 端点在部分网络环境下不可达
    for label, fn in (
        ("sina", _price_history_from_sina),
        ("eastmoney", _price_history_from_eastmoney),
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
            mask = df[indicator_col].astype(str).str.contains(metric, na=False, regex=False)
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
            raise RuntimeError(f"unexpected schema from stock_info_a_code_name: {list(df.columns)}")
        mask = df["name"].astype(str).str.contains(keyword, na=False, regex=False)
        hits = df[mask].head(limit)
        matches = [{"code": str(r["code"]), "name": str(r["name"])} for _, r in hits.iterrows()]
        return {"keyword": keyword, "matches": matches}

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(
            e,
            context=f"search_stock_by_name(keyword={keyword!r}, limit={limit})",
        )


# ---------------------------------------------------------------------
# 工具 6: 主要指数实时行情（上证、沪深300、创业板等）
# ---------------------------------------------------------------------
_INDEX_MAP: dict[str, str] = {
    "上证指数": "000001",
    "深证成指": "399001",
    "沪深300": "000300",
    "创业板指": "399006",
    "科创50": "000688",
    "中证500": "000905",
    "中证1000": "000852",
    "上证50": "000016",
}


@mcp.tool()
async def get_index_quotes() -> dict:
    """返回 A 股主要指数（上证指数、沪深300、创业板指、科创50 等）的最新行情。

    无需传入参数，一次性返回所有核心指数的最新价、涨跌幅、成交额。
    适合回答"今天大盘怎么样"、"A 股整体走势"等宏观类问题。

    Returns:
        包含 ``indices`` 列表的字典，每项含 name/code/最新价/涨跌幅/成交额等。
    """

    def _call_curl_cffi() -> dict[str, Any] | None:
        # 指数在 push2 的 fs 参数为各指数的 secid
        core_codes = set(_INDEX_MAP.values())
        secids = []
        for code in core_codes:
            # 上证指数=1.000001, 深证/创业板=0.399xxx
            prefix = "1" if code.startswith("0000") and code != "000300" else "0"
            if code in ("000300", "000905", "000852", "000688"):
                prefix = "1"
            secids.append(f"{prefix}.{code}")
        fs = ",".join(secids)

        url = (
            f"https://88.push2.eastmoney.com/api/qt/ulist.np/get?"
            f"fltt=2&fields=f2,f3,f4,f6,f12,f14&secids={fs}"
        )
        data = _curl_get_json(url, timeout=8)
        if not data or not data.get("data") or not data["data"].get("diff"):
            return None
        records = []
        for r in data["data"]["diff"]:
            records.append(
                {
                    "代码": r.get("f12"),
                    "名称": r.get("f14"),
                    "最新价": r.get("f2"),
                    "涨跌幅": r.get("f3"),
                    "涨跌额": r.get("f4"),
                    "成交额": r.get("f6"),
                }
            )
        return {
            "indices": records,
            "source": "eastmoney_push2_curl",
            "source_url": "https://quote.eastmoney.com/center/gridlist.html#index_sz",
        }

    def _call_akshare() -> dict[str, Any]:
        import akshare as ak

        df = ak.stock_zh_index_spot_em()
        core_codes = set(_INDEX_MAP.values())
        mask = df["代码"].isin(core_codes)
        result = df[mask][["代码", "名称", "最新价", "涨跌幅", "涨跌额", "成交额"]].copy()
        records = _df_to_records(result)
        return {
            "indices": records,
            "source": "eastmoney",
            "source_url": "https://quote.eastmoney.com/center/gridlist.html#index_sz",
        }

    def _call_tencent_index() -> dict[str, Any] | None:
        """通过腾讯 qt.gtimg.cn 获取指数实时行情，不依赖 push2。"""
        tencent_map = {
            "上证指数": "sh000001",
            "深证成指": "sz399001",
            "沪深300": "sh000300",
            "创业板指": "sz399006",
            "科创50": "sh000688",
            "中证500": "sh000905",
            "中证1000": "sh000852",
            "上证50": "sh000016",
        }
        url = f"https://qt.gtimg.cn/q={','.join(tencent_map.values())}"
        try:
            resp = _curl_requests.get(url, impersonate="chrome", timeout=10)
            records = []
            for line in resp.text.strip().split("\n"):
                line = line.strip().rstrip(";")
                if not line or "=" not in line:
                    continue
                parts = line.split("~")
                if len(parts) < 45:
                    continue
                try:
                    records.append(
                        {
                            "代码": parts[2],
                            "名称": parts[1],
                            "最新价": float(parts[3]) if parts[3] else None,
                            "涨跌幅": float(parts[32]) if parts[32] else None,
                            "涨跌额": float(parts[31]) if parts[31] else None,
                            "成交额": float(parts[37]) * 10000 if parts[37] else None,
                        }
                    )
                except (ValueError, IndexError):
                    continue
            if records:
                return {
                    "indices": records,
                    "source": "tencent_realtime",
                    "source_url": "https://quote.eastmoney.com/center/gridlist.html#index_sz",
                }
        except Exception as e:
            logger.debug("腾讯指数行情失败: %s", e)
        return None

    try:
        result = await asyncio.to_thread(_call_curl_cffi)
        if result:
            return result
        # 不再预先检查 push2 连通性，直接尝试 akshare
        try:
            return await asyncio.to_thread(_call_akshare)
        except Exception as e:
            logger.debug("akshare 指数行情失败: %s", e)
        # 腾讯兜底
        if _HAS_CURL_CFFI:
            result = await asyncio.to_thread(_call_tencent_index)
            if result:
                return result
        return {
            "error": "所有指数行情数据源均不可用",
            "context": "get_index_quotes()",
            "hint": "可尝试 get_stock_price_history 查看指数成分股的历史走势",
        }
    except Exception as e:
        return _fmt_error(e, context="get_index_quotes()")


# ---------------------------------------------------------------------
# 工具 7: 板块资金流向（行业板块 / 概念板块）
# ---------------------------------------------------------------------
@mcp.tool()
async def get_sector_fund_flow(sector_type: str = "行业", limit: int = 15) -> dict:
    """返回 A 股板块资金流向排行。

    Args:
        sector_type: ``"行业"``（申万一级行业）或 ``"概念"``（东方财富概念板块）。
        limit: 返回条目数（默认 15，上限 50）。

    Returns:
        包含板块名称、主力净流入、涨跌幅等排行的字典。
        适合回答"今天哪些板块最强"、"科技板块资金流向"等问题。
    """
    limit = max(1, min(limit, 50))

    def _call() -> dict[str, Any]:
        import akshare as ak

        if sector_type == "概念":
            df = ak.stock_board_concept_name_em()
        else:
            df = ak.stock_board_industry_name_em()
        df = df.head(limit)
        records = _df_to_records(df)
        sector_slug = "concept" if sector_type == "概念" else "industry"
        return {
            "sector_type": sector_type,
            "sectors": records,
            "source": "eastmoney",
            "source_url": f"https://data.eastmoney.com/bkzj/{sector_slug}.html",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_sector_fund_flow(sector_type={sector_type!r})")


# ---------------------------------------------------------------------
# 工具 8: A 股涨跌幅排行（今日涨幅/跌幅前 N）
# ---------------------------------------------------------------------
@mcp.tool()
async def get_stock_rank(direction: str = "涨幅榜", limit: int = 20) -> dict:
    """返回今日 A 股涨跌幅排行榜。

    Args:
        direction: ``"涨幅榜"`` 或 ``"跌幅榜"``。
        limit: 返回条目数（默认 20，上限 50）。

    Returns:
        按涨/跌幅排序的股票列表，包含代码、名称、最新价、涨跌幅、成交额。
        适合回答"今天涨停最多的是什么股"、"哪些股票涨得最好"等问题。
    """
    limit = max(1, min(limit, 50))

    def _call_sina() -> dict[str, Any] | None:
        """新浪实时行情：全部 A 股，约 34 秒。"""
        import akshare as ak

        df = ak.stock_zh_a_spot()
        if df is None or df.empty:
            return None
        # 统一列名（新浪用中文列名: 代码/名称/最新价/涨跌幅/涨跌额/成交量/成交额）
        if "代码" in df.columns:
            df["代码"] = df["代码"].astype(str).str.replace(r"^(sh|sz|bj)", "", regex=True)
        for col in ("最新价", "涨跌幅", "涨跌额", "成交额"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        ascending = direction == "跌幅榜"
        df = df.sort_values("涨跌幅", ascending=ascending).head(limit)
        cols = ["代码", "名称", "最新价", "涨跌幅", "涨跌额", "成交额"]
        available_cols = [c for c in cols if c in df.columns]
        rank_slug = "fall" if direction == "跌幅榜" else "rise"
        return {
            "direction": direction,
            "stocks": _df_to_records(df[available_cols]),
            "count": len(df),
            "source": "sina_realtime",
            "source_url": f"https://quote.eastmoney.com/center/gridlist.html#hs_a_board-{rank_slug}",
        }

    def _call_tencent() -> dict[str, Any] | None:
        """腾讯批量行情：需先获取代码列表再批量查询，约 21 秒。"""
        global _ALL_STOCKS_CACHE  # noqa: PLW0603
        if _ALL_STOCKS_CACHE is None:
            import akshare as ak

            _ALL_STOCKS_CACHE = ak.stock_info_a_code_name()
        all_codes = [
            c
            for c in _ALL_STOCKS_CACHE["code"].tolist()
            if c.startswith(("0", "3", "6")) and not c.startswith(("200", "900"))
        ]
        df = _fetch_tencent_realtime(all_codes)
        if df is None or df.empty:
            return None
        ascending = direction == "跌幅榜"
        df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
        df = df.sort_values("涨跌幅", ascending=ascending).head(limit)
        cols = ["代码", "名称", "最新价", "涨跌幅", "涨跌额", "成交额", "换手率"]
        available_cols = [c for c in cols if c in df.columns]
        rank_slug = "fall" if direction == "跌幅榜" else "rise"
        return {
            "direction": direction,
            "stocks": _df_to_records(df[available_cols]),
            "count": len(df),
            "source": "tencent_realtime",
            "source_url": f"https://quote.eastmoney.com/center/gridlist.html#hs_a_board-{rank_slug}",
        }

    try:
        result = await asyncio.to_thread(_call_sina)
        if result:
            return result
        result = await asyncio.to_thread(_call_tencent)
        if result:
            return result
        return {
            "error": "实时行情数据源均不可用",
            "context": f"get_stock_rank(direction={direction!r})",
            "hint": "可用 get_stock_price_history 查看个股历史涨跌",
        }
    except Exception as e:
        return _fmt_error(e, context=f"get_stock_rank(direction={direction!r})")


# =====================================================================
# 工具 9: 分时数据（日内 1/5/15/30/60 分钟 K 线）
# =====================================================================
@mcp.tool()
async def get_intraday(
    symbol: str,
    period: str = "5",
    limit: int = 48,
) -> dict:
    """返回 A 股的日内分时数据。

    Args:
        symbol: 6 位 A 股代码，例如 ``"600519"``。
        period: 分钟周期 — ``"1"`` / ``"5"`` / ``"15"`` / ``"30"`` / ``"60"``。
        limit: 返回条目数（默认 48，上限 240）。

    Returns:
        包含时间、开盘、收盘、最高、最低、成交量的列表。
    """
    limit = max(1, min(limit, 240))

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.stock_zh_a_hist_min_em(symbol=symbol, period=period, adjust="")
        df = df.tail(limit)
        cols = [
            c
            for c in ["时间", "开盘", "收盘", "最高", "最低", "成交量", "成交额"]
            if c in df.columns
        ]
        return {
            "symbol": symbol,
            "period": f"{period}min",
            "records": _df_to_records(df[cols]),
            "count": len(df),
            "source": "eastmoney",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_intraday(symbol={symbol!r}, period={period!r})")


# =====================================================================
# 工具 10: 龙虎榜详情（当日/指定日期）
# =====================================================================
@mcp.tool()
async def get_lhb_detail(date: str = "", limit: int = 20) -> dict:
    """返回龙虎榜（大单异动）详情。

    Args:
        date: 日期，格式 ``"YYYYMMDD"``。留空则取最近交易日。
        limit: 返回条目数（默认 20，上限 50）。

    Returns:
        龙虎榜上榜个股列表，包含代码、名称、收盘价、涨跌幅、上榜原因、买入额、卖出额、净买入额等。
    """
    limit = max(1, min(limit, 50))

    def _call() -> dict[str, Any]:
        import akshare as ak

        if date:
            df = ak.stock_lhb_detail_em(start_date=date, end_date=date)
        else:
            today = datetime.now().strftime("%Y%m%d")
            week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
            df = ak.stock_lhb_detail_em(start_date=week_ago, end_date=today)
        df = df.head(limit)
        return {
            "records": _df_to_records(df),
            "count": len(df),
            "source": "eastmoney",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_lhb_detail(date={date!r})")


# =====================================================================
# 工具 11: 融资融券（个股明细）
# =====================================================================
@mcp.tool()
async def get_margin_detail(symbol: str, limit: int = 20) -> dict:
    """返回个股融资融券数据明细。

    Args:
        symbol: 6 位 A 股代码。
        limit: 返回条目数（默认 20，上限 60）。

    Returns:
        包含日期、融资余额、融资买入额、融券余量、融券卖出量等字段。
    """
    limit = max(1, min(limit, 60))

    def _call() -> dict[str, Any]:
        import akshare as ak

        # stock_margin_detail_sse/szse 按日期查询全市场，再按代码过滤
        today = datetime.now().strftime("%Y%m%d")
        errors: list[str] = []
        for fn_name, fn in [
            ("sse", ak.stock_margin_detail_sse),
            ("szse", ak.stock_margin_detail_szse),
        ]:
            try:
                df = fn(date=today)
                if df.empty:
                    continue
                code_col = next(
                    (c for c in df.columns if "代码" in c or "标的" in c or "code" in c.lower()),
                    None,
                )
                if code_col:
                    df = df[df[code_col].astype(str).str.contains(symbol, na=False)]
                if not df.empty:
                    df = df.tail(limit)
                    return {
                        "symbol": symbol,
                        "records": _df_to_records(df),
                        "count": len(df),
                        "source": fn_name,
                    }
            except Exception as e:
                errors.append(f"{fn_name}: {type(e).__name__}: {str(e)[:80]}")
        # 全市场汇总作为 fallback
        try:
            if symbol.startswith("6"):
                df = ak.stock_margin_sse(start_date=today, end_date=today)
            else:
                df = ak.stock_margin_szse(start_date=today, end_date=today)
            return {
                "symbol": symbol,
                "records": _df_to_records(df.tail(limit)),
                "count": len(df),
                "source": "market_aggregate",
                "note": "个股明细不可用，返回市场汇总",
            }
        except Exception:
            pass
        return {
            "symbol": symbol,
            "records": [],
            "count": 0,
            "source": "none",
            "note": f"融资融券数据不可用: {'; '.join(errors) if errors else '无数据'}",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_margin_detail(symbol={symbol!r})")


# =====================================================================
# 工具 12: 十大流通股东
# =====================================================================
@mcp.tool()
async def get_top_holders(symbol: str) -> dict:
    """返回个股最新一期的十大流通股东。

    Args:
        symbol: 6 位 A 股代码。

    Returns:
        股东列表，包含股东名称、持股数量、持股比例、增减情况。
    """

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.stock_circulate_stock_holder(symbol=symbol)
        if df.empty:
            return {"symbol": symbol, "holders": [], "source": "eastmoney"}
        date_col = next(
            (c for c in df.columns if "日期" in c or "截止" in c or "报告" in c),
            df.columns[0],
        )
        latest_date = df[date_col].iloc[0]
        df = df[df[date_col] == latest_date]
        return {
            "symbol": symbol,
            "report_date": str(latest_date),
            "holders": _df_to_records(df.head(10)),
            "source": "eastmoney",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_top_holders(symbol={symbol!r})")


# =====================================================================
# 工具 13: ETF 实时行情
# =====================================================================
@mcp.tool()
async def get_etf_spot(limit: int = 30) -> dict:
    """返回 A 股 ETF 基金实时行情排行。

    Args:
        limit: 返回条目数（默认 30，上限 100）。

    Returns:
        按成交额排序的 ETF 列表，包含代码、名称、最新价、涨跌幅、成交额。
    """
    limit = max(1, min(limit, 100))

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.fund_etf_spot_em()
        df = df.sort_values("成交额", ascending=False).head(limit)
        cols = [
            c for c in ["代码", "名称", "最新价", "涨跌幅", "成交额", "流通市值"] if c in df.columns
        ]
        return {
            "etfs": _df_to_records(df[cols] if cols else df),
            "count": len(df),
            "source": "eastmoney",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context="get_etf_spot()")


# =====================================================================
# 工具 14: 宏观经济数据（GDP/CPI/PMI/社融等）
# =====================================================================
@mcp.tool()
async def get_macro_china(indicator: str = "gdp", limit: int = 12) -> dict:
    """返回中国宏观经济数据。

    Args:
        indicator: 指标名称，可选 ``"gdp"`` / ``"cpi"`` / ``"pmi"`` / ``"money_supply"`` / ``"social_financing"``。
        limit: 返回条目数（默认 12 期）。

    Returns:
        该指标的时间序列数据。
    """
    limit = max(1, min(limit, 60))
    indicator_map = {
        "gdp": "macro_china_gdp",
        "cpi": "macro_china_cpi",
        "pmi": "macro_china_pmi",
        "money_supply": "macro_china_money_supply",
        "social_financing": "macro_china_shrzgm",
    }

    def _call() -> dict[str, Any]:
        import akshare as ak

        func_name = indicator_map.get(indicator, "macro_china_gdp")
        fn = getattr(ak, func_name, None)
        if fn is None:
            return {"error": f"不支持的指标: {indicator}", "supported": list(indicator_map.keys())}
        df = fn()
        df = df.tail(limit)
        return {
            "indicator": indicator,
            "records": _df_to_records(df),
            "count": len(df),
            "source": "eastmoney/stats.gov.cn",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_macro_china(indicator={indicator!r})")


# =====================================================================
# 工具 15: 概念板块列表 + 成分股
# =====================================================================
@mcp.tool()
async def get_concept_board(board_name: str = "", limit: int = 20) -> dict:
    """返回 A 股概念板块行情或指定板块的成分股。

    Args:
        board_name: 概念板块名称，如 ``"人工智能"`` / ``"芯片"``。
            留空返回所有概念板块排行。
        limit: 返回条目数（默认 20，上限 50）。

    Returns:
        板块行情列表（含板块名、涨跌幅、领涨股）或指定板块的成分股列表。
    """
    limit = max(1, min(limit, 50))

    def _call() -> dict[str, Any]:
        import akshare as ak

        if not board_name:
            df = ak.stock_board_concept_name_em()
            df = (
                df.sort_values("涨跌幅", ascending=False).head(limit)
                if "涨跌幅" in df.columns
                else df.head(limit)
            )
            return {
                "type": "概念板块列表",
                "boards": _df_to_records(df),
                "count": len(df),
                "source": "eastmoney",
            }
        df = ak.stock_board_concept_cons_em(symbol=board_name)
        df = df.head(limit)
        cols = [c for c in ["代码", "名称", "最新价", "涨跌幅", "成交额"] if c in df.columns]
        return {
            "board": board_name,
            "stocks": _df_to_records(df[cols] if cols else df),
            "count": len(df),
            "source": "eastmoney",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_concept_board(board_name={board_name!r})")


# =====================================================================
# 工具 16: 行业板块列表 + 成分股
# =====================================================================
@mcp.tool()
async def get_industry_board(board_name: str = "", limit: int = 20) -> dict:
    """返回 A 股行业板块行情或指定行业的成分股。

    Args:
        board_name: 行业名称，如 ``"半导体"`` / ``"白酒"``。
                    留空返回所有行业板块排行。
        limit: 返回条目数（默认 20，上限 50）。

    Returns:
        行业列表（含板块名、涨跌幅、领涨股）或指定行业的成分股列表。
    """
    limit = max(1, min(limit, 50))

    def _call() -> dict[str, Any]:
        import akshare as ak

        if not board_name:
            df = ak.stock_board_industry_name_em()
            df = (
                df.sort_values("涨跌幅", ascending=False).head(limit)
                if "涨跌幅" in df.columns
                else df.head(limit)
            )
            return {
                "type": "行业板块列表",
                "boards": _df_to_records(df),
                "count": len(df),
                "source": "eastmoney",
            }
        df = ak.stock_board_industry_cons_em(symbol=board_name)
        df = df.head(limit)
        cols = [c for c in ["代码", "名称", "最新价", "涨跌幅", "成交额"] if c in df.columns]
        return {
            "board": board_name,
            "stocks": _df_to_records(df[cols] if cols else df),
            "count": len(df),
            "source": "eastmoney",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_industry_board(board_name={board_name!r})")


# =====================================================================
# 工具 17: 个股资金流向
# =====================================================================
@mcp.tool()
async def get_individual_fund_flow(symbol: str, limit: int = 20) -> dict:
    """返回个股的资金流向数据（主力、超大单、大单、中单、小单）。

    Args:
        symbol: 6 位 A 股代码。
        limit: 返回条目数（默认 20，上限 60）。

    Returns:
        按日期排列的资金流向数据。
    """
    limit = max(1, min(limit, 60))

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.stock_individual_fund_flow(stock=symbol, market=_exchange_prefix(symbol))
        df = df.tail(limit)
        return {
            "symbol": symbol,
            "records": _df_to_records(df),
            "count": len(df),
            "source": "eastmoney",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_individual_fund_flow(symbol={symbol!r})")


# =====================================================================
# 工具 18: 港股通（北向/南向资金流）
# =====================================================================
@mcp.tool()
async def get_hsgt_flow(direction: str = "north", limit: int = 20) -> dict:
    """返回沪深港通资金流向数据。

    Args:
        direction: ``"north"`` 北向资金（外资流入 A 股）或 ``"south"`` 南向资金（内资流入港股）。
        limit: 返回条目数（默认 20，上限 60）。

    Returns:
        每日资金净流入/流出数据。
    """
    limit = max(1, min(limit, 60))

    def _call() -> dict[str, Any]:
        import akshare as ak

        label = "北向资金" if direction == "north" else "南向资金"
        df = ak.stock_hsgt_hist_em(symbol=label)
        df = df.tail(limit)
        return {
            "direction": label,
            "records": _df_to_records(df),
            "count": len(df),
            "source": "eastmoney",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_hsgt_flow(direction={direction!r})")


# =====================================================================
# 工具 19: 市场交易状态（开盘/收盘/节假日）
# =====================================================================
_TRADE_DATES_CACHE: set[str] | None = None
_TRADE_DATES_TS: float = 0.0


def _load_trade_dates() -> set[str]:
    """加载新浪交易日历并缓存 24 小时。"""
    global _TRADE_DATES_CACHE, _TRADE_DATES_TS  # noqa: PLW0603
    import time

    now = time.time()
    if _TRADE_DATES_CACHE and (now - _TRADE_DATES_TS < 86400):
        return _TRADE_DATES_CACHE
    try:
        import akshare as ak

        df = ak.tool_trade_date_hist_sina()
        _TRADE_DATES_CACHE = set(df["trade_date"].astype(str).str[:10])
        _TRADE_DATES_TS = now
    except Exception:
        if _TRADE_DATES_CACHE:
            return _TRADE_DATES_CACHE
        _TRADE_DATES_CACHE = set()
    return _TRADE_DATES_CACHE


@mcp.tool()
async def get_market_status() -> dict:
    """返回 A 股市场当前交易状态（开盘中 / 已收盘 / 未开盘 / 非交易日）。

    无需参数。返回字典包含：
    - ``status``: ``"trading"`` / ``"closed"`` / ``"pre_market"`` / ``"non_trading_day"``
    - ``is_trading_day``: 今天是否为交易日
    - ``current_time``: 当前北京时间
    - ``message``: 中文状态描述
    - ``last_trading_day``: 最近一个交易日（如果今天非交易日或盘前）
    - ``hint``: 给 LLM 的指导提示

    **建议**：当用户问"今天大盘怎么样""收盘分析""市场情况"等问题时，
    先调用此工具判断市场状态，再决定如何描述数据的时效性。
    """

    try:
        return await asyncio.to_thread(_compute_market_status)
    except Exception as e:
        return _fmt_error(e, context="get_market_status()")


def _compute_market_status(
    *, _now: datetime | None = None
) -> dict[str, Any]:
    """纯函数：根据当前时间和交易日历计算市场状态。

    ``_now`` 仅供单元测试注入假时间，生产代码不传。
    """
    from zoneinfo import ZoneInfo

    bj = ZoneInfo("Asia/Shanghai")
    now = _now or datetime.now(bj)
    today_str = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    weekday = now.weekday()

    trade_dates = _load_trade_dates()
    is_trading_day = today_str in trade_dates if trade_dates else weekday < 5

    def _find_last_trading_day() -> str:
        d = now.date() - timedelta(days=1)
        for _ in range(10):
            ds = d.strftime("%Y-%m-%d")
            if trade_dates:
                if ds in trade_dates:
                    return ds
            elif d.weekday() < 5:
                return ds
            d -= timedelta(days=1)
        return (now.date() - timedelta(days=1)).strftime("%Y-%m-%d")

    if not is_trading_day:
        reason = "周末" if weekday >= 5 else "节假日"
        last_td = _find_last_trading_day()
        return {
            "status": "non_trading_day",
            "is_trading_day": False,
            "current_time": current_time,
            "today": today_str,
            "reason": reason,
            "last_trading_day": last_td,
            "message": f"今天（{today_str}）是{reason}，A 股不开盘。",
            "hint": (
                f"当前可用数据为上一个交易日（{last_td}）的收盘数据。"
                f"请明确告知用户数据对应的日期，不要说'今日收盘分析'。"
            ),
        }

    hour_min = now.hour * 100 + now.minute
    if hour_min < 915:
        last_td = _find_last_trading_day()
        return {
            "status": "pre_market",
            "is_trading_day": True,
            "current_time": current_time,
            "today": today_str,
            "last_trading_day": last_td,
            "message": f"今天是交易日，但尚未开盘（当前 {now.strftime('%H:%M')}，9:30 开盘）。",
            "hint": (
                f"当前可用数据为上一个交易日（{last_td}）的收盘数据。"
                f"请告知用户'盘前，以下为昨日收盘数据'。"
            ),
        }
    if hour_min < 930:
        return {
            "status": "call_auction",
            "is_trading_day": True,
            "current_time": current_time,
            "today": today_str,
            "message": f"集合竞价中（{now.strftime('%H:%M')}），9:30 正式开盘。",
            "hint": "可获取集合竞价阶段的参考价格，但正式行情需等 9:30。",
        }
    if hour_min <= 1130 or (1300 <= hour_min <= 1500):
        return {
            "status": "trading",
            "is_trading_day": True,
            "current_time": current_time,
            "today": today_str,
            "message": f"A 股交易中（{now.strftime('%H:%M')}）。",
            "hint": "市场正在交易，获取的行情数据为实时数据。",
        }
    if 1130 < hour_min < 1300:
        return {
            "status": "lunch_break",
            "is_trading_day": True,
            "current_time": current_time,
            "today": today_str,
            "message": f"午间休市（{now.strftime('%H:%M')}，13:00 恢复交易）。",
            "hint": "上午交易已结束，数据为上午收盘时的最新状态，13:00 后恢复实时更新。",
        }
    return {
        "status": "closed",
        "is_trading_day": True,
        "current_time": current_time,
        "today": today_str,
        "last_trading_day": today_str,
        "message": f"今日 A 股已收盘（{now.strftime('%H:%M')}）。",
        "hint": "市场已收盘，获取的数据为今日收盘数据，可以直接说'今日收盘分析'。",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
