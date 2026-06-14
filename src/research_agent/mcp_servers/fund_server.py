"""MCP Server — 通过 ``akshare`` 获取中国公募基金数据。

本服务器专注于基金分析，覆盖 ETF、LOF、开放式基金、货币基金等品种。

提供的工具
----------
1. ``search_fund``          — 按名称/代码搜索基金。
2. ``get_fund_info``        — 基金概况（类型、规模、基金经理、成立日期等）。
3. ``get_fund_nav``         — 开放式基金历史净值（单位净值 + 累计净值）。
4. ``get_fund_etf_spot``    — ETF 基金实时行情排行（成交额/涨跌幅排序）。
5. ``get_fund_lof_spot``    — LOF 基金实时行情排行。
6. ``get_fund_etf_hist``    — 单只 ETF 历史 K 线。
7. ``get_fund_holdings``    — 基金持仓明细（重仓股）。
8. ``get_fund_rating``      — 基金综合评级（上海证券/招商/济安/晨星）。
9. ``get_fund_rank``        — 基金业绩排行（按近1年/3年/5年收益）。
10. ``get_fund_daily``      — 当日全市场开放式基金净值列表。

数据来源
--------
- 东方财富基金网 / 天天基金网 (fund.eastmoney.com)
- push2.eastmoney.com — ETF/LOF 实时行情推送（工具 4/5/6 优先使用）

设计说明
--------
- ``akshare`` 是同步的，用 ``asyncio.to_thread`` 包装避免阻塞。
- 所有工具在边界捕获 ``Exception``，记录日志并返回 ``{"error": "..."}``。
- 工具 4/5/6 实现双层逻辑：push2 可达时走实时行情，不可达时降级为收盘净值排行。
- 基金代码为 6 位数字（如 510300、159915、018735）。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import pandas as pd
from fastmcp import FastMCP

logger = logging.getLogger("fund_server")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

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

mcp = FastMCP("FundData")

# ---------------------------------------------------------------------------
# push2 连通性检测 + curl_cffi 直连助手
# ---------------------------------------------------------------------------
# push2 服务器做了 TLS 指纹检测（JA3/JA4），会拒绝 Python requests/urllib 的标准 TLS 栈。
# 必须用 curl_cffi 模拟 Chrome 指纹才能连通。
# push2（实时行情，IP 43.x）目前永久封锁；
# push2his（历史K线，IP 101.x / 61.x）通过 curl_cffi 可正常访问。

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
    """探测 push2 实时端点是否可达（curl_cffi Chrome 指纹）。"""
    url = "https://88.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1&fs=b:MK0021&fields=f12"
    data = _curl_get_json(url, timeout=8)
    if data and data.get("data"):
        logger.info("push2 实时端点探测成功")
        return True
    # 回退到 requests（以防 curl_cffi 不可用）
    try:
        resp = _requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200 and resp.text:
            logger.info("push2 实时端点探测成功 (requests)")
            return True
    except Exception:
        pass
    return False


def _probe_push2his_connectivity() -> bool:
    """探测 push2his（历史K线）端点是否可达。"""
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.510300&fields1=f1,f2,f3&fields2=f51,f52,f53&klt=101&fqt=1&end=20500101&lmt=1"
    data = _curl_get_json(url, timeout=8)
    if data and data.get("data"):
        logger.info("push2his 探测成功")
        return True
    try:
        resp = _requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200 and "klines" in resp.text:
            logger.info("push2his 探测成功 (requests)")
            return True
    except Exception:
        pass
    return False


_PUSH2_AVAILABLE: bool | None = None
_PUSH2HIS_AVAILABLE: bool | None = None
_PROBE_TS: float = 0.0
_PROBE_HIS_TS: float = 0.0
_PROBE_TTL: float = 300.0


def _is_push2_available() -> bool:
    """检测 push2 实时端点连通性（带 5 分钟 TTL 缓存）。"""
    global _PUSH2_AVAILABLE, _PROBE_TS  # noqa: PLW0603
    now = _time.time()
    if _PUSH2_AVAILABLE is None or (now - _PROBE_TS > _PROBE_TTL):
        _PROBE_TS = now
        _PUSH2_AVAILABLE = _probe_push2_connectivity()
        if not _PUSH2_AVAILABLE:
            logger.warning("push2.eastmoney.com 不可达——ETF/LOF 实时行情将降级为收盘净值排行。")
        else:
            logger.info("push2.eastmoney.com 连通性正常，ETF/LOF 使用实时行情")
    return _PUSH2_AVAILABLE


def _is_push2his_available() -> bool:
    """检测 push2his 历史K线端点连通性（带 5 分钟 TTL 缓存）。"""
    global _PUSH2HIS_AVAILABLE, _PROBE_HIS_TS  # noqa: PLW0603
    now = _time.time()
    if _PUSH2HIS_AVAILABLE is None or (now - _PROBE_HIS_TS > _PROBE_TTL):
        _PROBE_HIS_TS = now
        _PUSH2HIS_AVAILABLE = _probe_push2his_connectivity()
        if not _PUSH2HIS_AVAILABLE:
            logger.warning("push2his.eastmoney.com 不可达——ETF 历史 K 线将降级为净值序列。")
        else:
            logger.info("push2his.eastmoney.com 连通性正常")
    return _PUSH2HIS_AVAILABLE


def _fetch_etf_kline_via_curl(
    symbol: str, period: str = "daily", adjust: str = "qfq", limit: int = 60
) -> pd.DataFrame | None:
    """通过 curl_cffi 直连 push2his 获取 ETF/LOF K 线数据，绕过 TLS 指纹检测。

    返回与 akshare fund_etf_hist_em 相同格式的 DataFrame，失败返回 None。
    """
    if not _HAS_CURL_CFFI:
        return None

    period_map = {"daily": "101", "weekly": "102", "monthly": "103"}
    adjust_map = {"qfq": "1", "hfq": "2", "": "0"}

    # 判断 market_id: 沪市=1, 深市=0
    market_id = 1 if symbol.startswith(("5", "6")) else 0

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": f"{market_id}.{symbol}",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": period_map.get(period, "101"),
        "fqt": adjust_map.get(adjust, "1"),
        "end": "20500101",
        "lmt": str(limit),
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    full_url = f"{url}?{query}"

    data = _curl_get_json(full_url, timeout=12)
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


def _fetch_etf_spot_via_curl(
    fs: str = "b:MK0021,b:MK0022,b:MK0023,b:MK0024,b:MK0827",
) -> pd.DataFrame | None:
    """通过 curl_cffi 直连 push2 获取 ETF 实时行情排行，绕过 TLS 指纹检测。"""
    if not _HAS_CURL_CFFI:
        return None

    url = (
        f"https://88.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=200&po=1&np=1"
        f"&fltt=2&invt=2&fid=f6&fs={fs}"
        f"&fields=f2,f3,f4,f5,f6,f7,f12,f14,f15,f16,f17,f18,f20,f21"
    )
    data = _curl_get_json(url, timeout=10)
    if not data or not data.get("data") or not data["data"].get("diff"):
        return None

    rows = data["data"]["diff"]
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
    for r in rows:
        rec = {}
        for fk, cn in col_map.items():
            val = r.get(fk, "-")
            rec[cn] = val if val != "-" else None
        records.append(rec)
    return pd.DataFrame(records)


def _fetch_sina_etf_realtime() -> pd.DataFrame | None:
    """通过新浪接口获取全市场 ETF 实时行情（含最新价、涨跌幅、成交额）。

    ``fund_etf_category_sina("ETF基金")`` 直接返回全部场内交易 ETF 的实时数据，
    无需预设代码列表、无需腾讯中转。约 2 秒返回 ~1500 支 ETF。
    """
    import akshare as ak

    df = ak.fund_etf_category_sina(symbol="ETF基金")
    if df is None or df.empty:
        return None
    # 新浪返回的代码带 sh/sz 前缀，去掉
    if "代码" in df.columns:
        df["代码"] = df["代码"].astype(str).str.replace(r"^(sh|sz)", "", regex=True)
    for col in ("最新价", "涨跌幅", "涨跌额", "成交量", "成交额"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fetch_sina_lof_realtime() -> pd.DataFrame | None:
    """通过新浪接口获取全市场 LOF 实时行情。"""
    import akshare as ak

    df = ak.fund_etf_category_sina(symbol="LOF基金")
    if df is None or df.empty:
        return None
    if "代码" in df.columns:
        df["代码"] = df["代码"].astype(str).str.replace(r"^(sh|sz)", "", regex=True)
    for col in ("最新价", "涨跌幅", "涨跌额", "成交量", "成交额"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


_FUND_NAME_CACHE: pd.DataFrame | None = None


def _fmt_error(exc: Exception, *, context: str) -> dict[str, Any]:
    logger.error("[%s] %s: %s", context, type(exc).__name__, exc)
    return {"error": f"{type(exc).__name__}: {exc}", "context": context}


def _df_to_records(df: pd.DataFrame, *, limit: int | None = None) -> list[dict[str, Any]]:
    if limit is not None:
        df = df.head(limit)
    from datetime import datetime as _dt

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rec: dict[str, Any] = {}
        for col, val in row.items():
            if pd.isna(val):
                rec[str(col)] = None
            elif isinstance(val, pd.Timestamp | _dt):
                rec[str(col)] = val.strftime("%Y-%m-%d")
            elif isinstance(val, int | float | str | bool):
                rec[str(col)] = val
            else:
                rec[str(col)] = str(val)
        records.append(rec)
    return records


def _ensure_fund_cache() -> pd.DataFrame:
    global _FUND_NAME_CACHE  # noqa: PLW0603
    if _FUND_NAME_CACHE is None:
        import akshare as ak

        _FUND_NAME_CACHE = ak.fund_name_em()
    return _FUND_NAME_CACHE


# =====================================================================
# 工具 1: 基金搜索
# =====================================================================
@mcp.tool()
async def search_fund(keyword: str, limit: int = 10) -> dict:
    """按名称或代码搜索基金，返回匹配的基金代码和名称。

    同时匹配基金代码和基金简称。输入 ``"018735"`` 可以精确匹配代码，
    输入 ``"沪深300"`` 可以模糊匹配名称。

    Args:
        keyword: 搜索关键词（基金代码或名称），如 ``"018735"``、``"沪深300"``、``"科技"``。
        limit: 返回条目数（默认 10，上限 30）。

    Returns:
        匹配的基金列表，包含基金代码、简称、类型。
    """
    limit = max(1, min(limit, 30))

    def _call() -> dict[str, Any]:
        df = _ensure_fund_cache()
        code_mask = df["基金代码"].astype(str).str.contains(keyword, case=False, na=False)
        name_mask = df["基金简称"].str.contains(keyword, case=False, na=False)
        mask = code_mask | name_mask
        matched = df[mask].head(limit)
        cols = [c for c in ["基金代码", "基金简称", "基金类型"] if c in matched.columns]
        return {
            "keyword": keyword,
            "funds": _df_to_records(matched[cols] if cols else matched),
            "count": len(matched),
            "source": "eastmoney",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"search_fund(keyword={keyword!r})")


# =====================================================================
# 工具 2: 基金概况
# =====================================================================
@mcp.tool()
async def get_fund_info(symbol: str) -> dict:
    """返回单只基金的概况信息。

    优先东方财富（含规模、经理、费率等完整概况）。

    Args:
        symbol: 6 位基金代码，如 ``"510300"``（沪深300ETF）、``"018735"``（场外基金）。

    Returns:
        基金类型、成立日期、基金规模、基金经理、管理人、托管人等。
    """

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.fund_overview_em(symbol=symbol)
        if df.empty:
            raise ValueError(f"fund_overview_em 返回空数据: {symbol}")
        # fund_overview_em 返回 1 行 × 20 列的 DataFrame，直接转为 dict
        info: dict[str, Any] = {}
        row = df.iloc[0]
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                info[str(col)] = None
            else:
                info[str(col)] = str(val)
        return {"symbol": symbol, "info": info, "source": "eastmoney"}

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_fund_info(symbol={symbol!r})")


# =====================================================================
# 工具 3: 开放式基金历史净值
# =====================================================================
@mcp.tool()
async def get_fund_nav(symbol: str, limit: int = 30) -> dict:
    """返回开放式基金的历史净值数据。

    支持场内（ETF/LOF）和场外（开放式）基金。

    Args:
        symbol: 6 位基金代码，如 ``"018735"``、``"510300"``。
        limit: 返回条目数（默认 30，上限 120）。

    Returns:
        日期、单位净值、累计净值、日增长率。
    """
    limit = max(1, min(limit, 120))

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.fund_open_fund_info_em(symbol=symbol, indicator="单位净值走势")
        if df is None or df.empty:
            return {"symbol": symbol, "records": [], "count": 0, "source": "eastmoney"}
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
        return _fmt_error(e, context=f"get_fund_nav(symbol={symbol!r})")


# =====================================================================
# 工具 4: ETF 实时行情排行
# =====================================================================
@mcp.tool()
async def get_fund_etf_spot(sort_by: str = "成交额", limit: int = 30) -> dict:
    """返回全市场 ETF 基金实时行情排行（盘中实时价格、涨跌幅、成交额）。

    通过新浪财经接口获取全部 ~1500 支场内 ETF 的实时行情数据。
    若新浪不可用则降级为东方财富收盘净值排行。

    Args:
        sort_by: 排序字段 — ``"成交额"``（默认）/ ``"涨跌幅"`` / ``"最新价"``。
        limit: 返回条目数（默认 30，上限 100）。

    Returns:
        ETF 列表，含代码、名称、最新价、涨跌幅、成交额等实时字段。
        响应中 ``source`` 字段标识数据来源，``realtime`` 字段标识是否为盘中实时数据。
    """
    limit = max(1, min(limit, 100))

    def _call_sina() -> dict[str, Any] | None:
        df = _fetch_sina_etf_realtime()
        if df is None or df.empty:
            return None
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)
        df = df.head(limit)
        cols = [
            c
            for c in ["代码", "名称", "最新价", "涨跌幅", "涨跌额", "成交额", "成交量"]
            if c in df.columns
        ]
        return {
            "sort_by": sort_by,
            "etfs": _df_to_records(df[cols] if cols else df),
            "count": len(df),
            "source": "sina_realtime",
            "realtime": True,
            "source_url": "https://quote.eastmoney.com/center/gridlist.html#fund_etf",
        }

    def _call_fallback() -> dict[str, Any]:
        import akshare as ak

        fallback_sort = sort_by if sort_by in ("今年来", "近1周", "近1月", "近1年") else "今年来"
        df = ak.fund_open_fund_rank_em(symbol="指数型")
        if fallback_sort in df.columns:
            df[fallback_sort] = pd.to_numeric(df[fallback_sort], errors="coerce")
            df = df.sort_values(fallback_sort, ascending=False)
        df = df.head(limit)
        cols = [
            c
            for c in ["基金代码", "基金简称", "单位净值", "今年来", "近1周", "近1月", "近1年"]
            if c in df.columns
        ]
        return {
            "sort_by": fallback_sort,
            "etfs": _df_to_records(df[cols] if cols else df),
            "count": len(df),
            "source": "eastmoney_rank",
            "realtime": False,
            "note": "实时数据不可用，已降级为收盘净值排行",
            "source_url": "https://fund.eastmoney.com/data/fundranking.html",
        }

    try:
        result = await asyncio.to_thread(_call_sina)
        if result:
            return result
        return await asyncio.to_thread(_call_fallback)
    except Exception:
        try:
            return await asyncio.to_thread(_call_fallback)
        except Exception as e2:
            return _fmt_error(e2, context="get_fund_etf_spot(fallback)")


# =====================================================================
# 工具 5: LOF 实时行情排行
# =====================================================================
@mcp.tool()
async def get_fund_lof_spot(sort_by: str = "成交额", limit: int = 20) -> dict:
    """返回全市场 LOF 基金实时行情排行（盘中实时价格、涨跌幅、成交额）。

    通过新浪财经接口获取全部 ~380 支场内 LOF 的实时行情数据。
    若新浪不可用则降级为东方财富收盘净值排行。

    Args:
        sort_by: 排序字段 — ``"成交额"``（默认）/ ``"涨跌幅"`` / ``"最新价"``。
        limit: 返回条目数（默认 20，上限 50）。

    Returns:
        LOF 列表，含代码、名称、最新价、涨跌幅、成交额等实时字段。
        响应中 ``source`` 字段标识数据来源，``realtime`` 字段标识是否为盘中实时数据。
    """
    limit = max(1, min(limit, 50))

    def _call_sina() -> dict[str, Any] | None:
        df = _fetch_sina_lof_realtime()
        if df is None or df.empty:
            return None
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)
        df = df.head(limit)
        cols = [
            c
            for c in ["代码", "名称", "最新价", "涨跌幅", "涨跌额", "成交额", "成交量"]
            if c in df.columns
        ]
        return {
            "sort_by": sort_by,
            "lofs": _df_to_records(df[cols] if cols else df),
            "count": len(df),
            "source": "sina_realtime",
            "realtime": True,
            "source_url": "https://quote.eastmoney.com/center/gridlist.html#fund_lof",
        }

    def _call_fallback() -> dict[str, Any]:
        import akshare as ak

        fallback_sort = sort_by if sort_by in ("今年来", "近1周", "近1月", "近1年") else "今年来"
        try:
            df = ak.fund_open_fund_rank_em(symbol="LOF")
        except (IndexError, KeyError, ValueError):
            df = ak.fund_open_fund_rank_em(symbol="全部")
            df = (
                df[df["基金简称"].str.contains("LOF", case=False, na=False)]
                if "基金简称" in df.columns
                else df
            )

        if df is None or df.empty:
            return {
                "sort_by": fallback_sort,
                "lofs": [],
                "count": 0,
                "source": "eastmoney_rank",
                "realtime": False,
            }

        if fallback_sort in df.columns:
            df[fallback_sort] = pd.to_numeric(df[fallback_sort], errors="coerce")
            df = df.sort_values(fallback_sort, ascending=False)
        df = df.head(limit)
        cols = [
            c
            for c in ["基金代码", "基金简称", "单位净值", "今年来", "近1周", "近1月", "近1年"]
            if c in df.columns
        ]
        return {
            "sort_by": fallback_sort,
            "lofs": _df_to_records(df[cols] if cols else df),
            "count": len(df),
            "source": "eastmoney_rank",
            "realtime": False,
            "note": "实时数据不可用，已降级为收盘净值排行",
            "source_url": "https://fund.eastmoney.com/data/fundranking.html",
        }

    try:
        result = await asyncio.to_thread(_call_sina)
        if result:
            return result
        return await asyncio.to_thread(_call_fallback)
    except Exception:
        try:
            return await asyncio.to_thread(_call_fallback)
        except Exception as e2:
            return _fmt_error(e2, context="get_fund_lof_spot(fallback)")


# =====================================================================
# 工具 6: 单只 ETF 历史 K 线
# =====================================================================
@mcp.tool()
async def get_fund_etf_hist(
    symbol: str,
    period: str = "daily",
    limit: int = 60,
) -> dict:
    """返回单只 ETF/LOF 的历史 K 线数据。

    优先通过 ``fund_etf_hist_em``（push2 端点）获取完整 OHLCV K 线；
    若 push2 不可达则降级为 ``fund_open_fund_info_em``（仅单位净值时间序列）。

    Args:
        symbol: 6 位基金代码，如 ``"510300"``、``"159915"``。
        period: K 线周期 — ``"daily"``（日线）/ ``"weekly"``（周线）/ ``"monthly"``（月线）。
                仅实时模式有效，降级模式固定为日级净值。
        limit: 返回条目数（默认 60，上限 250）。

    Returns:
        实时模式: 日期、开盘、收盘、最高、最低、成交量、涨跌幅。
        降级模式: 净值日期、单位净值、日增长率。
        响应中 ``source`` 字段标识数据来源，``realtime`` 字段标识是否为实时 K 线。
    """
    limit = max(1, min(limit, 250))

    def _call_curl_cffi() -> dict[str, Any] | None:
        df = _fetch_etf_kline_via_curl(symbol=symbol, period=period, adjust="qfq", limit=limit)
        if df is None or df.empty:
            return None
        cols = [
            c
            for c in ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "涨跌幅"]
            if c in df.columns
        ]
        return {
            "symbol": symbol,
            "period": period,
            "records": _df_to_records(df[cols] if cols else df),
            "count": len(df),
            "source": "eastmoney_push2his_curl",
            "realtime": True,
        }

    def _call_akshare_realtime() -> dict[str, Any]:
        import akshare as ak

        df = ak.fund_etf_hist_em(symbol=symbol, period=period, adjust="qfq")
        df = df.tail(limit)
        cols = [
            c
            for c in ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "涨跌幅"]
            if c in df.columns
        ]
        return {
            "symbol": symbol,
            "period": period,
            "records": _df_to_records(df[cols] if cols else df),
            "count": len(df),
            "source": "eastmoney_push2",
            "realtime": True,
        }

    def _call_fallback() -> dict[str, Any]:
        import akshare as ak

        df = ak.fund_open_fund_info_em(symbol=symbol, indicator="单位净值走势")
        if df is None or df.empty:
            return {
                "symbol": symbol,
                "records": [],
                "count": 0,
                "source": "eastmoney_nav",
                "realtime": False,
            }
        df = df.tail(limit)
        return {
            "symbol": symbol,
            "period": "daily(nav)",
            "records": _df_to_records(df),
            "count": len(df),
            "source": "eastmoney_nav",
            "realtime": False,
            "note": "push2his 不可达，已降级为历史净值序列（无 OHLCV）",
        }

    try:
        # 优先 curl_cffi 直连 push2his（绕过 TLS 指纹检测）
        result = await asyncio.to_thread(_call_curl_cffi)
        if result:
            return result
        # 回退 akshare（如 push2his 对 requests 可用）
        if _is_push2his_available():
            return await asyncio.to_thread(_call_akshare_realtime)
        return await asyncio.to_thread(_call_fallback)
    except Exception:
        try:
            return await asyncio.to_thread(_call_fallback)
        except Exception as e2:
            return _fmt_error(e2, context=f"get_fund_etf_hist(symbol={symbol!r}, fallback)")


# =====================================================================
# 工具 7: 基金持仓（重仓股）
# =====================================================================
@mcp.tool()
async def get_fund_holdings(symbol: str, year: str = "2024") -> dict:
    """返回单只基金的重仓股持仓明细。

    Args:
        symbol: 6 位基金代码。
        year: 年份，如 ``"2024"``。

    Returns:
        持仓股票列表，包含股票代码、名称、持仓市值、占净值比例。
    """

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.fund_portfolio_hold_em(symbol=symbol, date=year)
        if df.empty:
            return {"symbol": symbol, "holdings": [], "source": "eastmoney"}
        latest_date = df["季度"].iloc[0] if "季度" in df.columns else ""
        if "季度" in df.columns:
            df = df[df["季度"] == latest_date]
        cols = [
            c
            for c in ["序号", "股票代码", "股票名称", "占净值比例", "持股数", "持仓市值"]
            if c in df.columns
        ]
        return {
            "symbol": symbol,
            "report_period": str(latest_date),
            "holdings": _df_to_records(df[cols] if cols else df, limit=20),
            "source": "eastmoney",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_fund_holdings(symbol={symbol!r})")


# =====================================================================
# 工具 8: 基金评级
# =====================================================================
@mcp.tool()
async def get_fund_rating(limit: int = 30) -> dict:
    """返回公募基金综合评级排行（上海证券/招商/济安/晨星四家机构）。

    Args:
        limit: 返回条目数（默认 30，上限 100）。

    Returns:
        基金列表，包含代码、名称、5星评级家数、各机构评级、基金类型。
    """
    limit = max(1, min(limit, 100))

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.fund_rating_all()
        if "5星评级家数" in df.columns:
            df = df.sort_values("5星评级家数", ascending=False)
        df = df.head(limit)
        cols = [
            c
            for c in [
                "代码",
                "简称",
                "5星评级家数",
                "上海证券",
                "招商证券",
                "济安金信",
                "晨星评级",
                "类型",
            ]
            if c in df.columns
        ]
        return {
            "ratings": _df_to_records(df[cols] if cols else df),
            "count": len(df),
            "source": "eastmoney",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context="get_fund_rating()")


# =====================================================================
# 工具 9: 基金业绩排行
# =====================================================================
@mcp.tool()
async def get_fund_rank(
    fund_type: str = "全部",
    sort_by: str = "近1年",
    limit: int = 20,
) -> dict:
    """返回公募基金业绩排行榜。

    Args:
        fund_type: 基金类型 — ``"全部"`` / ``"股票型"`` / ``"混合型"``
                   / ``"债券型"`` / ``"指数型"`` / ``"QDII"``。
        sort_by: 排序字段 — ``"近1年"`` / ``"近3年"`` / ``"近5年"``
                 / ``"今年来"`` / ``"近1周"`` / ``"近1月"``。
        limit: 返回条目数（默认 20，上限 50）。

    Returns:
        基金列表，包含代码、名称、单位净值、各周期收益率、基金经理。
    """
    limit = max(1, min(limit, 50))

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.fund_open_fund_rank_em(symbol=fund_type)
        if sort_by in df.columns:
            df[sort_by] = pd.to_numeric(df[sort_by], errors="coerce")
            df = df.sort_values(sort_by, ascending=False)
        df = df.head(limit)
        cols = [
            c
            for c in [
                "基金代码",
                "基金简称",
                "单位净值",
                "今年来",
                "近1周",
                "近1月",
                "近1年",
                "近3年",
            ]
            if c in df.columns
        ]
        return {
            "fund_type": fund_type,
            "sort_by": sort_by,
            "funds": _df_to_records(df[cols] if cols else df),
            "count": len(df),
            "source": "eastmoney",
            "source_url": "https://fund.eastmoney.com/data/fundranking.html",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_fund_rank(fund_type={fund_type!r})")


# =====================================================================
# 工具 10: 当日开放式基金净值列表
# =====================================================================
@mcp.tool()
async def get_fund_daily(fund_type: str = "股票型", limit: int = 30) -> dict:
    """返回当日开放式基金净值列表。

    Args:
        fund_type: ``"股票型"`` / ``"混合型"`` / ``"债券型"`` / ``"指数型"``  / ``"QDII"`` / ``"LOF"`` / ``"FOF"``。
        limit: 返回条目数（默认 30，上限 100）。

    Returns:
        基金列表，包含代码、名称、单位净值、累计净值、日增长率。
    """
    limit = max(1, min(limit, 100))

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.fund_open_fund_daily_em()
        if "基金类型" in df.columns:
            df = df[df["基金类型"].str.contains(fund_type, na=False)]
        if "日增长率" in df.columns:
            df["日增长率"] = pd.to_numeric(df["日增长率"], errors="coerce")
            df = df.sort_values("日增长率", ascending=False)
        df = df.head(limit)
        cols = [
            c
            for c in ["基金代码", "基金简称", "单位净值", "累计净值", "日增长率"]
            if c in df.columns
        ]
        return {
            "fund_type": fund_type,
            "funds": _df_to_records(df[cols] if cols else df),
            "count": len(df),
            "source": "eastmoney",
            "source_url": "https://fund.eastmoney.com/fund.html",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_fund_daily(fund_type={fund_type!r})")


if __name__ == "__main__":
    mcp.run(transport="stdio")
