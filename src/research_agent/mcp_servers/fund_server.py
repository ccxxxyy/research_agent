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
11. ``get_fund_qdii_rank``  — QDII 基金业绩排行（专项入口）。
12. ``get_fund_manager``    — 按基金代码查基金经理与基本档案字段。
13. ``search_private_fund`` — 中基协私募产品备案（协会服务端关键词；无实时净值）。
14. ``search_private_manager`` — 中基协私募管理人（协会服务端关键词）。
15. ``get_private_fund_info`` — 私募产品近名/精确备案详情。

数据来源
--------
- 东方财富基金网 / 天天基金网 (fund.eastmoney.com)
- push2.eastmoney.com — ETF/LOF 实时行情推送（工具 4/5/6 优先使用）
- 中国证券投资基金业协会 AMAC 公示 API（``/api/pof/manager|fund`` + keyword；私募备案，无净值）

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
import re
from typing import Any

import pandas as pd
from fastmcp import FastMCP

from research_agent.cache import (
    TTL_DAILY,
    TTL_LONG,
    TTL_REALTIME,
    cached_tool,
)

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
                # 基金代码保持 6 位字符串，避免 018735 → 18735
                if str(col) in {"基金代码", "代码", "symbol"} and isinstance(val, (int, float)):
                    rec[str(col)] = str(int(val)).zfill(6)
                elif str(col) in {"基金代码", "代码", "symbol"} and isinstance(val, str):
                    rec[str(col)] = val.strip().zfill(6) if val.strip().isdigit() else val.strip()
                else:
                    rec[str(col)] = val
            else:
                rec[str(col)] = str(val)
        records.append(rec)
    return records


def _normalize_fund_code(symbol: str) -> str:
    s = str(symbol or "").strip()
    # 去掉概况接口偶发的「（前端）」后缀
    s = re.sub(r"[（(].*?[）)]\s*$", "", s).strip()
    if s.isdigit():
        return s.zfill(6)
    return s


def _normalize_open_fund_daily_df(df: pd.DataFrame) -> pd.DataFrame:
    """东财 ``fund_open_fund_daily_em`` 列名带日期前缀，统一为标准字段。

    实际列类似 ``2026-07-24-单位净值`` / ``日增长率``，
    没有裸 ``单位净值``，旧逻辑按固定列名取值会导致净值整列丢失。
    """
    out = df.copy()
    if "基金代码" in out.columns:
        out["基金代码"] = out["基金代码"].astype(str).map(_normalize_fund_code)

    def _latest_col(suffix: str) -> str | None:
        cols = [c for c in out.columns if str(c).endswith(suffix)]
        if not cols:
            return None
        # 日期前缀字典序即可取到最新交易日列
        return sorted(cols, key=str)[-1]

    unit_col = _latest_col("单位净值")
    acc_col = _latest_col("累计净值")
    if unit_col and "单位净值" not in out.columns:
        out["单位净值"] = pd.to_numeric(out[unit_col], errors="coerce")
        out["净值日期"] = str(unit_col).replace("-单位净值", "")
    if acc_col and "累计净值" not in out.columns:
        out["累计净值"] = pd.to_numeric(out[acc_col], errors="coerce")
    if "日增长率" in out.columns:
        out["日增长率"] = pd.to_numeric(out["日增长率"], errors="coerce")
    return out


def _ensure_fund_cache() -> pd.DataFrame:
    global _FUND_NAME_CACHE  # noqa: PLW0603
    if _FUND_NAME_CACHE is None:
        import akshare as ak

        cache = ak.fund_name_em()
        if "基金代码" in cache.columns:
            cache = cache.copy()
            cache["基金代码"] = cache["基金代码"].astype(str).map(_normalize_fund_code)
        _FUND_NAME_CACHE = cache
    return _FUND_NAME_CACHE


# =====================================================================
# 工具 1: 基金搜索
# =====================================================================
@mcp.tool()
@cached_tool(ttl=TTL_LONG, namespace="fund")
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
        kw = keyword.strip()
        codes = df["基金代码"].astype(str)
        names = df["基金简称"].astype(str)
        cols = [c for c in ["基金代码", "基金简称", "基金类型"] if c in df.columns]

        # 完整 6 位代码：精确匹配。短数字（如 "300"）走包含匹配，
        # 否则 zfill("300")→"000300" 会误匹配失败（测试/用户搜指数简称都会挂）。
        if re.fullmatch(r"\d{6}", kw):
            matched = df[codes == kw].head(limit)
        elif re.fullmatch(r"\d{1,5}", kw):
            padded = kw.zfill(6)
            exact = df[codes == padded]
            if not exact.empty:
                matched = exact.head(limit)
            else:
                code_hit = df[codes.str.contains(kw, na=False)]
                name_hit = df[names.str.contains(kw, case=False, na=False)]
                matched = (
                    pd.concat([code_hit, name_hit], ignore_index=True)
                    .drop_duplicates(subset=["基金代码"])
                    .head(limit)
                )
        else:
            exact_name = df[names == kw]
            start_name = df[names.str.startswith(kw, na=False)]
            contain_name = df[names.str.contains(re.escape(kw), case=False, na=False)]
            code_hit = df[codes.str.contains(re.escape(kw), case=False, na=False)]
            matched = (
                pd.concat([exact_name, start_name, contain_name, code_hit], ignore_index=True)
                .drop_duplicates(subset=["基金代码"])
                .head(limit)
            )

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
@cached_tool(ttl=TTL_LONG, namespace="fund")
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

        code = _normalize_fund_code(symbol)
        df = ak.fund_overview_em(symbol=code)
        if df.empty:
            raise ValueError(f"fund_overview_em 返回空数据: {code}")
        # fund_overview_em 返回 1 行 × 20 列的 DataFrame，直接转为 dict
        info: dict[str, Any] = {}
        row = df.iloc[0]
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                info[str(col)] = None
            else:
                text = str(val)
                if str(col) == "基金代码":
                    text = _normalize_fund_code(text)
                info[str(col)] = text
        return {"symbol": code, "info": info, "source": "eastmoney"}

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_fund_info(symbol={symbol!r})")


# =====================================================================
# 工具 3: 开放式基金历史净值
# =====================================================================
@mcp.tool()
@cached_tool(ttl=TTL_DAILY, namespace="fund")
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

        code = _normalize_fund_code(symbol)
        df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
        if df is None or df.empty:
            return {"symbol": code, "records": [], "count": 0, "source": "eastmoney"}
        if "净值日期" in df.columns:
            df = df.copy()
            df["净值日期"] = pd.to_datetime(df["净值日期"], errors="coerce")
            df = df.sort_values("净值日期")
        for col in ("单位净值", "累计净值", "日增长率"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.tail(limit)
        return {
            "symbol": code,
            "records": _df_to_records(df),
            "count": len(df),
            "source": "eastmoney",
            "note": "日增长率为百分比数值（如 -3.63 表示 -3.63%），勿再乘 100",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_fund_nav(symbol={symbol!r})")


# =====================================================================
# 工具 4: ETF 实时行情排行
# =====================================================================
@mcp.tool()
@cached_tool(ttl=TTL_REALTIME, namespace="fund")
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
@cached_tool(ttl=TTL_REALTIME, namespace="fund")
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
@cached_tool(ttl=TTL_DAILY, namespace="fund")
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
@cached_tool(ttl=TTL_LONG, namespace="fund")
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
@cached_tool(ttl=TTL_LONG, namespace="fund")
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
@cached_tool(ttl=TTL_DAILY, namespace="fund")
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
@cached_tool(ttl=TTL_DAILY, namespace="fund")
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

        raw = ak.fund_open_fund_daily_em()
        df = _normalize_open_fund_daily_df(raw)
        # 该接口本身无「基金类型」列；若用户指定类型，用名称库联表过滤
        note = "日增长率为百分比数值（如 3.73 表示 +3.73%）"
        if fund_type and fund_type not in {"全部", "all", "*"}:
            name_df = _ensure_fund_cache()
            if "基金类型" in name_df.columns:
                typed = name_df[name_df["基金类型"].astype(str).str.contains(fund_type, na=False)][
                    "基金代码"
                ].astype(str)
                before = len(df)
                df = df[df["基金代码"].isin(set(typed))]
                note += f"；已按类型「{fund_type}」过滤 {before}→{len(df)}"
            else:
                note += f"；上游无类型字段，未能按「{fund_type}」过滤"
        if "日增长率" in df.columns:
            df = df.sort_values("日增长率", ascending=False)
        df = df.head(limit)
        cols = [
            c
            for c in ["基金代码", "基金简称", "净值日期", "单位净值", "累计净值", "日增长率"]
            if c in df.columns
        ]
        return {
            "fund_type": fund_type,
            "funds": _df_to_records(df[cols] if cols else df),
            "count": len(df),
            "source": "eastmoney",
            "source_url": "https://fund.eastmoney.com/fund.html",
            "note": note,
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_fund_daily(fund_type={fund_type!r})")


# =====================================================================
# 工具 11: QDII 业绩排行
# =====================================================================
@mcp.tool()
@cached_tool(ttl=TTL_DAILY, namespace="fund")
async def get_fund_qdii_rank(sort_by: str = "近1年", limit: int = 20) -> dict:
    """返回 QDII 基金业绩排行（专项入口，避免与股票型排行混淆）。

    Args:
        sort_by: ``"近1年"`` / ``"近3年"`` / ``"近5年"`` / ``"今年来"`` / ``"近1周"`` / ``"近1月"``。
        limit: 返回条目数（默认 20，上限 50）。
    """
    limit = max(1, min(limit, 50))

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.fund_open_fund_rank_em(symbol="QDII")
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
            "fund_type": "QDII",
            "sort_by": sort_by,
            "funds": _df_to_records(df[cols] if cols else df),
            "count": len(df),
            "source": "eastmoney",
            "source_url": "https://fund.eastmoney.com/data/fundranking.html",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context="get_fund_qdii_rank()")


# =====================================================================
# 工具 12: 基金经理 / 档案
# =====================================================================
@mcp.tool()
@cached_tool(ttl=TTL_LONG, namespace="fund")
async def get_fund_manager(symbol: str) -> dict:
    """按基金代码查询基金经理与基本档案字段（天天基金概况）。

    Args:
        symbol: 6 位基金代码，如 ``110011``、``161725``。
    """
    code = re.sub(r"\D", "", symbol or "")[:6]
    if len(code) != 6:
        return {"error": "invalid fund code", "context": f"get_fund_manager({symbol!r})"}

    def _call() -> dict[str, Any]:
        import akshare as ak

        df = ak.fund_overview_em(symbol=code)
        records = _df_to_records(df) if df is not None else []
        # 常见为 项目/值 两列
        profile: dict[str, Any] = {}
        for row in records:
            keys = list(row.keys())
            if len(keys) >= 2:
                k = str(row.get(keys[0]) or "").strip()
                v = row.get(keys[1])
                if k:
                    profile[k] = v
            else:
                profile.update({str(k): v for k, v in row.items() if v is not None})

        manager_keys = [k for k in profile if "经理" in str(k)]
        managers = {k: profile[k] for k in manager_keys} if manager_keys else {}
        return {
            "symbol": code,
            "profile": profile,
            "managers": managers,
            "count": len(profile),
            "source": "eastmoney",
            "source_url": f"https://fundf10.eastmoney.com/jbgk_{code}.html",
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"get_fund_manager({code!r})")


# =====================================================================
# 私募（AMAC 协会备案公示；无实时净值）
# 走协会服务端关键词检索，禁止冷启动全量/多页翻表。
# =====================================================================
_AMAC_MANAGER_API = "https://gs.amac.org.cn/amac-infodisc/api/pof/manager"
_AMAC_FUND_API = "https://gs.amac.org.cn/amac-infodisc/api/pof/fund"
_AMAC_HTTP_TIMEOUT = float(os.environ.get("AMAC_HTTP_TIMEOUT", "25"))
_AMAC_NOTE = (
    "中国证券投资基金业协会备案公示，仅含登记信息，无实时净值/业绩；"
    "不可与公募 fund_get_fund_nav 口径混用。"
    "品牌名优先查管理人；境外美元基金可能不在协会名册。"
)
_AMAC_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/json",
}


def _amac_ms_to_date(val: Any) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        ts = int(val)
        return pd.to_datetime(ts, unit="ms").strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        s = str(val).strip()
        return s or None


def _amac_post_keyword(url: str, keyword: str, *, page: int, size: int) -> dict[str, Any]:
    """POST 协会 API；body 使用 ``{"keyword": ...}``（管理人侧已验证有效）。"""
    params = {"page": max(0, int(page)), "size": max(1, min(int(size), 100))}
    resp = _requests.post(
        url,
        params=params,
        json={"keyword": keyword.strip()},
        headers=_AMAC_HEADERS,
        timeout=_AMAC_HTTP_TIMEOUT,
        verify=False,
    )
    if resp.status_code >= 500:
        raise RuntimeError(f"AMAC API HTTP {resp.status_code}（协会接口不可用或间歇故障）: {url}")
    if resp.status_code >= 400:
        raise RuntimeError(f"AMAC API HTTP {resp.status_code}: {resp.text[:200]}")
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError(f"AMAC API empty body: {url}")
    try:
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"AMAC API non-JSON ({resp.status_code}): {text[:120]}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("AMAC API returned non-object JSON")
    return payload


def _map_amac_manager_row(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "私募基金管理人名称": raw.get("managerName"),
        "法定代表人/执行事务合伙人(委派代表)姓名": raw.get("artificialPersonName"),
        "机构类型": raw.get("primaryInvestType"),
        "注册地": raw.get("registerProvince"),
        "登记编号": raw.get("registerNo"),
        "成立时间": _amac_ms_to_date(raw.get("establishDate")),
        "登记时间": _amac_ms_to_date(raw.get("registerDate")),
    }


def _map_amac_fund_row(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "基金名称": raw.get("fundName"),
        "私募基金管理人名称": raw.get("managerName"),
        "私募基金管理人类型": raw.get("managerType"),
        "运行状态": raw.get("workingState"),
        "备案时间": _amac_ms_to_date(raw.get("putOnRecordDate")),
        "建立时间": _amac_ms_to_date(raw.get("establishDate")),
        "托管人名称": raw.get("mandatorName"),
    }


def _amac_search_manager_sync(keyword: str, limit: int) -> dict[str, Any]:
    payload = _amac_post_keyword(_AMAC_MANAGER_API, keyword, page=0, size=limit)
    content = payload.get("content") or []
    if not isinstance(content, list):
        content = []
    matches = [_map_amac_manager_row(x) for x in content[:limit] if isinstance(x, dict)]
    total = payload.get("totalElements")
    return {
        "keyword": keyword.strip(),
        "matches": matches,
        "count": len(matches),
        "total_elements": int(total) if total is not None else len(matches),
        "source": "amac",
        "source_url": "https://gs.amac.org.cn/amac-infodisc/res/pof/manager/index.html",
        "note": _AMAC_NOTE,
        "query_mode": "server_keyword",
    }


def _amac_search_fund_sync(keyword: str, limit: int) -> dict[str, Any]:
    payload = _amac_post_keyword(_AMAC_FUND_API, keyword, page=0, size=limit)
    content = payload.get("content") or []
    if not isinstance(content, list):
        content = []
    matches = [_map_amac_fund_row(x) for x in content[:limit] if isinstance(x, dict)]
    total = payload.get("totalElements")
    return {
        "keyword": keyword.strip(),
        "matches": matches,
        "count": len(matches),
        "total_elements": int(total) if total is not None else len(matches),
        "source": "amac",
        "source_url": "https://gs.amac.org.cn/amac-infodisc/res/pof/fund/index.html",
        "note": _AMAC_NOTE,
        "query_mode": "server_keyword",
    }


@mcp.tool()
@cached_tool(ttl=TTL_LONG, namespace="fund")
async def search_private_fund(keyword: str, limit: int = 10) -> dict:
    """按关键词搜索中基协私募基金产品备案公示（无实时净值）。

    走协会服务端关键词检索；若协会产品接口 5xx，返回明确错误（不翻页下载全表）。
    品牌名（如「红杉」）更建议先 ``search_private_manager``。

    Args:
        keyword: 产品名 / 管理人名片段，如 ``"高毅"``、``"景林"``。
        limit: 最大返回条数（默认 10，上限 50）。
    """
    if not (keyword or "").strip():
        return _fmt_error(ValueError("keyword must be non-empty"), context="search_private_fund()")
    limit = max(1, min(int(limit), 50))

    try:
        return await asyncio.to_thread(_amac_search_fund_sync, keyword.strip(), limit)
    except Exception as e:
        err = _fmt_error(e, context=f"search_private_fund(keyword={keyword!r})")
        err["source"] = "amac"
        err["note"] = (
            _AMAC_NOTE + " 产品列表接口可能间歇 500；可改用 search_private_manager 查管理人备案。"
        )
        return err


@mcp.tool()
@cached_tool(ttl=TTL_LONG, namespace="fund")
async def search_private_manager(keyword: str, limit: int = 10) -> dict:
    """按关键词搜索中基协私募基金管理人公示（协会服务端 keyword，无全表下载）。

    Args:
        keyword: 管理人名称片段（如 ``"红杉"``、``"高毅"``）。
        limit: 最大返回条数（默认 10，上限 50）。
    """
    if not (keyword or "").strip():
        return _fmt_error(
            ValueError("keyword must be non-empty"), context="search_private_manager()"
        )
    limit = max(1, min(int(limit), 50))

    try:
        return await asyncio.to_thread(_amac_search_manager_sync, keyword.strip(), limit)
    except Exception as e:
        return _fmt_error(e, context=f"search_private_manager(keyword={keyword!r})")


@mcp.tool()
@cached_tool(ttl=TTL_LONG, namespace="fund")
async def get_private_fund_info(name: str) -> dict:
    """按产品名称取一条私募备案公示详情（精确优先，否则近名首条）。

    Args:
        name: 私募产品全称或显著片段。
    """
    if not (name or "").strip():
        return _fmt_error(ValueError("name must be non-empty"), context="get_private_fund_info()")

    def _call() -> dict[str, Any]:
        result = _amac_search_fund_sync(name.strip(), 20)
        matches = result.get("matches") or []
        exact = [m for m in matches if str(m.get("基金名称") or "") == name.strip()]
        hit = exact[0] if exact else (matches[0] if matches else None)
        if hit is None:
            return {
                "name": name.strip(),
                "info": None,
                "found": False,
                "source": "amac",
                "source_url": result.get("source_url"),
                "note": _AMAC_NOTE,
            }
        return {
            "name": name.strip(),
            "info": hit,
            "found": True,
            "source": "amac",
            "source_url": result.get("source_url"),
            "note": _AMAC_NOTE,
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        err = _fmt_error(e, context=f"get_private_fund_info(name={name!r})")
        err["source"] = "amac"
        err["note"] = _AMAC_NOTE + " 产品列表接口可能间歇 500；可改用 search_private_manager。"
        return err


if __name__ == "__main__":
    mcp.run(transport="stdio")
