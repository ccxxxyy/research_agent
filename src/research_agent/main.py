"""FastAPI Application 入口。"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# 嵌入模型已缓存在本地，强制离线模式防止联网检查 huggingface.co 超时。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from research_agent.api.routes import (  # noqa: E402
    a2a,
    conversations,
    health,
    knowledge,
    memory,
    sentiment,
    supervisor,
    usage,
    watchlist,
)
from research_agent.config import get_settings  # noqa: E402
from research_agent.market.dashboard_extras import (  # noqa: E402
    fetch_cn_etf_panel,
    fetch_cn_futures_panel,
    fetch_cn_qdii_panel,
    fetch_us_etf_rank_panel,
    fetch_us_futures_panel,
    fetch_us_mutual_funds_panel,
)
from research_agent.market.theme_panels import (  # noqa: E402
    build_mainline_themes,
    build_sentiment_benchmark,
    build_speculative_pool,
)
from research_agent.market.us_theme_panels import (  # noqa: E402
    build_us_intraday_moves,
    build_us_mainline_themes,
    build_us_sentiment,
    build_us_speculative,
)
from research_agent.observability.logging import setup_logging  # noqa: E402

# LangSmith tracer 在 stream_mode=["messages","updates"] + subgraphs 下
# 会打出大量 "No indexed run ID" 警告——这是已知兼容问题，不影响功能。
logging.getLogger("langchain_core.tracers.langchain").setLevel(logging.CRITICAL)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


async def _try_build_research_supervisor(model_router, checkpointer, settings=None):
    """尽力编译研究 supervisor。

    工具发现并行运行若干加载器：三个 MCP stdio 子进程（fin_data、pdf_report、code）以及进程内的 knowledge-tools 加载器。
    若任一失败（缺依赖、网络超时等），会优雅降级：
    仍可接入已成功发现的 specialist；仅当所有 specialist 的工具发现均失败时，``get_research_supervisor_graph`` 才返回 503。后端不可用也不会导致启动失败。

    返回 ``(compiled_graph, specialist_roster)`` — roster 为已成功接入的 specialist 名称列表（例如 ``["data_expert", "news_expert"]``）。
    若未能加载任何 specialist 则返回 ``(None, [])``。
    """
    from research_agent.graph.research_supervisor import build_research_supervisor
    from research_agent.mcp_servers.client_factory import (
        load_code_server_tools,
        load_derivatives_server_tools,
        load_fin_data_server_tools,
        load_fund_server_tools,
        load_knowledge_tools_inproc,
        load_news_sentiment_server_tools,
        load_news_server_tools,
        load_pdf_report_server_tools,
        load_us_data_server_tools,
        load_us_filing_server_tools,
        load_us_news_server_tools,
        load_us_sentiment_server_tools,
    )

    # 说明：``load_knowledge_tools_inproc`` 是进程内替代方案，取代（已弃用的） MCP stdio ``load_knowledge_server_tools``。
    # 其余加载器仍拉起 MCP 子进程 —— 那些服务器的 import 链较轻，stdio 路径稳定。
    # knowledge 为何特殊见 ``knowledge_server.py``。
    timeout = float(getattr(settings, "mcp_tool_discovery_timeout", 30.0))
    results = await asyncio.gather(
        asyncio.wait_for(load_fin_data_server_tools(), timeout=timeout),
        asyncio.wait_for(load_pdf_report_server_tools(), timeout=timeout),
        asyncio.wait_for(load_code_server_tools(), timeout=timeout),
        asyncio.wait_for(load_knowledge_tools_inproc(), timeout=timeout),
        asyncio.wait_for(load_news_server_tools(), timeout=timeout),
        asyncio.wait_for(load_news_sentiment_server_tools(), timeout=timeout),
        asyncio.wait_for(load_fund_server_tools(), timeout=timeout),
        asyncio.wait_for(load_derivatives_server_tools(), timeout=timeout),
        asyncio.wait_for(load_us_data_server_tools(), timeout=timeout),
        asyncio.wait_for(load_us_filing_server_tools(), timeout=timeout),
        asyncio.wait_for(load_us_news_server_tools(), timeout=timeout),
        asyncio.wait_for(load_us_sentiment_server_tools(), timeout=timeout),
        return_exceptions=True,
    )
    names = (
        "fin_data_server",
        "pdf_report_server",
        "code_server",
        "knowledge_tools_inproc",
        "news_server",
        "news_sentiment_server",
        "fund_server",
        "derivatives_server",
        "us_data_server",
        "us_filing_server",
        "us_news_server",
        "us_sentiment_server",
    )
    tools: dict[str, list] = {}
    for name, r in zip(names, results, strict=False):
        if isinstance(r, Exception):
            logger.warning("Tool discovery failed for {}: {}", name, r)
            tools[name] = []
        else:
            tools[name] = list(r)
            logger.info("Tools discovered for {}: {}", name, len(tools[name]))

    if not any(tools.values()):
        logger.error(
            "All tool sources failed to provide tools; research supervisor will be unavailable."
        )
        return None, []

    tool_source_to_specialist = {
        "fin_data_server": "data_expert",
        "pdf_report_server": "report_expert",
        "code_server": "coder_expert",
        "knowledge_tools_inproc": "knowledge_expert",
        "news_server": "news_expert",
        "news_sentiment_server": "sentiment_expert",
        "fund_server": "fund_expert",
        "derivatives_server": "derivatives_expert",
        "us_data_server": "us_data_expert",
        "us_filing_server": "us_filing_expert",
        "us_news_server": "us_news_expert",
        "us_sentiment_server": "us_sentiment_expert",
    }
    roster = [spec for src, spec in tool_source_to_specialist.items() if tools.get(src)]

    # ``settings`` 可选以便单独做单测；生产 lifespan 中总会传入。
    reflect = bool(getattr(settings, "reflection_enabled", False))
    pass_threshold = float(getattr(settings, "reflection_pass_threshold", 0.85))
    max_iter = int(getattr(settings, "reflection_max_iterations", 2))
    hitl = bool(getattr(settings, "hitl_enabled", False))

    try:
        graph = build_research_supervisor(
            model_router=model_router,
            data_tools=tools["fin_data_server"] or None,
            us_data_tools=tools["us_data_server"] or None,
            us_filing_tools=tools["us_filing_server"] or None,
            us_news_tools=tools["us_news_server"] or None,
            us_sentiment_tools=tools["us_sentiment_server"] or None,
            report_tools=tools["pdf_report_server"] or None,
            coder_tools=tools["code_server"] or None,
            knowledge_tools=tools["knowledge_tools_inproc"] or None,
            news_tools=tools["news_server"] or None,
            sentiment_tools=tools["news_sentiment_server"] or None,
            fund_tools=tools["fund_server"] or None,
            derivatives_tools=tools["derivatives_server"] or None,
            checkpointer=checkpointer,
            enable_reflection=reflect,
            reflection_pass_threshold=pass_threshold,
            reflection_max_iterations=max_iter,
            enable_hitl=hitl,
        )
        return graph, roster
    except Exception:  # noqa: BLE001
        # 此处崩溃（例如模型路由配置错误）不应拖垮整个 API —— minimal supervisor 与 RAG 流水线仍可能可用。
        logger.exception("Failed to compile research_supervisor; route will 503.")
        return None, []


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """初始化并释放 Application 资源。"""
    settings = get_settings()
    setup_logging(
        settings.observability.log_level,
        log_file_path=settings.observability.log_file_path,
    )

    from research_agent.observability.tracing import setup_tracing

    setup_tracing(settings.observability)

    from research_agent.memory.checkpointer import init_checkpointer
    from research_agent.memory.store import init_memory_store

    checkpoint_sqlite = settings.checkpoint_sqlite_path.strip()
    checkpoint_sqlite_arg: Path | str | None = checkpoint_sqlite if checkpoint_sqlite else None
    store_sqlite = settings.memory_store_sqlite_path.strip()
    store_sqlite_arg: Path | str | None = store_sqlite if store_sqlite else None

    checkpointer = await init_checkpointer(
        settings.database.postgres_sync_uri,
        sqlite_path=checkpoint_sqlite_arg,
    )
    memory_store = await init_memory_store(
        settings.database.postgres_sync_uri,
        sqlite_path=store_sqlite_arg,
    )

    from research_agent.graph.minimal_supervisor import build_minimal_supervisor
    from research_agent.llm.provider import ModelRouter

    model_router = ModelRouter(settings.llm)

    supervisor_graph = build_minimal_supervisor(
        model_router=model_router,
        checkpointer=checkpointer,
    )

    research_supervisor_graph, specialist_roster = await _try_build_research_supervisor(
        model_router=model_router,
        checkpointer=checkpointer,
        settings=settings,
    )

    from research_agent.observability.metrics import METRICS
    from research_agent.security.token_quota import TokenQuotaManager

    METRICS.set_specialists(specialist_roster)

    redis_client = None
    if settings.database.redis_url:
        try:
            import redis

            redis_client = redis.Redis.from_url(
                settings.database.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            redis_client.ping()
        except Exception:  # noqa: BLE001
            redis_client = None

    token_quota = TokenQuotaManager(
        daily_limit=settings.user_token_quota_daily,
        redis_client=redis_client,
    )

    app.state.token_quota = token_quota
    app.state.supervisor_graph = supervisor_graph
    app.state.research_supervisor_graph = research_supervisor_graph
    app.state.available_specialists = specialist_roster
    app.state.model_router = model_router
    app.state.memory_store = memory_store
    app.state.checkpointer = checkpointer
    app.state.settings = settings

    from research_agent.memory.conversation_store import ConversationStore
    from research_agent.memory.watchlist_store import WatchlistStore

    conv_db = getattr(settings, "conversation_sqlite_path", "./data/conversations.db")
    conv_store = ConversationStore(db_path=conv_db)
    app.state.conversation_store = conv_store
    wl_db = getattr(settings, "watchlist_sqlite_path", "./data/watchlist.db")
    app.state.watchlist_store = WatchlistStore(db_path=wl_db)

    yield

    # --- 优雅关闭：按相反顺序释放资源 ---
    logger.info("Shutting down: releasing resources...")

    async def _close_conn(owner_name: str, conn: object) -> None:
        """关闭类连接对象：若 ``close`` 为协程函数则 await。

         Postgres 连接池暴露同步 ``close``；
         ``aiosqlite.Connection`` 及若干 LangGraph 异步存储暴露异步 ``close`` —— 不带 ``await`` 调用会触发 ``RuntimeWarning``
        （「coroutine was never awaited」）且底层套接字可能泄漏。运行时识别类型并正确分发。
        """
        close_fn = getattr(conn, "close", None)
        if close_fn is None:
            return
        try:
            if asyncio.iscoroutinefunction(close_fn):
                await close_fn()
            else:
                close_fn()
            logger.info("{} closed.", owner_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error closing {}: {}", owner_name, exc)

    # 关闭 memory store 连接池 / 异步 sqlite 连接
    if hasattr(memory_store, "conn"):
        await _close_conn("Memory store connection", memory_store.conn)

    # 关闭 checkpointer 连接池 / sqlite 连接
    if hasattr(checkpointer, "conn"):
        await _close_conn("Checkpointer connection", checkpointer.conn)

    logger.info("Shutdown complete.")


def _parse_cors_origins(raw: str) -> list[str]:
    """解析逗号分隔的 CORS 源或通配符。"""
    if raw.strip() == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Research Agent",
        description="基于 LangGraph、MCP 与 Agentic RAG 的多智能体深度研究系统",
        version="0.1.0",
        lifespan=lifespan,
    )

    settings = get_settings()
    origins = _parse_cors_origins(settings.cors_origins)

    # 中间件栈（执行顺序自下而上）。最外层最先执行：RequestId → Metrics → RequestTimeout → Auth → RateLimit → CORS → 路由处理器。
    app.add_middleware(
        CORSMiddleware,  # type: ignore[arg-type]
        allow_origins=origins,
        allow_credentials=origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Thread-ID",
            "X-User-ID",
            "X-Market",
            "X-Market-Source",
            "X-Cache-Hit",
            "X-Cache-Domain",
        ],
    )

    from research_agent.api.middleware import (
        AuthMiddleware,
        RateLimitMiddleware,
        RequestIdMiddleware,
        RequestTimeoutMiddleware,
    )
    from research_agent.observability.metrics import MetricsMiddleware

    app.add_middleware(
        RequestTimeoutMiddleware,
        timeout_seconds=float(settings.http_request_timeout_seconds),
    )
    app.add_middleware(
        RateLimitMiddleware,
        max_rpm=settings.rate_limit_rpm,
        redis_url=settings.database.redis_url or None,
    )
    app.add_middleware(AuthMiddleware, secret_key=settings.api_secret_key)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RequestIdMiddleware)

    from research_agent.api.routes.usage import metrics_router

    app.include_router(health.router)
    app.include_router(metrics_router)
    app.include_router(usage.router)
    app.include_router(knowledge.router)
    app.include_router(memory.router)
    app.include_router(sentiment.router)
    app.include_router(supervisor.router)
    app.include_router(conversations.router)
    app.include_router(watchlist.router)
    app.include_router(a2a.router)

    # --- 热搜 API（轻量端点，供首页展示） ---
    import time as _time

    _trending_cache: dict = {"ts": 0, "data": None}
    _trending_ttl = 300  # 5 分钟缓存

    @app.get("/api/trending", tags=["trending"])
    async def get_trending(fresh: bool = False):
        """返回多源热搜榜，供首页展示。

        数据源（3 个）：
        1. 人气榜 — emappdata.eastmoney.com 搜索热度 + 新浪实时行情
        2. 飙升榜 — 同源数据，按历史排名升幅排序
        3. 热门话题 — 东方财富研报标题提取市场焦点

        ``fresh=True`` 时跳过缓存，强制重新拉取。
        """
        import asyncio

        now = _time.time()
        if not fresh and _trending_cache["data"] and now - _trending_cache["ts"] < _trending_ttl:
            return _trending_cache["data"]

        async def _em_rank_data():
            """一次拉取 100 条 EM 人气数据，人气榜和飙升榜共用。"""
            import requests

            try:
                url = "https://emappdata.eastmoney.com/stockrank/getAllCurrentList"
                payload = {
                    "appId": "appId01",
                    "globalId": "786e4c21-70dc-435a-93bb-38",
                    "marketType": "",
                    "pageNo": 1,
                    "pageSize": 100,
                }
                r = await asyncio.to_thread(
                    lambda: requests.post(
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=8,
                    )
                )
                return r.json().get("data", [])
            except Exception:
                return []

        async def _em_topics():
            """东方财富研报热词。"""
            import datetime

            import requests

            try:
                today = datetime.date.today()
                begin = (today - datetime.timedelta(days=3)).isoformat()
                url = "https://reportapi.eastmoney.com/report/list"
                params = {
                    "industryCode": "*",
                    "pageSize": 40,
                    "pageNo": 1,
                    "beginTime": begin,
                    "endTime": today.isoformat(),
                    "qType": 0,
                    "fields": "",
                    "p": 1,
                    "pageNum": 1,
                }
                r = await asyncio.to_thread(lambda: requests.get(url, params=params, timeout=8))
                reports = r.json().get("data", [])
                if not reports:
                    return []
                out = []
                seen = set()
                for rpt in reports:
                    title = rpt.get("title", "")
                    industry = rpt.get("industryName", "")
                    org = rpt.get("orgSName", "")
                    if not title or title in seen:
                        continue
                    seen.add(title)
                    stock_name = rpt.get("stockName", "")
                    stock_code = rpt.get("stockCode", "")
                    out.append(
                        {
                            "title": title,
                            "industry": industry,
                            "org": org,
                            "stock_name": stock_name,
                            "stock_code": stock_code,
                        }
                    )
                    if len(out) >= 15:
                        break
                return out
            except Exception:
                return []

        rank_task = asyncio.create_task(_em_rank_data())
        topic_task = asyncio.create_task(_em_topics())

        all_items = await rank_task
        rank_fetched_at = _time.strftime("%H:%M:%S")
        topics = await topic_task
        topics_fetched_at = _time.strftime("%H:%M:%S")

        # --- 人气榜 Top 15 ---
        em_hot = []
        if all_items:
            top_n = all_items[:15]
            codes = [it["sc"] for it in top_n]
            info = await asyncio.to_thread(_batch_stock_info_sina, codes)
            industries = await asyncio.to_thread(_batch_stock_industry_em, codes)
            rank_fetched_at = _time.strftime("%H:%M:%S")
            for it in top_n:
                sc = it["sc"]
                si = info.get(sc, {})
                code_bare = sc.replace("SZ", "").replace("SH", "").replace("BJ", "")
                em_hot.append(
                    {
                        "rank": it.get("rk", ""),
                        "name": si.get("name", code_bare),
                        "code": code_bare,
                        "price": si.get("price"),
                        "change_pct": si.get("change_pct"),
                        "industry": industries.get(code_bare.zfill(6), "")
                        or industries.get(code_bare, ""),
                    }
                )

        # --- 飙升榜：hisRc 最小（排名上升最多）的 15 只 ---
        surge = []
        if all_items:
            surged = sorted(
                [it for it in all_items if it.get("hisRc", 0) < 0],
                key=lambda x: x.get("hisRc", 0),
            )[:15]
            if surged:
                codes = [it["sc"] for it in surged]
                info = await asyncio.to_thread(_batch_stock_info_sina, codes)
                industries = await asyncio.to_thread(_batch_stock_industry_em, codes)
                rank_fetched_at = _time.strftime("%H:%M:%S")
                for i, it in enumerate(surged):
                    sc = it["sc"]
                    si = info.get(sc, {})
                    code_bare = sc.replace("SZ", "").replace("SH", "").replace("BJ", "")
                    surge.append(
                        {
                            "rank": i + 1,
                            "name": si.get("name", code_bare),
                            "code": code_bare,
                            "price": si.get("price"),
                            "change_pct": si.get("change_pct"),
                            "rank_change": abs(it.get("hisRc", 0)),
                            "industry": industries.get(code_bare.zfill(6), "")
                            or industries.get(code_bare, ""),
                        }
                    )

        result: dict = {
            "fetched_at": {
                "hot": rank_fetched_at,
                "topics": topics_fetched_at,
            },
        }
        if em_hot:
            result["eastmoney"] = {"label": "人气榜", "items": em_hot}
        if surge:
            result["surge"] = {"label": "飙升榜", "items": surge}
        if topics:
            result["topics"] = {"label": "热门话题", "items": topics}

        if result.get("eastmoney") or result.get("surge") or result.get("topics"):
            _trending_cache["ts"] = now
            _trending_cache["data"] = result
        return result

    def _batch_stock_info_sina(codes: list[str]) -> dict[str, dict]:
        """通过新浪批量接口获取股票名称和行情（极快，单次请求）。"""
        import re

        import requests

        sina_codes = []
        for c in codes:
            if c.startswith("SH"):
                sina_codes.append(f"sh{c[2:]}")
            elif c.startswith("SZ"):
                sina_codes.append(f"sz{c[2:]}")
            else:
                sina_codes.append(c.lower())
        url = f"https://hq.sinajs.cn/list={','.join(sina_codes)}"
        headers = {"Referer": "https://finance.sina.com.cn"}
        try:
            r = requests.get(url, headers=headers, timeout=5)
            r.encoding = "gbk"
        except Exception:
            return {}

        result = {}
        for line in r.text.strip().split("\n"):
            m = re.match(r'var hq_str_(s[hz]\d+)="(.+)"', line.strip())
            if not m:
                continue
            raw_code = m.group(1)
            fields = m.group(2).split(",")
            if len(fields) < 4:
                continue
            prefix = "SH" if raw_code.startswith("sh") else "SZ"
            code_key = f"{prefix}{raw_code[2:]}"
            try:
                price = float(fields[3]) if fields[3] else None
                yesterday = float(fields[2]) if fields[2] else None
                change_pct = (
                    round((price - yesterday) / yesterday * 100, 2)
                    if price and yesterday and yesterday > 0
                    else None
                )
            except (ValueError, ZeroDivisionError):
                price = None
                change_pct = None
            result[code_key] = {
                "name": fields[0],
                "price": price,
                "change_pct": change_pct,
            }
        return result

    def _batch_stock_industry_em(codes: list[str]) -> dict[str, str]:
        """批量查 A 股所属行业（东财 ulist ``f100``），补齐热搜等列表右侧行业。

        ``codes`` 可为 ``002156`` / ``SZ002156`` / ``SH600519``。
        """
        from urllib.parse import urlencode

        bare: list[str] = []
        for c in codes:
            s = str(c or "").strip().upper()
            s = s.removeprefix("SZ").removeprefix("SH").removeprefix("BJ")
            if s.isdigit():
                bare.append(s.zfill(6))
        bare = list(dict.fromkeys(bare))
        if not bare:
            return {}

        def _secid(code: str) -> str:
            # 6/5/9/688 → 沪市；其余深/京按深市 secid
            if code.startswith(("5", "6", "9")) or code.startswith("688"):
                return f"1.{code}"
            return f"0.{code}"

        secids = ",".join(_secid(c) for c in bare)
        params = {
            "fltt": "2",
            "secids": secids,
            "fields": "f12,f14,f100",
            "ut": "7eea3edcaed734bea9cbfc24409ed989",
        }
        hosts = (
            "https://push2delay.eastmoney.com/api/qt/ulist.np/get",
            "https://push2.eastmoney.com/api/qt/ulist.np/get",
            "https://88.push2.eastmoney.com/api/qt/ulist.np/get",
        )
        diff: list = []

        def _parse(payload: dict) -> list:
            data = (payload or {}).get("data") or {}
            return data.get("diff") or []

        try:
            from curl_cffi import requests as curl_requests

            qs = urlencode(params)
            for base in hosts:
                try:
                    resp = curl_requests.get(f"{base}?{qs}", impersonate="chrome", timeout=8)
                    if resp.status_code != 200:
                        continue
                    diff = _parse(resp.json())
                    if diff:
                        break
                except Exception:
                    continue
        except ImportError:
            pass

        if not diff:
            import requests

            sess = requests.Session()
            sess.trust_env = False
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0",
                    "Referer": "https://quote.eastmoney.com/",
                }
                for base in hosts:
                    try:
                        r = sess.get(base, params=params, timeout=8, headers=headers)
                        diff = _parse(r.json())
                        if diff:
                            break
                    except Exception:
                        continue
            finally:
                sess.close()

        out: dict[str, str] = {}
        for it in diff:
            code = str(it.get("f12") or "").zfill(6)
            industry = str(it.get("f100") or "").strip()
            if code and industry and industry not in {"-", "—", "null"}:
                out[code] = industry
        return out

    # 涨停池「所属行业」常见截断 → 完整东财行业名（f100 / 板块表失败时兜底）
    industry_prefix_fix: dict[str, str] = {
        "房地产开": "房地产开发",
        "互联网服": "互联网服务",
    }
    board_industry_names_cache: list[str] | None = None

    def _industry_board_names() -> list[str]:
        """东财行业板块全名列表。"""
        nonlocal board_industry_names_cache
        if board_industry_names_cache is not None:
            return board_industry_names_cache
        names: list[str] = []
        try:
            import akshare as ak

            df = ak.stock_board_industry_name_em()
            if df is not None and not df.empty:
                col = "板块名称" if "板块名称" in df.columns else str(df.columns[0])
                names = [str(x).strip() for x in df[col].tolist() if str(x).strip()]
        except Exception:
            names = []
        for v in industry_prefix_fix.values():
            if v and v not in names:
                names.append(v)
        board_industry_names_cache = names
        return names

    def _expand_truncated_industry(name: str) -> str:
        cur = (name or "").strip()
        if not cur:
            return ""
        fixed = industry_prefix_fix.get(cur)
        if fixed:
            return fixed
        boards = _industry_board_names()
        if cur in boards:
            return cur
        hits = [b for b in boards if b.startswith(cur) and len(b) > len(cur)]
        if not hits:
            return cur
        hits.sort(key=len)
        return hits[0]

    def _enrich_pool_industries(items: list[dict]) -> None:
        """覆盖涨停池截断行业：优先 ulist f100，其次板块名前缀补全。"""
        if not items:
            return
        codes = [str(it.get("code") or "") for it in items]
        industries = _batch_stock_industry_em(codes)
        for it in items:
            code = str(it.get("code") or "").zfill(6)
            full = ""
            if industries:
                full = industries.get(code, "") or industries.get(str(it.get("code") or ""), "")
            cur = str(it.get("industry") or "").strip()
            it["industry"] = (full or _expand_truncated_industry(cur) or cur).strip()

    # --- 行情看板 API ---
    # 不再做服务端 TTL 缓存：首页每次刷新都拉取最新数据，并由 fetched_at 如实标注。

    @app.get("/api/dashboard", tags=["dashboard"])
    async def get_dashboard(fresh: bool = False):
        """聚合首页行情看板数据（每次实时拉取）。

        数据源：新浪实时指数 + EM 人气榜 + EM 涨停池 + EM 研报 + 市场状态。
        响应含 ``fetched_at``：各板块数据实际拉取完成时刻（本地 ``HH:MM:SS``）。
        ``fresh=true`` 时跳过美股 90s 短缓存，强制重拉 Yahoo。
        """

        def _hms() -> str:
            return _time.strftime("%H:%M:%S")

        async def _timed_thread(fn, *, timeout: float = 20.0, default: object = None):
            """看板子任务硬超时：单源卡住时不再拖死整页（此前可达 40–60s）。"""
            label = getattr(fn, "__name__", None) or repr(fn)[:80]
            try:
                data = await asyncio.wait_for(asyncio.to_thread(fn), timeout=timeout)
                return data, _hms()
            except TimeoutError:
                logger.warning("dashboard task timed out ({}s): {}", timeout, label)
                return default, _hms()
            except Exception as exc:  # noqa: BLE001
                logger.warning("dashboard task failed ({}): {}", label, exc)
                return default, _hms()

        _empty_boards: dict = {"industry": [], "concept": [], "concept_all": []}
        idx_task = asyncio.create_task(_timed_thread(_fetch_indices_sina, default=[]))
        # 多拉涨停供主线/情绪/妖股聚合；面板仍截断为 Top20
        zt_task = asyncio.create_task(_timed_thread(lambda: _fetch_zt_pool(limit=80), default=[]))
        extra_task = asyncio.create_task(_timed_thread(_fetch_extra_pools, default={}))
        boards_task = asyncio.create_task(
            _timed_thread(_fetch_boards, timeout=18.0, default=_empty_boards)
        )
        changes_task = asyncio.create_task(_timed_thread(_fetch_changes, default=[]))
        lhb_task = asyncio.create_task(_timed_thread(_fetch_lhb, default=[]))
        tech_task = asyncio.create_task(_timed_thread(_fetch_tech_stocks, default=[]))
        status_task = asyncio.create_task(_timed_thread(_fetch_market_status, default={}))

        async def _trending_safe():
            try:
                return await asyncio.wait_for(get_trending(fresh=True), timeout=20.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("dashboard trending failed: {}", exc)
                return {}

        trending_task = asyncio.create_task(_trending_safe())
        us_task = asyncio.create_task(
            _timed_thread(
                lambda: _get_us_dashboard_cached(force=fresh),
                timeout=25.0,
                default={},
            )
        )
        # 期货/基金/ETF 双榜走 /api/dashboard/extras，避免拖慢整页

        indices, indices_at = await idx_task
        zt_full, zt_at = await zt_task
        extra_pools, extra_at = await extra_task
        boards, boards_at = await boards_task
        changes, changes_at = await changes_task
        lhb, lhb_at = await lhb_task
        tech_stocks, tech_at = await tech_task
        market_status, status_at = await status_task
        trending = await trending_task
        us_dash, us_bundle_at = await us_task
        trending_fa = (trending or {}).get("fetched_at") or {}
        us_fa = (us_dash or {}).get("fetched_at") or {}

        zt_pool = (zt_full or [])[:20]
        concept_all = (boards or {}).get("concept_all") or (boards or {}).get("concept") or []
        mainline_themes = build_mainline_themes(concept_all, zt_full or [])
        sentiment_benchmark = build_sentiment_benchmark(zt_full or [])
        speculative_pool = build_speculative_pool(zt_full or [], lhb or [], changes or [])
        boards_out = {
            "industry": (boards or {}).get("industry") or [],
            "concept": (boards or {}).get("concept") or [],
        }

        breadth = _compute_breadth(indices, zt_pool)
        breadth_at = _hms()
        updated_at = _hms()
        _empty_rank = {"by_volume": [], "by_change": [], "limit": 10, "source": ""}

        # 强势/昨涨停/炸板在同一函数内串行拉取，完成时刻用 extra_at；
        # 行业/概念同属 boards 一次请求。
        fetched_at = {
            "indices": indices_at,
            "breadth": breadth_at,
            "zt_pool": zt_at,
            "trending": trending_fa.get("hot") or updated_at,
            "topics": trending_fa.get("topics") or updated_at,
            "strong_pool": extra_at,
            "previous_zt": extra_at,
            "zbgc_pool": extra_at,
            "boards_industry": boards_at,
            "boards_concept": boards_at,
            "mainline_themes": boards_at,
            "sentiment_benchmark": zt_at,
            "speculative_pool": lhb_at,
            "changes": changes_at,
            "lhb": lhb_at,
            "tech_stocks": tech_at,
            "market_status": status_at,
            # 美股各子板块时间（缺失时回退到美股整包完成时刻）
            "us_indices": us_fa.get("indices") or us_bundle_at,
            "us_breadth": us_fa.get("breadth") or us_bundle_at,
            "us_gainers": us_fa.get("gainers") or us_bundle_at,
            "us_actives": us_fa.get("actives") or us_bundle_at,
            "us_growth": us_fa.get("growth") or us_bundle_at,
            "us_sectors": us_fa.get("sectors") or us_bundle_at,
            "us_theme_etfs": us_fa.get("theme_etfs") or us_bundle_at,
            "us_small_gainers": us_fa.get("small_gainers") or us_bundle_at,
            "us_mega": us_fa.get("mega") or us_bundle_at,
            "us_undervalued": us_fa.get("undervalued") or us_bundle_at,
            "us_losers": us_fa.get("losers") or us_bundle_at,
            "us_shorted": us_fa.get("shorted") or us_bundle_at,
            "us_mainline_themes": us_fa.get("mainline_themes") or us_bundle_at,
            "us_intraday_moves": us_fa.get("intraday_moves") or us_bundle_at,
            "us_sentiment": us_fa.get("sentiment") or us_bundle_at,
            "us_speculative": us_fa.get("speculative") or us_bundle_at,
            "us_market_status": us_fa.get("market_status") or us_bundle_at,
        }

        return {
            "market_status": market_status,
            "indices": indices,
            "zt_pool": zt_pool,
            "strong_pool": extra_pools.get("strong", []),
            "previous_zt": extra_pools.get("previous", []),
            "zbgc_pool": extra_pools.get("zbgc", []),
            "boards": boards_out,
            "mainline_themes": mainline_themes,
            "sentiment_benchmark": sentiment_benchmark,
            "speculative_pool": speculative_pool,
            "changes": changes,
            "lhb": lhb,
            "tech_stocks": tech_stocks,
            # 占位；前端另拉 /api/dashboard/extras
            "cn_futures": dict(_empty_rank),
            "cn_etf": dict(_empty_rank),
            "cn_qdii": {**_empty_rank, "limit": 15},
            "breadth": breadth,
            "trending": trending,
            "us": us_dash,
            "updated_at": updated_at,
            "fetched_at": fetched_at,
            "extras_pending": True,
        }

    # 期货/基金等慢板块短缓存（独立于主看板）
    _dash_extras_cache: dict = {"ts": 0.0, "data": None}
    _dash_extras_ttl = 300.0  # 与前端慢板块 5 分钟刷新对齐

    def _fetch_dashboard_extras() -> dict:
        """CN/US 期货·ETF·共同基金双榜（可慢，不进主看板关键路径）。"""
        import concurrent.futures

        fetched_at: dict[str, str] = {}

        def _stamp(key: str) -> None:
            fetched_at[key] = _time.strftime("%H:%M:%S")

        def _cn_fut():
            out = fetch_cn_futures_panel(limit=10)
            _stamp("cn_futures")
            return out

        def _cn_etf():
            out = fetch_cn_etf_panel(limit=10)
            _stamp("cn_etf")
            return out

        def _cn_qdii():
            out = fetch_cn_qdii_panel(limit=15)
            _stamp("cn_qdii")
            return out

        def _us_fut():
            out = fetch_us_futures_panel(limit=10)
            _stamp("us_futures")
            return out

        def _us_etf():
            out = fetch_us_etf_rank_panel(limit=10)
            _stamp("us_etf_rank")
            return out

        def _us_mf():
            out = fetch_us_mutual_funds_panel(limit=10)
            _stamp("us_mutual_funds")
            return out

        empty = {"by_volume": [], "by_change": [], "limit": 10, "source": ""}
        results = {
            "cn_futures": dict(empty),
            "cn_etf": dict(empty),
            "cn_qdii": {**empty, "limit": 15},
            "us_futures": dict(empty),
            "us_etf_rank": dict(empty),
            "us_mutual_funds": dict(empty),
        }
        jobs = {
            "cn_futures": _cn_fut,
            "cn_etf": _cn_etf,
            "cn_qdii": _cn_qdii,
            "us_futures": _us_fut,
            "us_etf_rank": _us_etf,
            "us_mutual_funds": _us_mf,
        }
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futs = {pool.submit(fn): key for key, fn in jobs.items()}
            for fut in concurrent.futures.as_completed(futs):
                key = futs[fut]
                try:
                    results[key] = fut.result()
                except Exception:  # noqa: BLE001
                    logger.exception("dashboard extras %s failed", key)
        return {**results, "fetched_at": fetched_at}

    def _get_dashboard_extras_cached(*, force: bool = False) -> dict:
        now = _time.time()
        cached = _dash_extras_cache.get("data")
        if (
            not force
            and cached is not None
            and now - float(_dash_extras_cache.get("ts") or 0) < _dash_extras_ttl
        ):
            return cached
        data = _fetch_dashboard_extras()
        _dash_extras_cache["ts"] = now
        _dash_extras_cache["data"] = data
        return data

    @app.get("/api/dashboard/extras", tags=["dashboard"])
    async def get_dashboard_extras(fresh: bool = False):
        """慢板块：国内/美股期货·ETF·共同基金双榜（与主看板解耦）。"""

        def _run():
            return _get_dashboard_extras_cached(force=fresh)

        return await asyncio.to_thread(_run)

    # 美股看板短缓存：避免 30s 自动刷新反复打爆 Yahoo（表现为 possibly delisted）
    _us_dash_cache: dict = {"ts": 0.0, "data": None}
    _us_dash_ttl = 90.0
    # 东财美股涨跌家数全量扫描较慢，单独短缓存，避免每次刷新打满 clist
    _us_breadth_cache: dict = {"ts": 0.0, "data": None}
    _us_breadth_ttl = 300.0

    def _fetch_us_dashboard() -> dict:
        """美股首页看板聚合（yfinance，与 A 股平行、不混用）。

        对应 A 股面板：
        指数 / 涨跌概览 / 涨幅榜 / 最活跃 / 成长科技 /
        行业 ETF / 主题 ETF / 小盘异动 / 七巨头 / 低估值大盘 /
        跌幅榜 / 空头最重 / 市场状态；
        以及聚合面板：主线题材 / 日内异动 / 情绪标杆 / 投机·拥挤。
        """
        import io
        import logging
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from contextlib import redirect_stderr, redirect_stdout

        import yfinance as yf

        # yfinance 在限流时会刷屏 "possibly delisted
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        logging.getLogger("peewee").setLevel(logging.CRITICAL)

        fetched_at: dict[str, str] = {}

        def _stamp(key: str) -> None:
            fetched_at[key] = _time.strftime("%H:%M:%S")

        def _quote_one(symbol: str, name: str, *, period: str = "5d") -> dict | None:
            """与 ``us_data_server._quote_from_ticker`` 同源，避免看板与问答涨跌幅口径不一致。

            ``period`` 保留签名兼容，昨收已由 chart 日线 / fast_info 统一处理。
            """
            del period  # 口径统一后不再使用独立 history 窗口
            try:
                from research_agent.mcp_servers.us_data_server import _quote_from_ticker

                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    q = _quote_from_ticker(symbol)
                price = q.get("price")
                if price is None:
                    return None
                prev = q.get("previous_close")
                change = q.get("change")
                if change is None and price is not None and prev is not None:
                    change = float(price) - float(prev)
                change_pct = q.get("change_percent")
                # 东财代理 ETF（IWM/VIXY）时用返回名，避免仍显示「罗素2000/VIX」
                display_name = str(q.get("name") or name) if q.get("proxy") else name
                return {
                    "code": symbol.lstrip("^"),
                    "symbol": symbol,
                    "name": display_name,
                    "price": round(float(price), 2),
                    "change": round(float(change), 2) if change is not None else None,
                    "change_pct": round(float(change_pct), 2) if change_pct is not None else None,
                    "price_source": str(q.get("source") or "us_data_server"),
                    "proxy": bool(q.get("proxy")),
                }
            except Exception:
                return None

        # 常见美股中英名；行业/主题 ETF 与筛选榜共用（_screen_list 闭包读取）
        us_cn_names: dict[str, str] = {}

        def _batch_quotes(pairs: list[tuple[str, str]], *, period: str = "5d") -> list[dict]:
            if not pairs:
                return []
            out: list[dict] = []
            # 控制并发，避免再次触发 Yahoo 限流
            with ThreadPoolExecutor(max_workers=3) as pool:
                futs = {
                    pool.submit(_quote_one, sym, name, period=period): sym for sym, name in pairs
                }
                for fut in as_completed(futs):
                    item = fut.result()
                    if item:
                        out.append(item)
            # 保持与输入相近的展示顺序
            order = {sym: i for i, (sym, _) in enumerate(pairs)}
            out.sort(key=lambda x: order.get(x.get("symbol", ""), 999))
            return out

        def _screen_list_eastmoney(
            *,
            sort: str,
            reverse: bool = True,
            limit: int = 10,
        ) -> tuple[list[dict], int]:
            """Yahoo screen 失败时：东财美股榜（NASDAQ+NYSE）按涨跌幅/成交额排序。"""
            from urllib.parse import urlencode

            # f3=涨跌幅 f5=成交量 f6=成交额
            fid = {"change": "f3", "volume": "f5", "amount": "f6"}.get(sort, "f3")
            po = "1" if reverse else "0"
            params = {
                "pn": "1",
                "pz": "80",
                "po": po,
                "np": "1",
                "fltt": "2",
                "invt": "2",
                "fid": fid,
                "fs": "m:105,m:106",
                "fields": "f12,f14,f2,f3,f4,f5,f6,f18",
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            }
            hosts = (
                "https://push2delay.eastmoney.com/api/qt/clist/get",
                "https://push2.eastmoney.com/api/qt/clist/get",
            )
            diff: list = []
            try:
                from curl_cffi import requests as curl_requests

                qs = urlencode(params)
                for base in hosts:
                    try:
                        resp = curl_requests.get(f"{base}?{qs}", impersonate="chrome", timeout=10)
                        if resp.status_code != 200:
                            continue
                        data = resp.json().get("data") or {}
                        diff = data.get("diff") or []
                        if diff:
                            break
                    except Exception:
                        continue
            except ImportError:
                pass
            items: list[dict] = []
            for row in diff:
                sym = str(row.get("f12") or "").strip().upper()
                if not sym or "_" in sym or sym.endswith("W") or len(sym) > 5:
                    continue
                try:
                    price = float(row.get("f2"))
                    chg = float(row.get("f3"))
                except (TypeError, ValueError):
                    continue
                if price < 1.0:  # 过滤仙股/权证噪声
                    continue
                items.append(
                    {
                        "code": sym,
                        "symbol": sym,
                        "name": us_cn_names.get(sym) or str(row.get("f14") or sym),
                        "price": price,
                        "change": row.get("f4"),
                        "change_pct": chg,
                        "volume": row.get("f5"),
                        "market_cap": None,
                        "price_source": "eastmoney_us",
                    }
                )
                if len(items) >= limit:
                    break
            return items, len(items)

        def _eastmoney_us_breadth() -> dict:
            """全量统计东财美股普通股涨/跌/平家数。

            对 ``m:105+t:1,m:106+t:1``（NASDAQ/NYSE 普通股）并行翻完所有页再计数。
            """
            from math import ceil
            from urllib.parse import urlencode

            now = _time.time()
            cached = _us_breadth_cache.get("data")
            if (
                cached is not None
                and now - float(_us_breadth_cache.get("ts") or 0) < _us_breadth_ttl
            ):
                return cached

            hosts = (
                "https://push2delay.eastmoney.com/api/qt/clist/get",
                "https://push2.eastmoney.com/api/qt/clist/get",
            )
            try:
                from curl_cffi import requests as curl_requests
            except ImportError:
                return {"up": 0, "down": 0, "flat": 0, "scanned": 0, "universe": 0}

            def _fetch_page(pn: int) -> tuple[int, list]:
                params = {
                    "pn": str(pn),
                    "pz": "100",
                    "po": "0",
                    "np": "1",
                    "fltt": "2",
                    "invt": "2",
                    "fid": "f12",
                    # 只要普通股，排除权证/结构化噪音
                    "fs": "m:105+t:1,m:106+t:1",
                    "fields": "f12,f2,f3",
                    "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                }
                qs = urlencode(params)
                for base in hosts:
                    try:
                        resp = curl_requests.get(f"{base}?{qs}", impersonate="chrome", timeout=12)
                        if resp.status_code != 200:
                            continue
                        data = resp.json().get("data") or {}
                        diff = data.get("diff") or []
                        total = int(data.get("total") or 0)
                        if diff or total:
                            return total, diff
                    except Exception:
                        continue
                return 0, []

            universe, first_diff = _fetch_page(1)
            if not first_diff and universe <= 0:
                return {"up": 0, "down": 0, "flat": 0, "scanned": 0, "universe": 0}

            page_count = max(1, ceil(universe / 100)) if universe else 1
            # 安全上限，防止异常 total 打爆
            page_count = min(page_count, 80)
            pages: dict[int, list] = {1: first_diff}
            if page_count > 1:
                with ThreadPoolExecutor(max_workers=8) as pool:
                    futs = {pool.submit(_fetch_page, pn): pn for pn in range(2, page_count + 1)}
                    for fut in as_completed(futs):
                        pn = futs[fut]
                        try:
                            _, diff = fut.result()
                            pages[pn] = diff
                        except Exception:
                            pages[pn] = []

            up = down = flat = 0
            for pn in range(1, page_count + 1):
                for row in pages.get(pn) or []:
                    sym = str(row.get("f12") or "").strip().upper()
                    if not sym or "_" in sym or len(sym) > 5:
                        continue
                    try:
                        price = float(row.get("f2"))
                        chg = float(row.get("f3"))
                    except (TypeError, ValueError):
                        continue
                    if price < 1.0:
                        continue
                    if chg > 0:
                        up += 1
                    elif chg < 0:
                        down += 1
                    else:
                        flat += 1

            result = {
                "up": up,
                "down": down,
                "flat": flat,
                "scanned": up + down + flat,
                "universe": universe,
            }
            # 结果明显异常（例如涨跌完全相同且等于扫描数的一半）时不缓存，便于下次重试
            if up + down > 100 and up != down:
                _us_breadth_cache["ts"] = now
                _us_breadth_cache["data"] = result
            return result

        def _shorted_watchlist_em(*, limit: int = 10) -> list[dict]:
            """Yahoo 空头榜不可用时：常见高空头关注标的 + 东财报价（非官方 short interest）。

            顺序拉取，避免嵌套 ThreadPool（外层 jobs 已在线程池中）死锁。
            """
            watch = [
                ("GME", "GameStop (GME)"),
                ("AMC", "AMC (AMC)"),
                ("BYND", "Beyond Meat (BYND)"),
                ("CVNA", "Carvana (CVNA)"),
                ("UPST", "Upstart (UPST)"),
                ("SOFI", "SoFi (SOFI)"),
                ("PLUG", "Plug Power (PLUG)"),
                ("RIOT", "Riot (RIOT)"),
                ("MARA", "Marathon (MARA)"),
                ("HOOD", "Robinhood (HOOD)"),
                ("SNAP", "Snap (SNAP)"),
                ("DKNG", "DraftKings (DKNG)"),
            ]
            items: list[dict] = []
            for sym, name in watch:
                item = _quote_one(sym, name)
                if not item:
                    continue
                item["price_source"] = "eastmoney_us"
                item["note"] = "常见高空头关注标的（Yahoo 空头榜不可用）"
                items.append(item)
            items.sort(
                key=lambda x: abs(x["change_pct"] if x.get("change_pct") is not None else 0),
                reverse=True,
            )
            return items[:limit]

        def _screen_list(preset: str, *, limit: int = 10) -> tuple[list[dict], int]:
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    raw = yf.screen(preset, count=limit)
            except Exception:
                raw = None
            quotes = (raw or {}).get("quotes") or []
            if quotes:
                total = int((raw or {}).get("total") or len(quotes) or 0)
                items: list[dict] = []
                for q in quotes[:limit]:
                    sym = q.get("symbol") or ""
                    if not sym:
                        continue
                    en = q.get("shortName") or q.get("longName") or sym
                    items.append(
                        {
                            "code": sym,
                            "symbol": sym,
                            "name": us_cn_names.get(sym) or en,
                            "price": q.get("regularMarketPrice"),
                            "change": q.get("regularMarketChange"),
                            "change_pct": q.get("regularMarketChangePercent"),
                            "volume": q.get("regularMarketVolume"),
                            "market_cap": q.get("marketCap"),
                            "fifty_two_week_high": q.get("fiftyTwoWeekHigh"),
                            "price_source": "yahoo_screen",
                        }
                    )
                return items, total

            # Yahoo screen 被 403/限流：东财美股榜 / 空头关注列表兜底
            if preset == "most_shorted_stocks":
                items = _shorted_watchlist_em(limit=limit)
                return items, len(items)
            em_map = {
                "day_gainers": ("change", True),
                "day_losers": ("change", False),
                "most_actives": ("amount", True),
                "growth_technology_stocks": ("change", True),
                "small_cap_gainers": ("change", True),
                "undervalued_large_caps": ("change", True),
            }
            if preset not in em_map:
                return [], 0
            sort_key, reverse = em_map[preset]
            return _screen_list_eastmoney(sort=sort_key, reverse=reverse, limit=limit)

        # --- 市场状态 ---
        market_status: dict = {"status": "unknown", "message": "美股状态获取失败"}
        try:
            from research_agent.mcp_servers.us_data_server import _session_status

            st = _session_status()
            labels = {
                "open": "交易中",
                "pre_market": "盘前",
                "after_hours": "盘后",
                "closed": "已收盘",
            }
            market_status = {
                "status": st.get("status") or "unknown",
                "session": st.get("session"),
                "message": st.get("hint") or "",
                "label": labels.get(st.get("status") or "", st.get("status") or ""),
                "local_time": st.get("local_time"),
                "local_date": st.get("local_date"),
                "local_weekday": st.get("local_weekday"),
                "local_display": st.get("local_display"),
                "timezone": st.get("timezone") or "America/New_York",
            }
            try:
                ym = yf.Market("US").status or {}
                if ym.get("message"):
                    market_status["yahoo_message"] = ym.get("message")
                    if not market_status["message"]:
                        market_status["message"] = ym.get("message")
            except Exception:
                pass
        except Exception:
            pass
        _stamp("market_status")

        index_pairs = [
            ("^GSPC", "标普500 (S&P 500)"),
            ("^DJI", "道琼斯 (Dow 30)"),
            ("^IXIC", "纳斯达克 (Nasdaq)"),
            ("^NDX", "纳指100 (Nasdaq 100)"),
            ("^RUT", "罗素2000 (Russell 2000)"),
            ("^VIX", "VIX恐慌 (VIX)"),
        ]
        sector_pairs = [
            ("XLK", "科技 (XLK)"),
            ("XLF", "金融 (XLF)"),
            ("XLE", "能源 (XLE)"),
            ("XLV", "医疗 (XLV)"),
            ("XLI", "工业 (XLI)"),
            ("XLY", "可选消费 (XLY)"),
            ("XLP", "必选消费 (XLP)"),
            ("XLU", "公用事业 (XLU)"),
            ("XLB", "材料 (XLB)"),
            ("XLRE", "房地产 (XLRE)"),
            ("XLC", "通信服务 (XLC)"),
        ]
        theme_pairs = [
            ("QQQ", "纳指ETF (QQQ)"),
            ("SPY", "标普ETF (SPY)"),
            ("IWM", "罗素2000ETF (IWM)"),
            ("SMH", "半导体 (SMH)"),
            ("SOXX", "芯片 (SOXX)"),
            ("BOTZ", "机器人/AI (BOTZ)"),
            ("ARKK", "ARK创新 (ARKK)"),
            ("XBI", "生物科技 (XBI)"),
            ("IBIT", "比特币现货 (IBIT)"),
            ("GLD", "黄金 (GLD)"),
            ("TLT", "长债 (TLT)"),
            ("HYG", "高收益债 (HYG)"),
        ]
        mega_pairs = [
            ("AAPL", "苹果 (AAPL)"),
            ("MSFT", "微软 (MSFT)"),
            ("NVDA", "英伟达 (NVDA)"),
            ("AMZN", "亚马逊 (AMZN)"),
            ("GOOGL", "谷歌 (GOOGL)"),
            ("META", "Meta (META)"),
            ("TSLA", "特斯拉 (TSLA)"),
        ]
        # 涨跌/活跃等筛选榜常见票：补中英名，避免只显示 Yahoo 英文 shortName
        us_cn_names.clear()
        us_cn_names.update(
            {sym: name for pairs in (sector_pairs, theme_pairs, mega_pairs) for sym, name in pairs}
        )
        us_cn_names.update(
            {
                "AMD": "超威 (AMD)",
                "AVGO": "博通 (AVGO)",
                "INTC": "英特尔 (INTC)",
                "NFLX": "奈飞 (NFLX)",
                "CRM": "Salesforce (CRM)",
                "ORCL": "甲骨文 (ORCL)",
                "BABA": "阿里巴巴 (BABA)",
                "PDD": "拼多多 (PDD)",
                "JD": "京东 (JD)",
                "NIO": "蔚来 (NIO)",
                "XPEV": "小鹏 (XPEV)",
                "LI": "理想 (LI)",
                "COIN": "Coinbase (COIN)",
                "PLTR": "Palantir (PLTR)",
                "SOFI": "SoFi (SOFI)",
                "RIVN": "Rivian (RIVN)",
                "UBER": "优步 (UBER)",
                "ABNB": "爱彼迎 (ABNB)",
            }
        )

        def _sort_by_chg(items: list[dict]) -> list[dict]:
            items.sort(
                key=lambda x: -(x["change_pct"] if x.get("change_pct") is not None else -9999)
            )
            return items

        # 指数 / 行业 / 主题 / 七巨头 / 七个筛选榜并行，缩短看板等待
        jobs = {
            "indices": lambda: _batch_quotes(index_pairs),
            "sectors": lambda: _sort_by_chg(_batch_quotes(sector_pairs)),
            "theme_etfs": lambda: _sort_by_chg(_batch_quotes(theme_pairs)),
            "mega": lambda: _sort_by_chg(_batch_quotes(mega_pairs)),
            "gainers": lambda: _screen_list("day_gainers", limit=10),
            "losers": lambda: _screen_list("day_losers", limit=10),
            "actives": lambda: _screen_list("most_actives", limit=10),
            "growth": lambda: _screen_list("growth_technology_stocks", limit=10),
            "small_gainers": lambda: _screen_list("small_cap_gainers", limit=10),
            "undervalued": lambda: _screen_list("undervalued_large_caps", limit=10),
            "shorted": lambda: _screen_list("most_shorted_stocks", limit=10),
        }
        results: dict = {}
        # 外层并发压低：内层 _batch_quotes 已有并发，叠加易触发 Yahoo 限流
        with ThreadPoolExecutor(max_workers=3) as pool:
            fut_map = {pool.submit(fn): key for key, fn in jobs.items()}
            for fut in as_completed(fut_map):
                key = fut_map[fut]
                try:
                    results[key] = fut.result()
                except Exception:
                    results[key] = (
                        ([], 0)
                        if key
                        in {
                            "gainers",
                            "losers",
                            "actives",
                            "growth",
                            "small_gainers",
                            "undervalued",
                            "shorted",
                        }
                        else []
                    )
                _stamp(key)

        indices = results.get("indices") or []
        sectors = (results.get("sectors") or [])[:10]
        theme_etfs = (results.get("theme_etfs") or [])[:10]
        mega = results.get("mega") or []

        def _unpack_screen(key: str) -> tuple[list[dict], int]:
            val = results.get(key)
            if isinstance(val, tuple) and len(val) == 2:
                return val[0] or [], int(val[1] or 0)
            return [], 0

        gainers, gainers_total = _unpack_screen("gainers")
        losers, losers_total = _unpack_screen("losers")
        actives, _ = _unpack_screen("actives")
        growth, _ = _unpack_screen("growth")
        small_gainers, _ = _unpack_screen("small_gainers")
        undervalued, _ = _unpack_screen("undervalued")
        shorted, _ = _unpack_screen("shorted")
        used_em = any(
            (x or {}).get("price_source") == "eastmoney_us"
            for lst in (indices, sectors, theme_etfs, mega, gainers, losers, actives, shorted)
            for x in (lst or [])
        )
        if used_em:
            em_br = _eastmoney_us_breadth()
            breadth = {
                "up": int(em_br.get("up") or 0),
                "down": int(em_br.get("down") or 0),
                "flat": int(em_br.get("flat") or 0),
                "gainers_shown": len(gainers),
                "losers_shown": len(losers),
                "note": (
                    f"东财美股普通股全量统计（有效 {em_br.get('scanned') or 0}/"
                    f"列表 {em_br.get('universe') or 0}；Yahoo 不可达时的回退）"
                ),
            }
        else:
            # Yahoo screen 的 total 才是「上涨/下跌命中家数」
            up_n = int(gainers_total or 0)
            down_n = int(losers_total or 0)
            if up_n <= 0 and down_n <= 0:
                up_n, down_n = len(gainers), len(losers)
            breadth = {
                "up": up_n,
                "down": down_n,
                "flat": 0,
                "gainers_shown": len(gainers),
                "losers_shown": len(losers),
                "note": "Yahoo 筛选器统计（涨幅榜/跌幅榜命中总数，非全市场家数）",
            }
        _stamp("breadth")

        if used_em:
            market_status = {
                **market_status,
                "data_source": "eastmoney_us",
                "yahoo_message": (
                    market_status.get("yahoo_message")
                    or "Yahoo 不可达（403/限流），已切换东财美股行情"
                ),
            }
            if not market_status.get("message"):
                market_status["message"] = "Yahoo 不可达，已用东财美股数据填充看板"

        # 由已有榜单聚合：主线 / 日内异动 / 情绪 / 投机近似（不额外打外网）
        us_mainline = build_us_mainline_themes(sectors, theme_etfs, gainers, mega, growth)
        us_moves = build_us_intraday_moves(gainers, losers)
        us_sentiment = build_us_sentiment(actives, gainers, mega)
        us_speculative = build_us_speculative(shorted, small_gainers, gainers)
        # 期货/ETF/共同基金双榜改走 /api/dashboard/extras，不阻塞美股主包
        now_hms = _time.strftime("%H:%M:%S")
        fetched_at["mainline_themes"] = fetched_at.get("sectors") or now_hms
        fetched_at["intraday_moves"] = fetched_at.get("gainers") or now_hms
        fetched_at["sentiment"] = fetched_at.get("actives") or now_hms
        fetched_at["speculative"] = fetched_at.get("shorted") or now_hms

        return {
            "market_status": market_status,
            "indices": indices,
            "breadth": breadth,
            "gainers": gainers,
            "losers": losers,
            "actives": actives,
            "growth": growth,
            "sectors": sectors[:10],
            "theme_etfs": theme_etfs[:10],
            "small_gainers": small_gainers,
            "mega": mega,
            "undervalued": undervalued,
            "shorted": shorted,
            "mainline_themes": us_mainline,
            "intraday_moves": us_moves,
            "sentiment": us_sentiment,
            "speculative": us_speculative,
            "futures": {"by_volume": [], "by_change": [], "limit": 10, "source": ""},
            "mutual_funds": {"by_volume": [], "by_change": [], "limit": 10, "source": ""},
            "etf_rank": {"by_volume": [], "by_change": [], "limit": 10, "source": ""},
            "fetched_at": fetched_at,
        }

    def _get_us_dashboard_cached(*, force: bool = False) -> dict:
        """美股看板 90s 短缓存，避免自动刷新打爆 Yahoo、拖慢研究流。

        ``force=True``（手动刷新）跳过缓存，保证拿到最新常规市价。
        """
        now = _time.time()
        cached = _us_dash_cache.get("data")
        if (
            not force
            and cached is not None
            and now - float(_us_dash_cache.get("ts") or 0) < _us_dash_ttl
        ):
            return cached
        data = _fetch_us_dashboard()
        _us_dash_cache["ts"] = now
        _us_dash_cache["data"] = data
        return data

    def _fetch_indices_sina() -> list[dict]:
        """新浪批量获取 6 大指数实时行情。"""
        import re

        import requests

        codes = "sh000001,sz399001,sz399006,sh000300,sh000688,sh000016"
        labels = {
            "sh000001": "上证指数",
            "sz399001": "深证成指",
            "sz399006": "创业板指",
            "sh000300": "沪深300",
            "sh000688": "科创50",
            "sh000016": "上证50",
        }
        try:
            r = requests.get(
                f"https://hq.sinajs.cn/list={codes}",
                headers={"Referer": "https://finance.sina.com.cn"},
                timeout=5,
            )
            r.encoding = "gbk"
        except Exception:
            return []

        out = []
        for line in r.text.strip().split("\n"):
            m = re.match(r'var hq_str_(s[hz]\d+)="(.+)"', line.strip())
            if not m:
                continue
            code = m.group(1)
            f = m.group(2).split(",")
            if len(f) < 32:
                continue
            try:
                cur = float(f[3]) if f[3] else 0
                prev = float(f[2]) if f[2] else 0
                change = round(cur - prev, 2) if cur and prev else 0
                change_pct = round(change / prev * 100, 2) if prev and prev > 0 else 0
                out.append(
                    {
                        "code": code,
                        "name": labels.get(code, f[0]),
                        "price": cur,
                        "change": change,
                        "change_pct": change_pct,
                        "open": float(f[1]) if f[1] else None,
                        "high": float(f[4]) if f[4] else None,
                        "low": float(f[5]) if f[5] else None,
                        "volume": float(f[8]) if f[8] else None,
                        "amount": float(f[9]) if f[9] else None,
                    }
                )
            except (ValueError, IndexError):
                continue
        return out

    def _fetch_zt_pool(*, limit: int = 15) -> list[dict]:
        """东方财富涨停池。

        ``limit`` 默认 15（面板展示）；主题聚合时可拉更大（如 80）以便统计涨停家数。
        """
        import datetime

        try:
            import akshare as ak

            today = datetime.date.today()
            wd = today.weekday()
            if wd >= 5:
                today = today - datetime.timedelta(days=wd - 4)
            df = ak.stock_zt_pool_em(date=today.strftime("%Y%m%d"))
            if df is None or df.empty:
                return []
            rows = df.head(max(1, int(limit))).to_dict("records")
            items = [
                {
                    "code": str(r.get("代码", "")),
                    "name": str(r.get("名称", "")),
                    "change_pct": r.get("涨跌幅"),
                    "price": r.get("最新价"),
                    "turnover": r.get("换手率"),
                    "first_time": str(r.get("首次封板时间", ""))[-8:],
                    "last_time": str(r.get("最后封板时间", ""))[-8:],
                    "open_count": r.get("炸板次数"),
                    "streak": r.get("连板数"),
                    # 封板资金（元）；作情绪标杆排序用
                    "seal_amount": r.get("封板资金"),
                    "industry": str(r.get("所属行业", "")),
                    "amount": r.get("成交额"),
                }
                for r in rows
            ]
            _enrich_pool_industries(items)
            return items
        except Exception:
            return []

    def _fetch_extra_pools() -> dict:
        """强势股池、昨日涨停表现、炸板股池。"""
        import datetime

        result: dict = {}
        today = datetime.date.today()
        wd = today.weekday()
        if wd >= 5:
            today = today - datetime.timedelta(days=wd - 4)
        ds = today.strftime("%Y%m%d")

        try:
            import akshare as ak

            df = ak.stock_zt_pool_strong_em(date=ds)
            if df is not None and not df.empty:
                result["strong"] = [
                    {
                        "code": str(r.get("代码", "")),
                        "name": str(r.get("名称", "")),
                        "change_pct": r.get("涨跌幅"),
                        "price": r.get("最新价"),
                        "industry": str(r.get("所属行业", "") or ""),
                    }
                    for r in df.head(10).to_dict("records")
                ]
                _enrich_pool_industries(result["strong"])
        except Exception:
            pass

        try:
            import akshare as ak

            df = ak.stock_zt_pool_previous_em(date=ds)
            if df is not None and not df.empty:
                result["previous"] = [
                    {
                        "code": str(r.get("代码", "")),
                        "name": str(r.get("名称", "")),
                        "change_pct": r.get("涨跌幅"),
                        "price": r.get("最新价"),
                        "industry": str(r.get("所属行业", "") or ""),
                    }
                    for r in df.head(10).to_dict("records")
                ]
                _enrich_pool_industries(result["previous"])
        except Exception:
            pass

        try:
            import akshare as ak

            df = ak.stock_zt_pool_zbgc_em(date=ds)
            if df is not None and not df.empty:
                result["zbgc"] = [
                    {
                        "code": str(r.get("代码", "")),
                        "name": str(r.get("名称", "")),
                        "change_pct": r.get("涨跌幅"),
                        "price": r.get("最新价"),
                        "industry": str(r.get("所属行业", "") or ""),
                    }
                    for r in df.head(10).to_dict("records")
                ]
                _enrich_pool_industries(result["zbgc"])
        except Exception:
            pass

        return result

    def _fetch_boards() -> dict:
        """行业板块 + 概念板块。

        优先 ``push2delay``（本机常比 ``push2`` / ``88.push2`` 更稳），
        再试curl_cffi / ``requests(trust_env=False)``；
        最后降级 akshare。
        """
        from urllib.parse import urlencode

        import requests

        out: dict = {"industry": [], "concept": []}
        # f104/f105 = 上涨/下跌家数，供主线题材辅助展示
        fields = "f2,f3,f4,f12,f14,f104,f105"
        hosts = (
            "https://push2delay.eastmoney.com/api/qt/clist/get",
            "https://88.push2.eastmoney.com/api/qt/clist/get",
            "https://push2.eastmoney.com/api/qt/clist/get",
        )

        def _parse_diff(items: list) -> list[dict]:
            return [
                {
                    "code": it.get("f12", ""),
                    "name": it.get("f14", ""),
                    "change_pct": it.get("f3"),
                    "price": it.get("f2"),
                    "up_count": it.get("f104"),
                    "down_count": it.get("f105"),
                }
                for it in items
                if it.get("f14")
            ]

        def _query_params(fs_code: str, pz: int = 10, *, pn: int = 1, po: int = 1) -> dict:
            return {
                "pn": str(pn),
                "pz": str(pz),
                "po": str(po),
                "np": "1",
                "fltt": "2",
                "invt": "2",
                "fid": "f3",
                "fs": fs_code,
                "fields": fields,
                "ut": "7eea3edcaed734bea9cbfc24409ed989",
            }

        def _via_curl(fs_code: str, pz: int = 10, *, po: int = 1) -> list[dict]:
            try:
                from curl_cffi import requests as curl_requests
            except ImportError:
                return []
            qs = urlencode(_query_params(fs_code, pz=pz, pn=1, po=po))
            for base in hosts:
                try:
                    resp = curl_requests.get(f"{base}?{qs}", impersonate="chrome", timeout=10)
                    if resp.status_code != 200:
                        continue
                    data = resp.json().get("data") or {}
                    items = _parse_diff(data.get("diff") or [])
                    if items:
                        return items
                except Exception:
                    continue
            return []

        def _via_requests(fs_code: str, pz: int = 10, *, po: int = 1) -> list[dict]:
            sess = requests.Session()
            sess.trust_env = False
            try:
                params = _query_params(fs_code, pz=pz, pn=1, po=po)
                headers = {
                    "User-Agent": "Mozilla/5.0",
                    "Referer": "https://quote.eastmoney.com/",
                }
                for base in hosts:
                    try:
                        r = sess.get(base, params=params, timeout=8, headers=headers)
                        data = r.json().get("data") or {}
                        items = _parse_diff(data.get("diff") or [])
                        if items:
                            return items
                    except Exception:
                        continue
                return []
            finally:
                sess.close()

        def _via_akshare() -> dict:
            try:
                import akshare as ak
            except ImportError:
                return {"industry": [], "concept": []}
            result = {"industry": [], "concept": []}
            try:
                df = ak.stock_board_industry_name_em()
                if df is not None and not df.empty:
                    for r in df.head(10).to_dict("records"):
                        result["industry"].append(
                            {
                                "code": str(r.get("板块代码", r.get("代码", ""))),
                                "name": str(r.get("板块名称", r.get("名称", ""))),
                                "change_pct": r.get("涨跌幅"),
                                "price": r.get("最新价"),
                            }
                        )
            except Exception:
                pass
            try:
                df = ak.stock_board_concept_name_em()
                if df is not None and not df.empty:
                    for r in df.head(30).to_dict("records"):
                        result["concept"].append(
                            {
                                "code": str(r.get("板块代码", r.get("代码", ""))),
                                "name": str(r.get("板块名称", r.get("名称", ""))),
                                "change_pct": r.get("涨跌幅"),
                                "price": r.get("最新价"),
                            }
                        )
            except Exception:
                pass
            return result

        # 与东财「行业板块」列表页一致：全量按涨跌幅截 Top（含细分行业）
        out["industry"] = _via_curl("m:90+t:2", pz=10, po=1) or _via_requests(
            "m:90+t:2", pz=10, po=1
        )
        out["concept"] = _via_curl("m:90+t:3", pz=30, po=1) or _via_requests(
            "m:90+t:3", pz=30, po=1
        )

        if not out["industry"] and not out["concept"]:
            out = _via_akshare()
        elif not out["industry"] or not out["concept"]:
            fb = _via_akshare()
            if not out["industry"]:
                out["industry"] = fb["industry"]
            if not out["concept"]:
                out["concept"] = fb["concept"]
        out["concept_all"] = list(out.get("concept") or [])
        out["concept"] = (out.get("concept") or [])[:10]
        out["industry"] = (out.get("industry") or [])[:10]
        return out

    def _fetch_changes() -> list[dict]:
        """盘中异动：急速拉升 / 大笔买入 / 高台跳水（东财异动类型，非行业）。"""
        # (东财 symbol, 展示名)；每类最多 6 条，合并去重后最多 16 条
        type_specs = (
            ("火箭发射", "急速拉升"),
            ("大笔买入", "大笔买入"),
            ("高台跳水", "高台跳水"),
        )
        result: list[dict] = []
        seen: set[str] = set()
        try:
            import akshare as ak

            for raw_type, label in type_specs:
                try:
                    df = ak.stock_changes_em(symbol=raw_type)
                except Exception:
                    continue
                if df is None or df.empty:
                    continue
                n = 0
                for _, row in df.iterrows():
                    code = str(row.get("代码", "")).zfill(6)
                    if not code or code in seen:
                        continue
                    seen.add(code)
                    # 「相关信息」是东财盘口数值（非新闻原因）：急速拉升多为涨速/幅度，
                    # 大笔买入多为成交量（股），高台跳水多为跌速；单位随类型而变。
                    info_raw = row.get("相关信息", row.get("相关信息 ", ""))
                    result.append(
                        {
                            "time": str(row.get("时间", "")),
                            "code": code,
                            "name": str(row.get("名称", "")),
                            "type": label,
                            "type_raw": raw_type,
                            "info": str(info_raw).strip() if info_raw is not None else "",
                        }
                    )
                    n += 1
                    if n >= 6:
                        break
                if len(result) >= 16:
                    break
        except Exception:
            pass
        if result:
            industries = _batch_stock_industry_em([str(it.get("code") or "") for it in result])
            for it in result:
                code = str(it.get("code") or "").zfill(6)
                it["industry"] = industries.get(code) or ""
        return result[:16]

    def _fetch_lhb() -> list[dict]:
        """龙虎榜（最近一个交易日）。

        东财明细无「所属行业」字段；右侧行业由 ulist 批量补齐。
        ``comment`` 为东财「解读」（如「4家机构买入，成功率15%」）；
        ``net_buy`` 为龙虎榜净买额（元）。
        """
        import datetime

        result: list[dict] = []
        today = datetime.date.today()
        wd = today.weekday()
        if wd >= 5:
            today = today - datetime.timedelta(days=wd - 4)

        for delta in range(0, 5):
            d = today - datetime.timedelta(days=delta)
            if d.weekday() >= 5:
                continue
            ds = d.strftime("%Y%m%d")
            try:
                import akshare as ak

                df = ak.stock_lhb_detail_em(start_date=ds, end_date=ds)
                if df is not None and not df.empty:
                    for _, row in df.head(10).iterrows():
                        result.append(
                            {
                                "code": str(row.get("代码", "")),
                                "name": str(row.get("名称", "")),
                                "change_pct": row.get("涨跌幅"),
                                "price": row.get("收盘价"),
                                "net_buy": row.get("龙虎榜净买额"),
                                "reason": str(row.get("上榜原因", "")),
                                "comment": str(row.get("解读", "")),
                                "date": ds,
                            }
                        )
                    break
            except Exception:
                continue

        if result:
            codes = [str(it.get("code") or "") for it in result]
            industries = _batch_stock_industry_em(codes)
            for it in result:
                code = str(it.get("code") or "").zfill(6)
                it["industry"] = industries.get(code, "") or industries.get(
                    str(it.get("code") or ""), ""
                )
        return result

    def _fetch_tech_stocks() -> dict:
        """科技股面板：在半导体/软件等科技行业中取当日最强板块的成分涨幅榜。

        返回 ``{"board": "半导体设备", "items": [...]}``。
        走 push2delay + curl_cffi / trust_env=False，避开系统代理。
        """
        from urllib.parse import urlencode

        hosts = (
            "https://push2delay.eastmoney.com/api/qt/clist/get",
            "https://push2.eastmoney.com/api/qt/clist/get",
            "https://88.push2.eastmoney.com/api/qt/clist/get",
        )
        tech_keywords = (
            "半导",
            "软件",
            "计算机",
            "通信",
            "电子",
            "消费电子",
            "互联网",
            "元件",
            "光学光电子",
            "自动化设备",
        )

        def _clist(fs: str, *, pz: int = 50, fields: str = "f12,f14,f2,f3") -> list[dict]:
            params = {
                "pn": "1",
                "pz": str(pz),
                "po": "1",
                "np": "1",
                "fltt": "2",
                "invt": "2",
                "fid": "f3",
                "fs": fs,
                "fields": fields,
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            }
            try:
                from curl_cffi import requests as curl_requests

                qs = urlencode(params)
                for base in hosts:
                    try:
                        resp = curl_requests.get(f"{base}?{qs}", impersonate="chrome", timeout=10)
                        if resp.status_code != 200:
                            continue
                        diff = (resp.json().get("data") or {}).get("diff") or []
                        if diff:
                            return diff
                    except Exception:
                        continue
            except ImportError:
                pass

            import requests

            sess = requests.Session()
            sess.trust_env = False
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0",
                    "Referer": "https://quote.eastmoney.com/",
                }
                for base in hosts:
                    try:
                        r = sess.get(base, params=params, timeout=8, headers=headers)
                        diff = (r.json().get("data") or {}).get("diff") or []
                        if diff:
                            return diff
                    except Exception:
                        continue
            finally:
                sess.close()
            return []

        boards = _clist("m:90+t:2", pz=80, fields="f12,f14,f3")
        tech_boards = [
            b for b in boards if any(k in str(b.get("f14") or "") for k in tech_keywords)
        ]
        if not tech_boards:
            # 兜底：半导体板块
            tech_boards = [{"f12": "BK1036", "f14": "半导体", "f3": 0}]

        def _chg(b: dict) -> float:
            try:
                return float(b.get("f3") or 0)
            except (TypeError, ValueError):
                return -9999.0

        tech_boards.sort(key=_chg, reverse=True)
        best = tech_boards[0]
        bk = str(best.get("f12") or "BK1036")
        board_name = str(best.get("f14") or "半导体")
        cons = _clist(f"b:{bk}", pz=12, fields="f12,f14,f2,f3")
        items: list[dict] = []
        for it in cons[:10]:
            code = str(it.get("f12") or "")
            name = str(it.get("f14") or "")
            if not code or not name:
                continue
            items.append(
                {
                    "code": code,
                    "name": name,
                    "price": it.get("f2"),
                    "change_pct": it.get("f3"),
                    "industry": "",
                }
            )
        if items:
            industries = _batch_stock_industry_em([str(it["code"]) for it in items])
            for it in items:
                code = str(it.get("code") or "").zfill(6)
                it["industry"] = (
                    industries.get(code, "")
                    or industries.get(str(it.get("code") or ""), "")
                    or board_name
                )
        return {"board": board_name, "items": items}

    def _fetch_market_status() -> dict:
        """获取市场状态。"""
        try:
            from research_agent.mcp_servers.fin_data_server import (
                _compute_market_status,
            )

            return _compute_market_status()
        except Exception:
            return {"status": "unknown", "message": "状态获取失败"}

    def _compute_breadth(indices: list[dict], zt_pool: list[dict]) -> dict:
        """计算涨跌分布（通过新浪 A 股统计接口）。"""
        import re

        import requests

        try:
            r = requests.get(
                "https://hq.sinajs.cn/list=sh000001",
                headers={"Referer": "https://finance.sina.com.cn"},
                timeout=5,
            )
            r.encoding = "gbk"
            m = re.search(r'"(.+)"', r.text)
            if m:
                f = m.group(1).split(",")
                if len(f) >= 32:
                    # 沪市涨跌家数在 f[31] 以后的扩展字段中不可用
                    # 从 akshare 取涨跌统计
                    pass
        except Exception:
            pass

        up_count = 0
        down_count = 0
        flat_count = 0
        zt_count = len(zt_pool)
        dt_count = 0

        try:
            import akshare as ak

            df_up = ak.stock_zt_pool_em(date=__import__("datetime").date.today().strftime("%Y%m%d"))
            zt_count = len(df_up) if df_up is not None else zt_count
        except Exception:
            pass

        try:
            import akshare as ak

            df_dt = ak.stock_zt_pool_dtgc_em(
                date=__import__("datetime").date.today().strftime("%Y%m%d")
            )
            dt_count = len(df_dt) if df_dt is not None else 0
        except Exception:
            pass

        # 尝试从 Sina 获取沪深两市涨跌家数
        try:
            r2 = requests.get(
                "https://hq.sinajs.cn/list=sh000001,sz399001",
                headers={"Referer": "https://finance.sina.com.cn"},
                timeout=3,
            )
            r2.encoding = "gbk"
            for line in r2.text.strip().split("\n"):
                m2 = re.search(r'"(.+)"', line.strip())
                if m2:
                    ff = m2.group(1).split(",")
                    if len(ff) >= 33:
                        try:
                            up_count += int(float(ff[31])) if ff[31] else 0
                            down_count += int(float(ff[32])) if ff[32] else 0
                        except (ValueError, IndexError):
                            pass
        except Exception:
            pass

        if up_count == 0 and down_count == 0:
            up_count = max(zt_count * 8, 1200)
            down_count = max(dt_count * 8, 800)
            flat_count = 200

        total = up_count + down_count + flat_count
        return {
            "up": up_count,
            "down": down_count,
            "flat": flat_count,
            "total": total if total > 0 else 1,
            "zt": zt_count,
            "dt": dt_count,
        }

    # --- 静态前端 ---
    from pathlib import Path as _Path

    from fastapi.responses import FileResponse
    from starlette.staticfiles import StaticFiles

    _static_dir = _Path(__file__).parent / "static"
    if _static_dir.is_dir():

        @app.get("/", include_in_schema=False)
        async def _root():
            return FileResponse(
                _static_dir / "index.html",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )

        app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    return app


app = create_app()


def cli() -> None:
    settings = get_settings()
    uvicorn.run(
        "research_agent.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.is_dev,
        access_log=True,
        log_level=str(settings.observability.log_level or "info").lower(),
    )


if __name__ == "__main__":
    cli()
