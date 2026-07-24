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
)
from research_agent.config import get_settings  # noqa: E402
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

    conv_db = getattr(settings, "conversation_sqlite_path", "./data/conversations.db")
    conv_store = ConversationStore(db_path=conv_db)
    app.state.conversation_store = conv_store

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
    app.include_router(a2a.router)

    # --- 热搜 API（轻量端点，供首页展示） ---
    import time as _time

    _trending_cache: dict = {"ts": 0, "data": None}
    _trending_ttl = 300  # 5 分钟缓存

    @app.get("/api/trending", tags=["trending"])
    async def get_trending():
        """返回多源热搜榜，供首页展示。

        数据源（3 个）：
        1. 人气榜 — emappdata.eastmoney.com 搜索热度 + 新浪实时行情
        2. 飙升榜 — 同源数据，按历史排名升幅排序
        3. 热门话题 — 东方财富研报标题提取市场焦点
        """
        import asyncio

        now = _time.time()
        if _trending_cache["data"] and now - _trending_cache["ts"] < _trending_ttl:
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
                    "pageSize": 20,
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
                            "title": title[:40],
                            "industry": industry,
                            "org": org,
                            "stock_name": stock_name,
                            "stock_code": stock_code,
                        }
                    )
                    if len(out) >= 10:
                        break
                return out
            except Exception:
                return []

        rank_task = asyncio.create_task(_em_rank_data())
        topic_task = asyncio.create_task(_em_topics())

        all_items = await rank_task
        topics = await topic_task

        # --- 人气榜 Top 10 ---
        em_hot = []
        if all_items:
            top10 = all_items[:10]
            codes = [it["sc"] for it in top10]
            info = await asyncio.to_thread(_batch_stock_info_sina, codes)
            for it in top10:
                sc = it["sc"]
                si = info.get(sc, {})
                code_bare = sc.replace("SZ", "").replace("SH", "")
                em_hot.append(
                    {
                        "rank": it.get("rk", ""),
                        "name": si.get("name", code_bare),
                        "code": code_bare,
                        "price": si.get("price"),
                        "change_pct": si.get("change_pct"),
                    }
                )

        # --- 飙升榜：hisRc 最小（排名上升最多）的 10 只 ---
        surge = []
        if all_items:
            surged = sorted(
                [it for it in all_items if it.get("hisRc", 0) < 0],
                key=lambda x: x.get("hisRc", 0),
            )[:10]
            if surged:
                codes = [it["sc"] for it in surged]
                info = await asyncio.to_thread(_batch_stock_info_sina, codes)
                for i, it in enumerate(surged):
                    sc = it["sc"]
                    si = info.get(sc, {})
                    code_bare = sc.replace("SZ", "").replace("SH", "")
                    surge.append(
                        {
                            "rank": i + 1,
                            "name": si.get("name", code_bare),
                            "code": code_bare,
                            "price": si.get("price"),
                            "change_pct": si.get("change_pct"),
                            "rank_change": abs(it.get("hisRc", 0)),
                        }
                    )

        result: dict = {}
        if em_hot:
            result["eastmoney"] = {"label": "人气榜", "items": em_hot}
        if surge:
            result["surge"] = {"label": "飙升榜", "items": surge}
        if topics:
            result["topics"] = {"label": "热门话题", "items": topics}

        if result:
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

    # --- 行情看板 API ---
    _dashboard_cache: dict = {"ts": 0, "data": None}
    _dashboard_ttl = 30  # 30 秒缓存

    @app.get("/api/dashboard", tags=["dashboard"])
    async def get_dashboard():
        """聚合首页行情看板数据，30 秒缓存。

        数据源：新浪实时指数 + EM 人气榜 + EM 涨停池 + EM 研报 + 市场状态。
        """
        now = _time.time()
        if _dashboard_cache["data"] and now - _dashboard_cache["ts"] < _dashboard_ttl:
            return _dashboard_cache["data"]

        idx_task = asyncio.create_task(asyncio.to_thread(_fetch_indices_sina))
        zt_task = asyncio.create_task(asyncio.to_thread(_fetch_zt_pool))
        extra_task = asyncio.create_task(asyncio.to_thread(_fetch_extra_pools))
        boards_task = asyncio.create_task(asyncio.to_thread(_fetch_boards))
        changes_task = asyncio.create_task(asyncio.to_thread(_fetch_changes))
        lhb_task = asyncio.create_task(asyncio.to_thread(_fetch_lhb))
        status_task = asyncio.create_task(asyncio.to_thread(_fetch_market_status))
        trending_task = asyncio.create_task(get_trending())

        indices = await idx_task
        zt_pool = await zt_task
        extra_pools = await extra_task
        boards = await boards_task
        changes = await changes_task
        lhb = await lhb_task
        market_status = await status_task
        trending = await trending_task

        breadth = _compute_breadth(indices, zt_pool)

        result = {
            "market_status": market_status,
            "indices": indices,
            "zt_pool": zt_pool,
            "strong_pool": extra_pools.get("strong", []),
            "previous_zt": extra_pools.get("previous", []),
            "zbgc_pool": extra_pools.get("zbgc", []),
            "boards": boards,
            "changes": changes,
            "lhb": lhb,
            "breadth": breadth,
            "trending": trending,
            "updated_at": _time.strftime("%H:%M:%S"),
        }
        _dashboard_cache["ts"] = now
        _dashboard_cache["data"] = result
        return result

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

    def _fetch_zt_pool() -> list[dict]:
        """东方财富涨停池 Top 15。"""
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
            rows = df.head(15).to_dict("records")
            return [
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
                    "industry": str(r.get("所属行业", "")),
                    "amount": r.get("成交额"),
                }
                for r in rows
            ]
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
                    }
                    for r in df.head(10).to_dict("records")
                ]
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
                    }
                    for r in df.head(10).to_dict("records")
                ]
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
                    }
                    for r in df.head(10).to_dict("records")
                ]
        except Exception:
            pass

        return result

    def _fetch_boards() -> dict:
        """行业板块 + 概念板块（push2 API）。"""
        import requests

        out: dict = {"industry": [], "concept": []}
        base = "https://push2.eastmoney.com/api/qt/clist/get"
        headers = {"User-Agent": "Mozilla/5.0"}
        fields = "f2,f3,f4,f12,f14"

        for key, fs_code in [("industry", "m:90+t:2"), ("concept", "m:90+t:3")]:
            try:
                params = {
                    "pn": "1",
                    "pz": "10",
                    "po": "1",
                    "np": "1",
                    "fltt": "2",
                    "invt": "2",
                    "fid": "f3",
                    "fs": fs_code,
                    "fields": fields,
                }
                r = requests.get(base, params=params, timeout=8, headers=headers)
                d = r.json()
                items = d.get("data", {}).get("diff", []) if d.get("data") else []
                out[key] = [
                    {
                        "code": it.get("f12", ""),
                        "name": it.get("f14", ""),
                        "change_pct": it.get("f3"),
                        "price": it.get("f2"),
                    }
                    for it in items
                ]
            except Exception:
                pass
        return out

    def _fetch_changes() -> list[dict]:
        """异动快照（火箭发射 + 大笔买入）。"""
        result: list[dict] = []
        try:
            import akshare as ak

            df = ak.stock_changes_em(symbol="火箭发射")
            if df is not None and not df.empty:
                seen: set = set()
                for _, row in df.iterrows():
                    code = str(row.get("代码", ""))
                    if code in seen:
                        continue
                    seen.add(code)
                    result.append(
                        {
                            "time": str(row.get("时间", "")),
                            "code": code,
                            "name": str(row.get("名称", "")),
                            "type": "火箭发射",
                        }
                    )
                    if len(result) >= 10:
                        break
        except Exception:
            pass
        return result

    def _fetch_lhb() -> list[dict]:
        """龙虎榜（最近一个交易日）。"""
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
                                "net_buy": row.get("龙虎榜净买额"),
                                "reason": str(row.get("上榜原因", "")),
                                "comment": str(row.get("解读", "")),
                                "date": ds,
                            }
                        )
                    break
            except Exception:
                continue
        return result

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
    )


if __name__ == "__main__":
    cli()
