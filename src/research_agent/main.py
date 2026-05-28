"""FastAPI Application 入口。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from research_agent.api.routes import a2a, health, knowledge, memory, sentiment, supervisor, usage
from research_agent.config import get_settings
from research_agent.observability.logging import setup_logging

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
        load_knowledge_tools_inproc,
        load_news_sentiment_server_tools,
        load_news_server_tools,
        load_pdf_report_server_tools,
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
        return_exceptions=True,
    )
    names = (
        "fin_data_server",
        "pdf_report_server",
        "code_server",
        "knowledge_tools_inproc",
        "news_server",
        "news_sentiment_server",
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
            report_tools=tools["pdf_report_server"] or None,
            coder_tools=tools["code_server"] or None,
            knowledge_tools=tools["knowledge_tools_inproc"] or None,
            news_tools=tools["news_server"] or None,
            sentiment_tools=tools["news_sentiment_server"] or None,
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

    METRICS.set_specialists(specialist_roster)

    app.state.supervisor_graph = supervisor_graph
    app.state.research_supervisor_graph = research_supervisor_graph
    app.state.available_specialists = specialist_roster
    app.state.model_router = model_router
    app.state.memory_store = memory_store
    app.state.checkpointer = checkpointer
    app.state.settings = settings

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
    app.include_router(a2a.router)

    # --- 静态前端 ---
    from pathlib import Path as _Path

    from fastapi.responses import FileResponse
    from starlette.staticfiles import StaticFiles

    _static_dir = _Path(__file__).parent / "static"
    if _static_dir.is_dir():

        @app.get("/", include_in_schema=False)
        async def _root():
            return FileResponse(_static_dir / "index.html")

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
