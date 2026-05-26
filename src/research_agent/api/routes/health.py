"""健康检查端点。"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request

from research_agent import __version__
from research_agent.api.schemas import HealthResponse
from research_agent.config import get_settings

router = APIRouter(tags=["health"])

_KNOWLEDGE_DB_DIR = Path("data/knowledge_db")


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """探测真实的服务依赖并如实报告状态。

    通过模块级 ``get_settings()``（``@lru_cache`` 工厂）而非 ``request.app.state`` 读取配置，使得在 ASGI 生命周期运行之前
    即可调用该探针 —— 例如集成测试中的 ``httpx.ASGITransport``，或 Kubernetes 启动探针在生命周期 ``on_startup`` 钩子完成初始化，长时运行资源之前触发的场景。
    """
    services: dict[str, str] = {}

    # Postgres —— 复用轻量 TCP 探测
    from research_agent.memory._pg_reachability import is_postgres_reachable

    settings = get_settings()
    pg_uri = settings.database.postgres_sync_uri
    services["postgres"] = "ok" if is_postgres_reachable(pg_uri) else "unreachable"

    # Redis —— 异步 ping（对 asyncio 事件循环无阻塞）。
    try:
        from redis.asyncio import Redis

        async with Redis.from_url(
            settings.database.redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        ) as client:
            await client.ping()
        services["redis"] = "ok"
    except Exception:
        services["redis"] = "unreachable"

    # 知识库（FAISS 目录是否存在）
    services["knowledge_db"] = "ok" if _KNOWLEDGE_DB_DIR.is_dir() else "not_initialized"

    # 研究主管图可用性。``app.state`` 可能尚不存在（生命周期未运行）——``getattr`` 保证安全访问。
    graph = getattr(request.app.state, "research_supervisor_graph", None)
    services["research_supervisor"] = "ok" if graph is not None else "unavailable"

    # 检查点后端 —— 报告当前激活的层级，以便了解短期记忆是否已静默降级为内存模式。
    checkpointer = getattr(request.app.state, "checkpointer", None)
    if checkpointer is not None:
        backend = type(checkpointer).__name__
        services["checkpointer"] = f"ok ({backend})"
    else:
        services["checkpointer"] = "unavailable"

    # 记忆存储后端 —— 与检查点后端同理。
    memory_store = getattr(request.app.state, "memory_store", None)
    if memory_store is not None:
        backend = type(memory_store).__name__
        services["memory_store"] = f"ok ({backend})"
    else:
        services["memory_store"] = "unavailable"

    # 汇总存活性。Postgres/Redis 是一级生产依赖，其余为建议性依赖。
    # 只要数据面正常即报告 ``ok``。
    # 可选服务缺失会降级响应，但不会将绿色仪表盘翻转为红色。
    critical = ("postgres", "redis")
    overall = (
        "ok" if all(services.get(k) == "ok" for k in critical) else "degraded"
    )

    return HealthResponse(
        status=overall,
        version=__version__,
        services=services,
    )
