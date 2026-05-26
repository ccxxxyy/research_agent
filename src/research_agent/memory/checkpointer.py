"""短期记忆 — 用于会话持久化的 LangGraph Checkpointer。

Checkpointer 在每次节点执行后对图状态进行快照，支持：
- 线程内多轮对话的连续性
- 从最后一个成功节点恢复故障
- 人机协作的暂停/恢复工作流

支持三种后端，按以下优先级选择：

1. PostgresSaver（生产环境）— 当提供 ``postgres_uri`` 时使用。
   持久化、支持并发，进程/主机重启后数据不丢失。
2. AsyncSqliteSaver（开发/演示）— 当提供 ``sqlite_path`` 时使用。
   基于文件，无需服务器，进程重启后数据不丢失。适合在无 docker 环境下复现"崩溃后恢复"场景。
3. MemorySaver（测试/冒烟）— 两者都未提供时的回退方案。
   快速、零配置，但数据随进程终止而丢失。
"""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from research_agent.memory._pg_reachability import is_postgres_reachable


async def init_checkpointer(
    postgres_uri: str | None = None,
    sqlite_path: str | Path | None = None,
) -> BaseCheckpointSaver:
    """为当前环境初始化最合适的 checkpointer。

    Args:
        postgres_uri: PostgreSQL 连接字符串。如果提供且可达，则优先使用。
            用于生产环境 / docker-compose 部署。
        sqlite_path: SQLite 数据库文件路径。如果提供且未使用 Postgres，则在该路径创建 :class:`~langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver`，父目录会自动创建。
            适用于 docker Postgres 未运行时（见 settings 中的 ``CHECKPOINT_SQLITE_PATH``）。

    Returns:
        适合当前环境的 :class:`BaseCheckpointSaver` 实例。调用方无需知道选择了哪个后端；抽象接口是统一的。

    Notes:
        - Postgres 初始化失败时会回退到 SQLite（如果已配置），然后回退到内存方案，并发出警告。这样可以避免在数据库容器关闭时破坏开发者的工作流。
        - SQLite 连接以 ``check_same_thread=False`` 打开，因为LangGraph 可能从工作线程访问 saver。
    """
    if postgres_uri:
        if not is_postgres_reachable(postgres_uri):
            # 完全跳过连接池。``ConnectionPool`` 是立即执行的，否则会启动后台重连循环，每隔几分钟就污染日志，
            # 而且在 Windows ProactorEventLoop 上，已观察到会饿死 FastAPI 请求处理程序（测试中 /health 调用挂起了 25 分钟以上）。见 _pg_reachability.py中的完整事件记录。
            logger.warning(
                "Postgres not reachable at startup; skipping PostgresSaver "
                "and falling back to sqlite/memory."
            )
        else:
            try:
                from psycopg.rows import dict_row
                from psycopg_pool import ConnectionPool
                from langgraph.checkpoint.postgres import PostgresSaver

                pool: ConnectionPool = ConnectionPool(
                    conninfo=postgres_uri,
                    kwargs={"row_factory": dict_row},
                )
                checkpointer = PostgresSaver(conn=pool)
                checkpointer.setup()
                logger.info("Checkpointer initialized: PostgresSaver")
                return checkpointer
            except Exception as e:
                logger.warning(
                    "PostgresSaver init failed ({}), trying sqlite/memory fallback",
                    e,
                )

    if sqlite_path is not None:
        try:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            path = Path(sqlite_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # 打开 aiosqlite 连接，使 AsyncSqliteSaver 能够与异步 agent 调用（``ainvoke`` / ``astream``）配合工作。
            # 同步的 SqliteSaver 变体在 asyncio 下会抛出 NotImplementedError。
            conn = await aiosqlite.connect(str(path))
            checkpointer = AsyncSqliteSaver(conn=conn)
            await checkpointer.setup()
            logger.info("Checkpointer initialized: AsyncSqliteSaver at {}", path)
            return checkpointer
        except Exception as e:
            logger.warning(
                "AsyncSqliteSaver init failed ({}), falling back to memory", e
            )

    logger.info("Checkpointer initialized: MemorySaver (in-memory, non-persistent)")
    return MemorySaver()
