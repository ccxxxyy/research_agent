"""长期记忆 — 跨会话持久化记忆存储。

MemoryStore 使 agent 能够在不同对话线程之间记住信息。典型用例：
- 用户偏好和研究兴趣
- 之前生成的可复用洞察
- 从过往研究会话中积累的领域知识

数据按 (user_id, memory_type) 命名空间隔离。
    用户 A 的偏好：("userA", "user_preferences") → {"language": "zh", ...}
    用户 A 的研究历史：("userA", "research_history") → {"query": "宁德时代业绩", ...}
    用户 B 的数据和 A 完全隔离

支持三种后端，按以下优先级选择：

1. PostgresStore（生产环境）— 当提供 ``postgres_uri`` 且可达时使用。
   持久化、支持并发，进程重启后数据不丢失。
2. AsyncSqliteStore（开发/演示）— 当提供 ``sqlite_path`` 时使用。
   基于文件，进程重启后数据不丢失，无需服务器。
3. InMemoryStore（测试/冒烟）— 两者都不可用时的回退方案。
   快速但数据随 Python 进程终止而丢失。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langgraph.store.memory import InMemoryStore
from loguru import logger

from research_agent.memory._pg_reachability import is_postgres_reachable

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore


async def init_memory_store(
    postgres_uri: str | None = None,
    sqlite_path: str | Path | None = None,
) -> BaseStore:
    """初始化长期记忆存储，支持三级回退。

    Args:
        postgres_uri: PostgreSQL 连接字符串。如果提供且可达，则优先使用。
            用于生产环境 / docker-compose 部署。
        sqlite_path: SQLite 数据库文件路径。如果提供且未使用 Postgres，则在该路径创建 ``AsyncSqliteStore``（父目录会自动创建）。
            适用于 docker Postgres 未运行时。

    Returns:
        一个 :class:`BaseStore` 实例。调用方无需知道选择了哪个后端。

    Notes:
        Postgres 初始化失败时会回退到 SQLite（如果已配置），然后回退到 内存方案，并发出警告。
    """
    if postgres_uri:
        if not is_postgres_reachable(postgres_uri):
            logger.warning(
                "Postgres not reachable at startup; skipping PostgresStore "
                "and falling back to sqlite/memory store."
            )
        else:
            try:
                from langgraph.store.postgres import PostgresStore
                from psycopg.rows import dict_row
                from psycopg_pool import ConnectionPool

                pool: ConnectionPool = ConnectionPool(
                    conninfo=postgres_uri,
                    kwargs={"row_factory": dict_row},
                )
                store = PostgresStore(conn=pool)
                store.setup()
                logger.info("MemoryStore initialized: PostgresStore")
                return store
            except Exception as e:
                logger.warning("PostgresStore init failed ({}), trying sqlite/memory fallback", e)

    if sqlite_path is not None:
        try:
            import aiosqlite
            from langgraph.store.sqlite import AsyncSqliteStore

            path = Path(sqlite_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            conn = await aiosqlite.connect(str(path))
            store = AsyncSqliteStore(conn=conn)
            await store.setup()
            logger.info("MemoryStore initialized: AsyncSqliteStore at {}", path)
            return store
        except Exception as e:
            logger.warning("AsyncSqliteStore init failed ({}), falling back to memory", e)

    logger.info("MemoryStore initialized: InMemoryStore (non-persistent)")
    return InMemoryStore()
