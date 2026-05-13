"""Long-term memory — cross-session persistent memory store.

MemoryStore enables agents to remember information across different
conversation threads. Typical use cases:
- User preferences and research interests
- Previously generated insights that can be reused
- Accumulated domain knowledge from past research sessions

Data is namespaced by (user_id, memory_type) for isolation.

Three backends are supported, chosen in priority order:

1. **PostgresStore** (production) — when ``postgres_uri`` is supplied
   and reachable. Durable, concurrent, survives process restart.
2. **AsyncSqliteStore** (dev / demos) — when ``sqlite_path`` is
   supplied. File-based, survives process restart, no server needed.
3. **InMemoryStore** (tests / smoke) — fallback when neither is
   available. Fast but data dies with the Python process.
"""

from __future__ import annotations

from pathlib import Path

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from loguru import logger

from research_agent.memory._pg_reachability import is_postgres_reachable


async def init_memory_store(
    postgres_uri: str | None = None,
    sqlite_path: str | Path | None = None,
) -> BaseStore:
    """Initialize long-term memory store with three-level fallback.

    Args:
        postgres_uri: PostgreSQL connection string. If given and
            reachable, wins. Used in production / docker-compose.
        sqlite_path: Path to a SQLite database file. If given and
            Postgres is not used, an ``AsyncSqliteStore`` is created
            at that path (parent dirs auto-created). Typical on
            Windows / laptop dev when docker Postgres is down.

    Returns:
        An instance of :class:`BaseStore`. Caller does not need to
        know which backend was chosen.

    Notes:
        Any failure to initialize Postgres falls back to SQLite (if
        configured) then to in-memory, with a warning.
    """
    if postgres_uri:
        if not is_postgres_reachable(postgres_uri):
            logger.warning(
                "Postgres not reachable at startup; skipping PostgresStore "
                "and falling back to sqlite/memory store."
            )
        else:
            try:
                from psycopg.rows import dict_row
                from psycopg_pool import ConnectionPool
                from langgraph.store.postgres import PostgresStore

                pool: ConnectionPool = ConnectionPool(
                    conninfo=postgres_uri,
                    kwargs={"row_factory": dict_row},
                )
                store = PostgresStore(conn=pool)
                store.setup()
                logger.info("MemoryStore initialized: PostgresStore")
                return store
            except Exception as e:
                logger.warning(
                    "PostgresStore init failed ({}), trying sqlite/memory fallback", e
                )

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
            logger.warning(
                "AsyncSqliteStore init failed ({}), falling back to memory", e
            )

    logger.info("MemoryStore initialized: InMemoryStore (non-persistent)")
    return InMemoryStore()
