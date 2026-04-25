"""Long-term memory — cross-session persistent memory store.

MemoryStore enables agents to remember information across different
conversation threads. Typical use cases:
- User preferences and research interests
- Previously generated insights that can be reused
- Accumulated domain knowledge from past research sessions

Data is namespaced by (user_id, memory_type) for isolation.
"""

from __future__ import annotations

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from loguru import logger

from research_agent.memory._pg_reachability import is_postgres_reachable


async def init_memory_store(postgres_uri: str | None = None) -> BaseStore:
    """Initialize long-term memory store.

    - Development: InMemoryStore (resets on restart)
    - Production: PostgresStore (persistent across restarts)

    Note: The returned PostgresStore holds an open connection pool.
    Call ``store.conn.close()`` on shutdown to release resources.

    Reachability shortcut: see :mod:`research_agent.memory._pg_reachability`
    for why we TCP-probe before instantiating the eager
    ``ConnectionPool`` (TL;DR: a missing Postgres otherwise spawns a
    background reconnect storm that hangs HTTP handlers on Windows).
    """
    if postgres_uri:
        if not is_postgres_reachable(postgres_uri):
            logger.warning(
                "Postgres not reachable at startup; skipping PostgresStore "
                "and falling back to in-memory store."
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
                logger.warning("PostgresStore init failed ({}), falling back to memory", e)

    logger.info("MemoryStore initialized: InMemoryStore (non-persistent)")
    return InMemoryStore()
