"""Short-term memory — LangGraph Checkpointer for conversation persistence.

Checkpointers snapshot graph state after every node execution, enabling:
- Multi-turn conversation continuity within a thread
- Fault recovery from the last successful node (not from scratch)
- Human-in-the-loop pause/resume workflows

Three backends are supported, chosen in this priority order:

1. **PostgresSaver** (production)  — when ``postgres_uri`` is supplied.
   Durable, concurrent, survives process/host restart.
2. **SqliteSaver**  (dev / demos)  — when ``sqlite_path`` is supplied.
   File-based, no server needed, survives process restart. Ideal for
   reproducing "resume after crash" stories without docker.
3. **MemorySaver**  (tests / smoke) — fallback when neither is supplied.
   Fast, zero setup, but data dies with the Python process.
"""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger


async def init_checkpointer(
    postgres_uri: str | None = None,
    sqlite_path: str | Path | None = None,
) -> BaseCheckpointSaver:
    """Initialize the best-fit checkpointer for the current environment.

    Args:
        postgres_uri: PostgreSQL connection string. If given and reachable,
            wins. Used in production / docker-compose setups.
        sqlite_path: Path to a SQLite database file. If given and Postgres
            is not, a :class:`SqliteSaver` is created at that path, with
            parent directories auto-created. Used in dev / demos for
            cross-process persistence without a DB server.

    Returns:
        An instance of :class:`BaseCheckpointSaver` appropriate for the
        environment. Caller does not need to know which backend was chosen;
        the abstract interface is uniform.

    Notes:
        - Any failure to initialize Postgres falls back to SQLite (if
          configured) then to in-memory, with a warning. This avoids
          breaking developer workflow when the DB container is down.
        - The SQLite connection is opened with ``check_same_thread=False``
          because LangGraph may access the saver from worker threads.
    """
    if postgres_uri:
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
                "PostgresSaver init failed ({}), trying sqlite/memory fallback", e
            )

    if sqlite_path is not None:
        try:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            path = Path(sqlite_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Open an aiosqlite connection so AsyncSqliteSaver can work with
            # async agent invocations (``ainvoke`` / ``astream``). The sync
            # SqliteSaver variant would raise NotImplementedError under asyncio.
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
