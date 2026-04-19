"""Short-term memory — LangGraph Checkpointer for conversation persistence.

Checkpointers snapshot graph state after every node execution, enabling:
- Multi-turn conversation continuity within a thread
- Fault recovery from the last successful node (not from scratch)
- Human-in-the-loop pause/resume workflows
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger


async def init_checkpointer(postgres_uri: str | None = None) -> BaseCheckpointSaver:
    """Initialize the appropriate checkpointer based on environment.

    - Development: InMemorySaver (fast, no persistence across restarts)
    - Production: PostgresSaver (durable, survives restarts)

    Note: The returned PostgresSaver holds an open connection pool.
    Call ``checkpointer.conn.close()`` on shutdown to release resources.
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
            logger.warning("PostgresSaver init failed ({}), falling back to memory", e)

    logger.info("Checkpointer initialized: MemorySaver (in-memory, non-persistent)")
    return MemorySaver()
