"""Memory manager — unified interface for reading/writing long-term memories."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langgraph.store.base import BaseStore
from loguru import logger


class MemoryNamespace:
    """Predefined namespaces for organizing long-term memories."""

    USER_PREFERENCES = "user_preferences"
    RESEARCH_HISTORY = "research_history"
    DOMAIN_KNOWLEDGE = "domain_knowledge"


class MemoryManager:
    """High-level API for managing long-term agent memories.

    Wraps LangGraph's BaseStore with convenient namespace-based
    read/write operations.
    """

    def __init__(self, store: BaseStore) -> None:
        self._store = store

    async def save_memory(
        self,
        user_id: str,
        namespace: str,
        key: str,
        value: dict[str, Any],
    ) -> None:
        """Store a memory item under the given namespace."""
        ns = (user_id, namespace)
        value["_updated_at"] = datetime.now(timezone.utc).isoformat()
        await self._store.aput(ns, key, value)
        logger.debug("Memory saved: ns={}, key={}", ns, key)

    async def get_memory(
        self,
        user_id: str,
        namespace: str,
        key: str,
    ) -> dict[str, Any] | None:
        """Retrieve a specific memory item."""
        ns = (user_id, namespace)
        item = await self._store.aget(ns, key)
        if item is None:
            return None
        return item.value

    async def search_memories(
        self,
        user_id: str,
        namespace: str,
        query: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search memories in a namespace, optionally filtered by query."""
        ns = (user_id, namespace)
        items = await self._store.asearch(ns, query=query, limit=limit)
        return [item.value for item in items]

    async def save_research_result(
        self,
        user_id: str,
        query: str,
        summary: str,
        thread_id: str,
    ) -> None:
        """Save a completed research result to history for future reference."""
        await self.save_memory(
            user_id=user_id,
            namespace=MemoryNamespace.RESEARCH_HISTORY,
            key=thread_id,
            value={
                "query": query,
                "summary": summary[:500],
                "thread_id": thread_id,
            },
        )

    async def get_user_context(self, user_id: str) -> dict[str, Any]:
        """Retrieve all relevant user context for personalizing agent behavior."""
        preferences = await self.search_memories(
            user_id, MemoryNamespace.USER_PREFERENCES, limit=5,
        )
        history = await self.search_memories(
            user_id, MemoryNamespace.RESEARCH_HISTORY, limit=5,
        )
        return {
            "preferences": preferences,
            "recent_research": history,
        }
