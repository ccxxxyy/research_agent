"""记忆管理器 — 读写长期记忆的统一接口。

在 store 之上封装了更友好的 API。store 是底层的 key-value 存储，manager 加了命名空间概念。
    store.py 负责"怎么存"：选择用 Postgres 还是 SQLite 还是内存，初始化连接。关心的是基础设施。
    manager.py 负责"存什么"：定义命名空间（用户偏好、研究历史、领域知识），提供 save_memory/get_memory/save_research_result 等业务方法。关心的是业务逻辑。
核心方法：
    save_memory — 存一条记忆
    get_memory — 按 key 读一条
    search_memories — 搜索某个命名空间下的记忆
    save_research_result — 快捷方法，把研究结果存到历史
    get_user_context — 一次性拉出用户的偏好+最近研究，用来个性化 Agent 行为
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore


class MemoryNamespace:
    """用于组织长期记忆的预定义命名空间。分别是用户偏好 / 研究历史 / 领域知识。 """

    USER_PREFERENCES = "user_preferences"
    RESEARCH_HISTORY = "research_history"
    DOMAIN_KNOWLEDGE = "domain_knowledge"


class MemoryManager:
    """管理 agent 长期记忆的高层 API。

    封装 LangGraph 的 BaseStore，提供基于命名空间的便捷读写操作。
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
        """在指定命名空间下存储一条记忆项。"""
        ns = (user_id, namespace)
        value["_updated_at"] = datetime.now(UTC).isoformat()
        await self._store.aput(ns, key, value)
        logger.debug("Memory saved: ns={}, key={}", ns, key)

    async def get_memory(
        self,
        user_id: str,
        namespace: str,
        key: str,
    ) -> dict[str, Any] | None:
        """检索指定的记忆项。"""
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
        """在命名空间中搜索记忆，可选按查询条件过滤。"""
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
        """将已完成的研究结果保存到历史记录中，供未来参考。"""
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
        """检索所有相关的用户上下文，用于个性化 agent 行为。"""
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
