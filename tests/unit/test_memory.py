"""长期记忆管理的测试。"""

import pytest

from research_agent.memory.manager import MemoryManager, MemoryNamespace


class TestMemoryManager:
    @pytest.mark.asyncio
    async def test_save_and_get_memory(self, memory_manager: MemoryManager):
        await memory_manager.save_memory(
            user_id="user1",
            namespace=MemoryNamespace.USER_PREFERENCES,
            key="lang",
            value={"language": "zh-CN"},
        )
        result = await memory_manager.get_memory(
            user_id="user1",
            namespace=MemoryNamespace.USER_PREFERENCES,
            key="lang",
        )
        assert result is not None
        assert result["language"] == "zh-CN"

    @pytest.mark.asyncio
    async def test_save_research_result(self, memory_manager: MemoryManager):
        await memory_manager.save_research_result(
            user_id="user1",
            query="AI Agent market analysis",
            summary="The AI Agent market is growing rapidly...",
            thread_id="thread-001",
        )
        context = await memory_manager.get_user_context("user1")
        assert len(context["recent_research"]) >= 1

    @pytest.mark.asyncio
    async def test_nonexistent_memory(self, memory_manager: MemoryManager):
        result = await memory_manager.get_memory("user1", "nonexistent", "key")
        assert result is None
