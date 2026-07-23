"""单元测试、集成测试和端到端测试的共享 fixture。"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from research_agent.cache import reset_semantic_cache_for_tests, reset_tool_cache_for_tests
from research_agent.config import LLMConfig, Settings
from research_agent.llm.provider import ModelRouter
from research_agent.memory.manager import MemoryManager


@pytest.fixture(autouse=True)
def _disable_caches_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """工具 TTL 缓存与语义缓存在单元测试中默认关闭。

    否则同一进程内 mock/图测试会被缓存 HIT 污染。
    需要测缓存本身的用例在本地 fixture 里显式开启即可。
    """
    monkeypatch.setenv("TOOL_CACHE_ENABLED", "false")
    monkeypatch.setenv("TOOL_CACHE_BACKEND", "memory")
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "false")
    reset_tool_cache_for_tests()
    reset_semantic_cache_for_tests()
    yield
    reset_tool_cache_for_tests()
    reset_semantic_cache_for_tests()


@pytest.fixture
def settings() -> Settings:
    """使用安全默认值的测试配置。"""
    return Settings(app_env="testing")


@pytest.fixture
def llm_config() -> LLMConfig:
    return LLMConfig(
        openai_api_key="test-key",
        deepseek_api_key="test-key",
    )


@pytest.fixture
def model_router(llm_config: LLMConfig) -> ModelRouter:
    return ModelRouter(llm_config)


@pytest.fixture
def checkpointer() -> MemorySaver:
    return MemorySaver()


@pytest.fixture
def memory_store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def memory_manager(memory_store: InMemoryStore) -> MemoryManager:
    return MemoryManager(memory_store)
