"""Shared test fixtures for unit, integration, and e2e tests."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from research_agent.config import LLMConfig, Settings
from research_agent.llm.provider import ModelRouter
from research_agent.memory.manager import MemoryManager


@pytest.fixture
def settings() -> Settings:
    """Test settings with safe defaults."""
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
