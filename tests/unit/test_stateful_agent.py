"""Integration tests for Phase-2 stateful agent functionality.

These tests exercise the checkpointer layer and its interaction with
``build_simple_agent`` using a **scripted stub LLM**, so they run fully
offline (no API calls, no API keys required) and deterministically.

The three concerns tested here are exactly the three that Phase 2 adds
on top of Phase 1's stateless single agent:

1. The ``init_checkpointer`` factory picks the right backend given the
   environment (memory / sqlite / postgres-with-fallback).
2. A LangGraph agent with a checkpointer accumulates message history
   under a ``thread_id`` across multiple invocations.
3. Two different ``thread_id``s remain fully isolated inside the same
   process and the same checkpointer instance.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from research_agent.memory.checkpointer import init_checkpointer


# ---------------------------------------------------------------------------
# Scripted stub chat model
# ---------------------------------------------------------------------------

class _ScriptedChatModel(BaseChatModel):
    """Deterministic chat model that pops pre-scripted ``AIMessage`` answers.

    The agent under test treats this model identically to a real one,
    so the rest of the LangGraph plumbing — message accumulation,
    checkpointing, state isolation — can be exercised without any API.
    """

    answers: list[str]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.answers:
            raise RuntimeError("Scripted model exhausted: no more answers.")
        reply = self.answers.pop(0)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "_ScriptedChatModel":  # noqa: ARG002
        """create_react_agent probes for this method; we just ignore tools."""
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted-stub"


def _build_agent_with(checkpointer, answers: list[str]):
    model = _ScriptedChatModel(answers=list(answers))
    # Zero tools: the stub never emits tool_calls, so the ReAct loop
    # completes after a single LLM step per turn.
    return create_react_agent(model=model, tools=[], checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# init_checkpointer
# ---------------------------------------------------------------------------

class TestInitCheckpointer:
    @pytest.mark.asyncio
    async def test_no_args_returns_memory_saver(self) -> None:
        saver = await init_checkpointer()
        assert isinstance(saver, MemorySaver)

    @pytest.mark.asyncio
    async def test_sqlite_path_creates_file_and_returns_async_saver(
        self, tmp_path: Path
    ) -> None:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        db_path = tmp_path / "nested" / "dir" / "cp.sqlite"
        saver = await init_checkpointer(sqlite_path=db_path)

        assert isinstance(saver, AsyncSqliteSaver)
        assert db_path.exists(), "sqlite file should be created on disk"
        assert db_path.parent.is_dir(), "parent dirs should be auto-created"

    @pytest.mark.asyncio
    async def test_unreachable_postgres_falls_back_to_sqlite(
        self, tmp_path: Path
    ) -> None:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        saver = await init_checkpointer(
            postgres_uri="postgresql://bad:bad@127.0.0.1:1/definitely_not_there",
            sqlite_path=tmp_path / "fallback.sqlite",
        )
        # Postgres fails → sqlite path is tried → AsyncSqliteSaver returned.
        assert isinstance(saver, AsyncSqliteSaver)

    @pytest.mark.asyncio
    async def test_unreachable_postgres_no_sqlite_falls_back_to_memory(self) -> None:
        saver = await init_checkpointer(
            postgres_uri="postgresql://bad:bad@127.0.0.1:1/definitely_not_there",
        )
        assert isinstance(saver, MemorySaver)


# ---------------------------------------------------------------------------
# Multi-turn memory with MemorySaver
# ---------------------------------------------------------------------------

class TestMemorySaverMultiTurn:
    @pytest.mark.asyncio
    async def test_state_accumulates_across_turns_same_thread(self) -> None:
        saver = MemorySaver()
        agent = _build_agent_with(saver, answers=["reply-1", "reply-2", "reply-3"])
        cfg = {"configurable": {"thread_id": "t1"}}

        r1 = await agent.ainvoke({"messages": [HumanMessage(content="q1")]}, config=cfg)
        r2 = await agent.ainvoke({"messages": [HumanMessage(content="q2")]}, config=cfg)
        r3 = await agent.ainvoke({"messages": [HumanMessage(content="q3")]}, config=cfg)

        # Turn 1: Human + AI = 2. Turn 2 adds Human + AI = 4. Turn 3 → 6.
        assert len(r1["messages"]) == 2
        assert len(r2["messages"]) == 4
        assert len(r3["messages"]) == 6

        contents = [m.content for m in r3["messages"]]
        assert contents == ["q1", "reply-1", "q2", "reply-2", "q3", "reply-3"]

    @pytest.mark.asyncio
    async def test_thread_isolation(self) -> None:
        saver = MemorySaver()
        agent = _build_agent_with(saver, answers=["alice-1", "bob-1", "alice-2"])

        cfg_a = {"configurable": {"thread_id": "alice"}}
        cfg_b = {"configurable": {"thread_id": "bob"}}

        await agent.ainvoke({"messages": [HumanMessage(content="a1")]}, config=cfg_a)
        await agent.ainvoke({"messages": [HumanMessage(content="b1")]}, config=cfg_b)
        r_alice2 = await agent.ainvoke(
            {"messages": [HumanMessage(content="a2")]}, config=cfg_a
        )

        alice_snapshot = await agent.aget_state(cfg_a)
        bob_snapshot = await agent.aget_state(cfg_b)

        alice_contents = [m.content for m in alice_snapshot.values["messages"]]
        bob_contents = [m.content for m in bob_snapshot.values["messages"]]

        assert alice_contents == ["a1", "alice-1", "a2", "alice-2"]
        assert bob_contents == ["b1", "bob-1"]
        # The returned messages from alice's second turn must not contain bob's state.
        assert "b1" not in [m.content for m in r_alice2["messages"]]


# ---------------------------------------------------------------------------
# Persistence with AsyncSqliteSaver across savers on the same file
# ---------------------------------------------------------------------------

class TestSqlitePersistence:
    @pytest.mark.asyncio
    async def test_new_saver_reads_previous_savers_state(self, tmp_path: Path) -> None:
        db = tmp_path / "persist.sqlite"
        cfg = {"configurable": {"thread_id": "p1"}}

        # --- first "process": write state then close ---
        saver_a = await init_checkpointer(sqlite_path=db)
        agent_a = _build_agent_with(saver_a, answers=["hello-from-A"])
        await agent_a.ainvoke({"messages": [HumanMessage(content="write-me")]}, config=cfg)
        # simulate process end: release resources
        await saver_a.conn.close()

        # --- second "process": fresh saver, same file, same thread id ---
        saver_b = await init_checkpointer(sqlite_path=db)
        agent_b = _build_agent_with(saver_b, answers=[])
        snapshot = await agent_b.aget_state(cfg)

        contents = [m.content for m in snapshot.values["messages"]]
        assert contents == ["write-me", "hello-from-A"]
        await saver_b.conn.close()


# ---------------------------------------------------------------------------
# build_simple_agent integration shape
# ---------------------------------------------------------------------------

def test_build_simple_agent_exposes_checkpointer_param() -> None:
    """Signature-level check: Phase-2 parameter surface is stable."""
    import inspect

    from research_agent.agents.simple import build_simple_agent

    sig = inspect.signature(build_simple_agent)
    assert "checkpointer" in sig.parameters
    param = sig.parameters["checkpointer"]
    assert param.default is None, "checkpointer must be optional"


# ---------------------------------------------------------------------------
# Helpers for pytest's async support
# ---------------------------------------------------------------------------

def _ensure_event_loop() -> asyncio.AbstractEventLoop:  # pragma: no cover
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
