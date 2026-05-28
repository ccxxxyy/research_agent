"""有状态 Agent 功能的集成测试。

这些测试通过预设脚本的桩 LLM 来验证 checkpointer 层及其与``build_simple_agent`` 的交互，因此完全离线运行（无 API 调用、无需 API 密钥）且行为确定。

此处测试的三个关注点是无状态单 Agent 之上新增的内容：

1. ``init_checkpointer`` 工厂根据环境选择正确的后端（内存 / SQLite / Postgres 带回退）。
2. 带有 checkpointer 的 LangGraph Agent 在同一 ``thread_id`` 下 跨多次调用累积消息历史。
3. 两个不同的 ``thread_id`` 在同一进程和同一 checkpointer 实例中保持完全隔离。
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from research_agent.memory.checkpointer import init_checkpointer

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.callbacks import CallbackManagerForLLMRun

# ---------------------------------------------------------------------------
# 预设脚本的桩聊天模型
# ---------------------------------------------------------------------------

class _ScriptedChatModel(BaseChatModel):
    """确定性聊天模型，依次弹出预设的 ``AIMessage`` 回答。

    被测 Agent 对待此模型与真实模型完全相同，因此 LangGraph 的其余管道 — 消息累积、checkpoint、状态隔离 — 可以在不调用任何 API 的情况下进行验证。
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

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> _ScriptedChatModel:  # noqa: ARG002
        """create_react_agent 会探测此方法；直接忽略工具。"""
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted-stub"


def _build_agent_with(checkpointer, answers: list[str]):
    model = _ScriptedChatModel(answers=list(answers))
    # 零工具：桩模型从不发出 tool_calls，因此 ReAct 循环每轮在单次 LLM 步骤后即完成。
    return create_react_agent(model=model, tools=[], checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# init_checkpointer 测试
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
        assert db_path.exists(), "SQLite 文件应在磁盘上创建"
        assert db_path.parent.is_dir(), "父目录应被自动创建"

    @pytest.mark.asyncio
    async def test_unreachable_postgres_falls_back_to_sqlite(
        self, tmp_path: Path
    ) -> None:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        saver = await init_checkpointer(
            postgres_uri="postgresql://bad:bad@127.0.0.1:1/definitely_not_there",
            sqlite_path=tmp_path / "fallback.sqlite",
        )
        # Postgres 失败 → 尝试 SQLite 路径 → 返回 AsyncSqliteSaver。
        assert isinstance(saver, AsyncSqliteSaver)

    @pytest.mark.asyncio
    async def test_unreachable_postgres_no_sqlite_falls_back_to_memory(self) -> None:
        saver = await init_checkpointer(
            postgres_uri="postgresql://bad:bad@127.0.0.1:1/definitely_not_there",
        )
        assert isinstance(saver, MemorySaver)


# ---------------------------------------------------------------------------
# 使用 MemorySaver 的多轮记忆
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

        # 第 1 轮：Human + AI = 2。第 2 轮新增 Human + AI = 4。第 3 轮 → 6。
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
        # alice 第二轮返回的消息不应包含 bob 的状态。
        assert "b1" not in [m.content for m in r_alice2["messages"]]


# ---------------------------------------------------------------------------
# 使用 AsyncSqliteSaver 在同一文件上跨 saver 持久化
# ---------------------------------------------------------------------------

class TestSqlitePersistence:
    @pytest.mark.asyncio
    async def test_new_saver_reads_previous_savers_state(self, tmp_path: Path) -> None:
        db = tmp_path / "persist.sqlite"
        cfg = {"configurable": {"thread_id": "p1"}}

        # --- 第一个"进程"：写入状态然后关闭 ---
        saver_a = await init_checkpointer(sqlite_path=db)
        agent_a = _build_agent_with(saver_a, answers=["hello-from-A"])
        await agent_a.ainvoke({"messages": [HumanMessage(content="write-me")]}, config=cfg)
        # 模拟进程结束：释放资源
        await saver_a.conn.close()

        # --- 第二个"进程"：新 saver、相同文件、相同 thread id ---
        saver_b = await init_checkpointer(sqlite_path=db)
        agent_b = _build_agent_with(saver_b, answers=[])
        snapshot = await agent_b.aget_state(cfg)

        contents = [m.content for m in snapshot.values["messages"]]
        assert contents == ["write-me", "hello-from-A"]
        await saver_b.conn.close()


# ---------------------------------------------------------------------------
# build_simple_agent 集成外形
# ---------------------------------------------------------------------------

def test_build_simple_agent_exposes_checkpointer_param() -> None:
    """签名级别检查：Phase-2 参数表面保持稳定。"""
    import inspect

    from research_agent.agents.simple import build_simple_agent

    sig = inspect.signature(build_simple_agent)
    assert "checkpointer" in sig.parameters
    param = sig.parameters["checkpointer"]
    assert param.default is None, "checkpointer must be optional"


# ---------------------------------------------------------------------------
# pytest 异步支持的辅助函数
# ---------------------------------------------------------------------------

def _ensure_event_loop() -> asyncio.AbstractEventLoop:  # pragma: no cover
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
