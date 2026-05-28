"""人机协作 (HITL) 审批/恢复流程的单元测试。

测试覆盖：
  * 当图被中断时，SSE 流发出 ``review_requested`` 事件。
  * ``POST /api/supervisor/research/{thread_id}/approve`` 恢复图执行。
  * ``POST /api/supervisor/research/{thread_id}/resume`` 携带修订反馈。
  * 线程未暂停时返回 409 响应。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage

from research_agent.api.dependencies import (
    get_memory_manager,
    get_research_supervisor_graph,
    get_supervisor_graph,
)
from research_agent.api.routes.supervisor import router as supervisor_router

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from research_agent.memory.manager import MemoryManager

# ---------------------------------------------------------------------------
# 用于 HITL 测试的模拟图
# ---------------------------------------------------------------------------


class _FakeState:
    """LangGraph StateSnapshot 的最小替身。

    真实的 ``StateSnapshot.values`` 始终是 dict（不会为 ``None``）。 不存在的线程通过 ``aget_state`` 返回 ``None`` 来表示；
    已完成的线程``next`` 为空但 ``values`` 有值；被中断的线程 ``next`` 非空。
    """

    def __init__(
        self,
        *,
        is_interrupted: bool = False,
        is_empty: bool = False,
    ) -> None:
        self.next = ("human_review",) if is_interrupted else ()
        self.tasks = ()
        self.values = {} if is_empty else {"messages": [AIMessage(content="draft")]}


class _HITLStreamGraph:
    """模拟一个流式输出更新后被中断的图。

    ``astream`` 产出预设的消息；``aget_state`` 报告图处于中断状态，使 SSE 层发出 ``review_requested``。
    """

    def __init__(self, scripted_messages: list[Any]) -> None:
        self._scripted = scripted_messages
        self._interrupted = True

    async def astream(
        self,
        inputs: dict,
        config: dict | None = None,
        stream_mode: str = "updates",
        **kwargs: object,
    ) -> AsyncIterator[dict | tuple]:
        subgraphs = kwargs.get("subgraphs", False)
        for msg in self._scripted:
            node = getattr(msg, "name", None) or "supervisor"
            chunk = {node: {"messages": [msg]}}
            yield ((), chunk) if subgraphs else chunk

    async def aget_state(self, config: dict) -> _FakeState:
        return _FakeState(is_interrupted=self._interrupted)


class _HITLResumeGraph(_HITLStreamGraph):
    """扩展 _HITLStreamGraph，添加 ``ainvoke`` 用于恢复测试。"""

    def __init__(
        self,
        scripted_messages: list[Any],
        resume_reply: str = "approved final answer",
    ) -> None:
        super().__init__(scripted_messages)
        self._resume_reply = resume_reply

    async def ainvoke(self, inputs: Any, config: dict | None = None) -> dict:
        self._interrupted = False
        return {
            "messages": [
                HumanMessage(content="original query"),
                AIMessage(content=self._resume_reply, name="supervisor"),
            ]
        }


class _CompletedGraph:
    """未被中断的图 — 用于测试 409 路径。"""

    async def aget_state(self, config: dict) -> _FakeState:
        return _FakeState(is_interrupted=False)

    async def ainvoke(self, inputs: Any, config: dict | None = None) -> dict:
        return {"messages": [AIMessage(content="done", name="supervisor")]}

    async def astream(
        self,
        inputs: dict,
        config: dict | None = None,
        stream_mode: str = "updates",
        **kwargs: object,
    ) -> AsyncIterator[dict | tuple]:
        subgraphs = kwargs.get("subgraphs", False)
        chunk = {"supervisor": {"messages": [AIMessage(content="done", name="supervisor")]}}
        yield ((), chunk) if subgraphs else chunk


# ---------------------------------------------------------------------------
# 测试应用构建器
# ---------------------------------------------------------------------------


def _build_hitl_app(graph: Any, *, memory: MemoryManager | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(supervisor_router)
    app.dependency_overrides[get_research_supervisor_graph] = lambda: graph
    app.dependency_overrides[get_supervisor_graph] = lambda: graph
    if memory is not None:
        app.dependency_overrides[get_memory_manager] = lambda: memory
    return app


def _parse_sse(body: bytes) -> list[dict]:
    events: list[dict] = []
    text = body.decode("utf-8")
    for frame in text.split("\n\n"):
        frame = frame.strip()
        if not frame or not frame.startswith("data:"):
            continue
        payload = frame[len("data:") :].strip()
        events.append(json.loads(payload))
    return events


# ---------------------------------------------------------------------------
# SSE review_requested 事件发送
# ---------------------------------------------------------------------------


class TestSSEReviewRequested:
    @pytest.mark.asyncio
    async def test_stream_emits_review_requested_when_interrupted(self) -> None:
        """当图被中断时，SSE 流应发出包含草稿内容的``review_requested`` 事件。"""
        graph = _HITLStreamGraph(
            [
                AIMessage(
                    content="",
                    name="supervisor",
                    tool_calls=[{"name": "transfer_to_data_expert", "args": {}, "id": "x"}],
                ),
                AIMessage(content="specialist data", name="data_expert"),
                AIMessage(content="supervisor draft synthesis", name="supervisor"),
            ]
        )
        app = _build_hitl_app(graph)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/supervisor/research/stream",
                json={"query": "test hitl"},
            )

        assert r.status_code == 200
        events = _parse_sse(r.content)
        phases = [e["phase"] for e in events]

        assert "review_requested" in phases
        review_evt = next(e for e in events if e["phase"] == "review_requested")
        assert review_evt["node"] == "human_review"
        assert "supervisor draft synthesis" in review_evt["content"]
        assert review_evt["metadata"]["action_required"] == "approve_or_revise"
        assert phases[-1] == "done"

    @pytest.mark.asyncio
    async def test_stream_no_review_requested_when_not_interrupted(self) -> None:
        """正常完成的图不应发出 review_requested 事件。"""
        graph = _CompletedGraph()
        app = _build_hitl_app(graph)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/supervisor/research/stream",
                json={"query": "test no hitl"},
            )

        events = _parse_sse(r.content)
        phases = [e["phase"] for e in events]
        assert "review_requested" not in phases


# ---------------------------------------------------------------------------
# POST /approve 审批接口
# ---------------------------------------------------------------------------


class TestApproveRoute:
    @pytest.mark.asyncio
    async def test_approve_resumes_interrupted_graph(self) -> None:
        graph = _HITLResumeGraph(
            scripted_messages=[AIMessage(content="draft", name="supervisor")],
            resume_reply="final after approval",
        )
        app = _build_hitl_app(graph)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/supervisor/research/test-thread-1/approve",
                json={"feedback": ""},
            )

        assert r.status_code == 200
        body = r.json()
        assert body["reply"] == "final after approval"
        assert body["thread_id"] == "test-thread-1"

    @pytest.mark.asyncio
    async def test_approve_with_feedback(self) -> None:
        graph = _HITLResumeGraph(
            scripted_messages=[],
            resume_reply="revised answer",
        )
        app = _build_hitl_app(graph)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/supervisor/research/test-thread-2/approve",
                json={"feedback": "Add more citations"},
            )

        assert r.status_code == 200
        assert r.json()["reply"] == "revised answer"

    @pytest.mark.asyncio
    async def test_approve_409_when_not_interrupted(self) -> None:
        graph = _CompletedGraph()
        app = _build_hitl_app(graph)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/supervisor/research/thread-done/approve",
                json={"feedback": ""},
            )

        assert r.status_code == 409
        assert "not paused" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# POST /resume 恢复接口
# ---------------------------------------------------------------------------


class TestResumeRoute:
    @pytest.mark.asyncio
    async def test_resume_with_feedback(self) -> None:
        graph = _HITLResumeGraph(
            scripted_messages=[],
            resume_reply="revised with feedback",
        )
        app = _build_hitl_app(graph)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/supervisor/research/thread-r1/resume",
                json={"feedback": "Focus on ESG metrics"},
            )

        assert r.status_code == 200
        body = r.json()
        assert body["reply"] == "revised with feedback"
        assert body["thread_id"] == "thread-r1"

    @pytest.mark.asyncio
    async def test_resume_without_feedback_acts_as_approve(self) -> None:
        graph = _HITLResumeGraph(
            scripted_messages=[],
            resume_reply="approved via resume",
        )
        app = _build_hitl_app(graph)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/supervisor/research/thread-r2/resume",
                json={"feedback": ""},
            )

        assert r.status_code == 200
        assert r.json()["reply"] == "approved via resume"

    @pytest.mark.asyncio
    async def test_resume_409_when_not_interrupted(self) -> None:
        graph = _CompletedGraph()
        app = _build_hitl_app(graph)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/supervisor/research/thread-done/resume",
                json={"feedback": "nope"},
            )

        assert r.status_code == 409


# ---------------------------------------------------------------------------
# 线程状态错误分类 (P1-C)
# ---------------------------------------------------------------------------


class _AbsentThreadGraph:
    """``aget_state`` 返回 ``None`` — 线程不存在。"""

    async def aget_state(self, config: dict) -> None:
        return None

    async def ainvoke(self, inputs: Any, config: dict | None = None) -> dict:
        return {"messages": []}


class _EmptyStateGraph:
    """``aget_state`` 返回一个 ``next`` 为空且 values 也为空的状态。

    LangGraph 的 PostgresStore / SqliteStore 对于未知的 thread_id可能返回空快照（取决于后端语义）而非 ``None``。将此视为"不存在"。
    """

    async def aget_state(self, config: dict) -> _FakeState:
        return _FakeState(is_interrupted=False, is_empty=True)

    async def ainvoke(self, inputs: Any, config: dict | None = None) -> dict:
        return {"messages": []}


class _CheckpointerFailingGraph:
    """``aget_state`` 抛出异常 — 模拟数据库 / checkpointer 故障。"""

    async def aget_state(self, config: dict):
        raise RuntimeError("checkpointer DB unreachable")

    async def ainvoke(self, inputs: Any, config: dict | None = None) -> dict:
        return {"messages": []}


class TestThreadStateErrorMatrix:
    """验证 ``_verify_thread_interrupted`` 产生正确的状态码。"""

    @pytest.mark.asyncio
    async def test_approve_404_when_thread_does_not_exist(self) -> None:
        graph = _AbsentThreadGraph()
        app = _build_hitl_app(graph)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/supervisor/research/missing-thread/approve",
                json={"feedback": ""},
            )
        assert r.status_code == 404
        assert "does not exist" in r.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_approve_404_when_state_is_empty(self) -> None:
        """空状态（无 next、无 values）被视为不存在。"""
        graph = _EmptyStateGraph()
        app = _build_hitl_app(graph)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/supervisor/research/empty-thread/resume",
                json={"feedback": ""},
            )
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_approve_500_when_checkpointer_fails(self) -> None:
        graph = _CheckpointerFailingGraph()
        app = _build_hitl_app(graph)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/supervisor/research/any-thread/approve",
                json={"feedback": ""},
            )
        assert r.status_code == 500
        assert "checkpointer DB unreachable" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_resume_409_message_says_completed(self) -> None:
        """已存在但已终止的线程应返回 409，并在消息中说明线程并非不存在，只是已完成。"""
        graph = _CompletedGraph()
        app = _build_hitl_app(graph)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/supervisor/research/completed-thread/resume",
                json={"feedback": ""},
            )
        assert r.status_code == 409
        detail = r.json()["detail"].lower()
        assert "not paused" in detail
        assert "already" in detail
