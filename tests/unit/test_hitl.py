"""Unit tests for the Human-in-the-Loop (HITL) approve / resume flow.

Tests cover:
  * SSE stream emitting ``review_requested`` when the graph is interrupted.
  * ``POST /api/supervisor/research/{thread_id}/approve`` resuming the graph.
  * ``POST /api/supervisor/research/{thread_id}/resume`` with revision feedback.
  * 409 response when thread is not paused.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

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
from research_agent.memory.manager import MemoryManager
from langgraph.store.memory import InMemoryStore


# ---------------------------------------------------------------------------
# Fake graphs for HITL testing
# ---------------------------------------------------------------------------


class _FakeState:
    """Minimal stand-in for a LangGraph StateSnapshot."""

    def __init__(self, *, is_interrupted: bool = False) -> None:
        self.next = ("human_review",) if is_interrupted else ()
        self.tasks = ()


class _HITLStreamGraph:
    """Simulates a graph that streams updates then gets interrupted.

    ``astream`` yields scripted messages; ``aget_state`` reports the
    graph as interrupted so the SSE layer emits ``review_requested``.
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
    """Extends _HITLStreamGraph with ``ainvoke`` for resume testing."""

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
    """A graph that is NOT interrupted — for testing the 409 path."""

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
# Test app builder
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
        payload = frame[len("data:"):].strip()
        events.append(json.loads(payload))
    return events


# ---------------------------------------------------------------------------
# SSE review_requested emission
# ---------------------------------------------------------------------------


class TestSSEReviewRequested:
    @pytest.mark.asyncio
    async def test_stream_emits_review_requested_when_interrupted(self) -> None:
        """When the graph is interrupted, the SSE stream should emit
        a ``review_requested`` event containing the draft."""
        graph = _HITLStreamGraph([
            AIMessage(
                content="",
                name="supervisor",
                tool_calls=[{"name": "transfer_to_data_expert", "args": {}, "id": "x"}],
            ),
            AIMessage(content="specialist data", name="data_expert"),
            AIMessage(content="supervisor draft synthesis", name="supervisor"),
        ])
        app = _build_hitl_app(graph)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
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
        """Normal completed graph should NOT emit review_requested."""
        graph = _CompletedGraph()
        app = _build_hitl_app(graph)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/api/supervisor/research/stream",
                json={"query": "test no hitl"},
            )

        events = _parse_sse(r.content)
        phases = [e["phase"] for e in events]
        assert "review_requested" not in phases


# ---------------------------------------------------------------------------
# POST /approve
# ---------------------------------------------------------------------------


class TestApproveRoute:
    @pytest.mark.asyncio
    async def test_approve_resumes_interrupted_graph(self) -> None:
        graph = _HITLResumeGraph(
            scripted_messages=[AIMessage(content="draft", name="supervisor")],
            resume_reply="final after approval",
        )
        app = _build_hitl_app(graph)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
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

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
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

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/api/supervisor/research/thread-done/approve",
                json={"feedback": ""},
            )

        assert r.status_code == 409
        assert "not paused" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# POST /resume
# ---------------------------------------------------------------------------


class TestResumeRoute:
    @pytest.mark.asyncio
    async def test_resume_with_feedback(self) -> None:
        graph = _HITLResumeGraph(
            scripted_messages=[],
            resume_reply="revised with feedback",
        )
        app = _build_hitl_app(graph)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
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

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
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

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/api/supervisor/research/thread-done/resume",
                json={"feedback": "nope"},
            )

        assert r.status_code == 409
