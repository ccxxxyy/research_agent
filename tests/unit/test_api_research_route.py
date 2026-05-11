"""Phase-4.5 API tests — research-supervisor HTTP endpoints.

These tests are deliberately *not* marked ``integration`` because
they substitute a fake ``CompiledStateGraph`` for the real
research-supervisor graph via FastAPI's dependency-overrides. No
LLM, no MCP subprocess, no network — just the HTTP-layer contract.

What we lock down
-----------------
  * ``POST /api/supervisor/research`` returns the right JSON shape,
    resolves ``thread_id`` when omitted, and reports the distinct
    specialists it saw the supervisor route to.
  * ``POST /api/supervisor/research/stream`` emits SSE frames in the
    expected order (``handoff`` → ``final`` → ``done``), optional idle
    ``heartbeat`` pings (see ``SSE_RESEARCH_HEARTBEAT_SECONDS``), and
    carries the ``X-Thread-ID`` header.
  * The 503 fallback path fires when the lifespan failed to build
    the graph (``app.state.research_supervisor_graph is None``).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage

import research_agent.api.routes.supervisor as supervisor_route
from research_agent.api.dependencies import (
    get_research_supervisor_graph,
    get_supervisor_graph,
)
from research_agent.api.routes.supervisor import router as supervisor_router


# ---------------------------------------------------------------------------
# Fake graph — mimics the CompiledStateGraph surface we actually call.
# ---------------------------------------------------------------------------


class _FakeGraph:
    """Minimal stand-in for a compiled LangGraph app.

    We implement only ``ainvoke`` and ``astream``. Both are driven
    by a scripted message trace injected at construction time so
    each test can assert the exact routing the endpoint observed.
    """

    def __init__(self, scripted_messages: list[Any]) -> None:
        self._scripted = scripted_messages

    async def ainvoke(self, inputs: dict, config: dict | None = None) -> dict:
        # The endpoint passes a HumanMessage as the first input; we
        # echo it back prepended to the scripted AI messages so
        # message_count is realistic.
        human = inputs["messages"][0]
        return {"messages": [human, *self._scripted]}

    async def astream(
        self, inputs: dict, config: dict | None = None, stream_mode: str = "updates"
    ) -> AsyncIterator[dict]:
        # Emit one update per scripted message, grouped by node.
        # The endpoint consumes ``{node_name: {"messages": [...]}}``
        # so we wrap accordingly.
        for msg in self._scripted:
            node = getattr(msg, "name", None) or "supervisor"
            yield {node: {"messages": [msg]}}


class _SlowFakeGraph(_FakeGraph):
    """Adds a fixed delay before each scripted update to simulate LLM idle."""

    async def astream(
        self,
        inputs: dict,
        config: dict | None = None,
        stream_mode: str = "updates",
    ) -> AsyncIterator[dict]:
        for msg in self._scripted:
            await asyncio.sleep(0.12)
            node = getattr(msg, "name", None) or "supervisor"
            yield {node: {"messages": [msg]}}


def _handoff(name: str) -> AIMessage:
    """Build an AIMessage that looks like a supervisor hand-off call."""
    return AIMessage(
        content="",
        name="supervisor",
        tool_calls=[{"name": f"transfer_to_{name}", "args": {}, "id": f"id-{name}"}],
    )


def _specialist_reply(name: str, text: str) -> AIMessage:
    return AIMessage(content=text, name=name)


def _supervisor_final(text: str) -> AIMessage:
    return AIMessage(content=text, name="supervisor")


# ---------------------------------------------------------------------------
# App fixture with overridable research-supervisor dep
# ---------------------------------------------------------------------------


def _build_test_app(graph: _FakeGraph | None) -> FastAPI:
    """Construct a trimmed FastAPI app with only the supervisor router.

    We do NOT boot the production lifespan (which would hit Chroma,
    Postgres, MCP). Instead we register just the router we care
    about and wire the dependency directly via ``dependency_overrides``.
    """
    app = FastAPI()
    app.include_router(supervisor_router)

    if graph is None:
        # Simulate the "MCP failed, graph was never built" case by
        # leaving ``app.state.research_supervisor_graph`` unset and
        # letting the real dependency raise the 503.
        pass
    else:
        app.dependency_overrides[get_research_supervisor_graph] = lambda: graph

    # The minimal-supervisor dep isn't exercised here, but leaving
    # it unset would make a typo in the test URL surface as a
    # confusing 500 instead of 404; wire a trivial override so the
    # routing table is fully satisfied.
    app.dependency_overrides[get_supervisor_graph] = lambda: graph

    return app


# ---------------------------------------------------------------------------
# /api/supervisor/research  (non-streaming JSON)
# ---------------------------------------------------------------------------


class TestResearchJSON:
    @pytest.mark.asyncio
    async def test_happy_path_reports_specialists(self) -> None:
        graph = _FakeGraph(
            [
                _handoff("data_expert"),
                _specialist_reply("data_expert", "ticker 300750 price 444.90"),
                _handoff("report_expert"),
                _specialist_reply("report_expert", "annual report excerpt..."),
                _supervisor_final("### 核心发现\n- 基本面稳健\n- 披露正常"),
            ]
        )
        app = _build_test_app(graph)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/api/supervisor/research",
                json={"query": "分析宁德时代"},
            )

        assert r.status_code == 200
        body = r.json()
        assert body["reply"].startswith("### 核心发现")
        assert body["thread_id"]  # resolved to a fresh UUID
        assert body["specialists_reached"] == ["data_expert", "report_expert"]
        assert body["message_count"] >= 5  # human + scripted

    @pytest.mark.asyncio
    async def test_thread_id_is_echoed_when_supplied(self) -> None:
        graph = _FakeGraph([_supervisor_final("ok")])
        app = _build_test_app(graph)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/api/supervisor/research",
                json={"query": "hello", "thread_id": "my-fixed-thread"},
            )

        assert r.status_code == 200
        assert r.json()["thread_id"] == "my-fixed-thread"

    @pytest.mark.asyncio
    async def test_503_when_graph_unavailable(self) -> None:
        """If the lifespan failed to build the supervisor, routes
        should surface a 503 — not a 500 — so clients can retry
        without parsing a stack trace."""
        app = _build_test_app(graph=None)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/api/supervisor/research", json={"query": "hi"}
            )

        assert r.status_code == 503
        assert "not available" in r.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_422_on_empty_query(self) -> None:
        graph = _FakeGraph([_supervisor_final("ok")])
        app = _build_test_app(graph)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/api/supervisor/research", json={"query": ""}
            )

        assert r.status_code == 422


# ---------------------------------------------------------------------------
# /api/supervisor/research/stream (SSE)
# ---------------------------------------------------------------------------


def _parse_sse(body: bytes) -> list[dict]:
    """Decode an SSE payload into a list of parsed JSON events.

    SSE frames are separated by ``\\n\\n`` and each ``data:`` line
    contains one JSON blob. We ignore comment/keep-alive lines.
    """
    events: list[dict] = []
    text = body.decode("utf-8")
    for frame in text.split("\n\n"):
        frame = frame.strip()
        if not frame or not frame.startswith("data:"):
            continue
        payload = frame[len("data:") :].strip()
        events.append(json.loads(payload))
    return events


class TestResearchSSE:
    @pytest.mark.asyncio
    async def test_stream_emits_handoff_then_final_then_done(self) -> None:
        graph = _FakeGraph(
            [
                _handoff("data_expert"),
                _specialist_reply("data_expert", "basics fetched"),
                _handoff("report_expert"),
                _specialist_reply("report_expert", "pdf parsed"),
                _supervisor_final("final synthesis body"),
            ]
        )
        app = _build_test_app(graph)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/api/supervisor/research/stream",
                json={"query": "go"},
            )

        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers.get("x-thread-id")

        events = _parse_sse(r.content)
        phases = [e["phase"] for e in events]

        # 1. First frame is the "stream opened" update we emit
        #    before any graph work, so clients see motion fast.
        assert phases[0] == "update"
        # 2. At least one handoff per specialist, in order.
        handoff_specialists = [
            e["metadata"]["specialist"]
            for e in events
            if e["phase"] == "handoff"
        ]
        assert handoff_specialists == ["data_expert", "report_expert"]
        # 3. Exactly one ``final`` phase (the first supervisor plain
        #    message with no tool-calls).
        final_frames = [e for e in events if e["phase"] == "final"]
        assert len(final_frames) == 1
        assert "final synthesis body" in final_frames[0]["content"]
        # 4. Last phase is always ``done``.
        assert phases[-1] == "done"

    @pytest.mark.asyncio
    async def test_stream_error_emits_error_phase(self) -> None:
        class _BoomGraph(_FakeGraph):
            async def astream(self, *a: Any, **kw: Any) -> AsyncIterator[dict]:
                # Yield nothing before raising so we exercise the
                # except branch of ``_research_event_stream``.
                if False:
                    yield {}
                raise RuntimeError("boom")

        graph = _BoomGraph([])
        app = _build_test_app(graph)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/api/supervisor/research/stream",
                json={"query": "go"},
            )

        assert r.status_code == 200  # SSE is 200 even on async error
        events = _parse_sse(r.content)
        phases = [e["phase"] for e in events]
        assert "error" in phases
        assert phases[-1] == "done"
        err_frame = next(e for e in events if e["phase"] == "error")
        assert "boom" in err_frame["content"]

    @pytest.mark.asyncio
    async def test_stream_503_when_graph_unavailable(self) -> None:
        app = _build_test_app(graph=None)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/api/supervisor/research/stream", json={"query": "hi"}
            )

        assert r.status_code == 503

    @pytest.mark.asyncio
    async def test_stream_emits_heartbeat_when_graph_idles(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Short heartbeat interval + slow graph ⇒ at least one ``heartbeat``."""

        monkeypatch.setattr(
            supervisor_route,
            "_sse_heartbeat_interval_seconds",
            lambda: 0.05,
        )
        scripted = [
            _handoff("data_expert"),
            _specialist_reply("data_expert", "slow chunk"),
            _supervisor_final("final synthesis body"),
        ]
        graph = _SlowFakeGraph(scripted)
        app = _build_test_app(graph)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/api/supervisor/research/stream",
                json={"query": "go"},
            )

        assert r.status_code == 200
        events = _parse_sse(r.content)
        phases = [e["phase"] for e in events]
        assert "heartbeat" in phases
        assert phases[-1] == "done"

        pings = [e for e in events if e["phase"] == "heartbeat"]
        assert pings
        tid = pings[0]["metadata"].get("thread_id")
        assert tid
        assert all(p["metadata"].get("thread_id") == tid for p in pings)

    @pytest.mark.asyncio
    async def test_stream_zero_heartbeat_interval_skips_ping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            supervisor_route,
            "_sse_heartbeat_interval_seconds",
            lambda: 0.0,
        )
        scripted = [
            _handoff("data_expert"),
            _specialist_reply("data_expert", "slow chunk"),
            _supervisor_final("final synthesis body"),
        ]
        graph = _SlowFakeGraph(scripted)
        app = _build_test_app(graph)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/api/supervisor/research/stream",
                json={"query": "go"},
            )

        assert r.status_code == 200
        events = _parse_sse(r.content)
        assert not any(e["phase"] == "heartbeat" for e in events)


# ---------------------------------------------------------------------------
# Hand-off extractor unit-level sanity
# ---------------------------------------------------------------------------


class TestSpecialistsReached:
    """Guard the dedup / ordering contract of the helper used in the
    JSON response. The helper is part of the public response shape
    so we lock it down separately from the HTTP tests."""

    def test_preserves_first_seen_order(self) -> None:
        from research_agent.api.routes.supervisor import _specialists_reached

        msgs = [
            HumanMessage(content="q"),
            _handoff("report_expert"),
            _handoff("data_expert"),
            _handoff("report_expert"),  # duplicate, must be dropped
            AIMessage(content="done", name="supervisor"),
        ]
        assert _specialists_reached(msgs) == ["report_expert", "data_expert"]

    def test_filters_transfer_back_to_supervisor(self) -> None:
        from research_agent.api.routes.supervisor import _specialists_reached

        msgs = [
            AIMessage(
                content="",
                tool_calls=[{"name": "transfer_to_supervisor", "args": {}, "id": "x"}],
            ),
            _handoff("data_expert"),
        ]
        assert _specialists_reached(msgs) == ["data_expert"]
