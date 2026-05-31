"""API 测试 — research-supervisor HTTP 端点。

这些测试刻意不标记为 ``integration``，因为它们通过 FastAPI 的dependency-overrides 用假的 ``CompiledStateGraph`` 替换了真实的research-supervisor 图。
无 LLM、无 MCP 子进程、无网络 — 仅测试HTTP 层契约。

锁定的行为
-----------------
  * ``POST /api/supervisor/research`` 返回正确的 JSON 结构，
    省略时自动解析 ``thread_id``，并报告 supervisor 路由到的不同专家。
  * ``POST /api/supervisor/research/stream`` 按预期顺序发出 SSE 帧(``handoff`` → ``final`` → ``done``)，
    可选的空闲 ``heartbeat``心跳（参见 ``SSE_RESEARCH_HEARTBEAT_SECONDS``），以及携带``X-Thread-ID`` 响应头。
  * 当 lifespan 未能构建图时（``app.state.research_supervisor_graph is None``），触发 503 回退路径。
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

import research_agent.api.routes.supervisor as supervisor_route
from research_agent.api.dependencies import (
    get_memory_manager,
    get_research_supervisor_graph,
    get_supervisor_graph,
    get_token_quota,
)
from research_agent.api.routes.supervisor import router as supervisor_router
from research_agent.memory.manager import MemoryManager
from research_agent.security.token_quota import TokenQuotaManager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# ---------------------------------------------------------------------------
# 模拟图 — 模仿我们实际调用的 CompiledStateGraph 接口。
# ---------------------------------------------------------------------------


class _FakeGraph:
    """编译后 LangGraph 应用的最小替身。

    仅实现 ``ainvoke`` 和 ``astream``。两者都由构造时注入的预设消息序列驱动，使每个测试可以断言端点观察到的精确路由。
    """

    def __init__(self, scripted_messages: list[Any]) -> None:
        self._scripted = scripted_messages

    async def ainvoke(self, inputs: dict, config: dict | None = None) -> dict:
        # 端点将 HumanMessage 作为第一个输入传入；我们将其回显并
        # 前置于预设的 AI 消息之前，使 message_count 更接近真实情况。
        human = inputs["messages"][0]
        return {"messages": [human, *self._scripted]}

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


class _SlowFakeGraph(_FakeGraph):
    """在每个预设更新前添加固定延迟以模拟 LLM 空闲。"""

    async def astream(
        self,
        inputs: dict,
        config: dict | None = None,
        stream_mode: str = "updates",
        **kwargs: object,
    ) -> AsyncIterator[dict | tuple]:
        subgraphs = kwargs.get("subgraphs", False)
        for msg in self._scripted:
            await asyncio.sleep(0.12)
            node = getattr(msg, "name", None) or "supervisor"
            chunk = {node: {"messages": [msg]}}
            yield ((), chunk) if subgraphs else chunk


def _handoff(name: str) -> AIMessage:
    """构建一条看起来像 supervisor 移交调用的 AIMessage。"""
    return AIMessage(
        content="",
        name="supervisor",
        tool_calls=[{"name": f"transfer_to_{name}", "args": {}, "id": f"id-{name}"}],
    )


def _specialist_reply(name: str, text: str) -> AIMessage:
    return AIMessage(content=text, name=name)


def _supervisor_final(text: str) -> AIMessage:
    return AIMessage(content=text, name="supervisor")


def _reflection_plain(text: str) -> AIMessage:
    return AIMessage(content=text, name="reflection")


class _SpyMemory(MemoryManager):
    """记录 ``save_research_result`` 调用；可选的假历史前言。"""

    def __init__(self, *, fake_recent_research: list[dict[str, str]] | None = None) -> None:
        super().__init__(InMemoryStore())
        self._fake_recent_research = fake_recent_research
        self.save_calls: list[dict[str, Any]] = []

    async def get_user_context(self, user_id: str) -> dict[str, Any]:  # type: ignore[override]
        if self._fake_recent_research is not None:
            return {
                "preferences": [],
                "recent_research": self._fake_recent_research,
            }
        return await super().get_user_context(user_id)

    async def save_research_result(  # type: ignore[override]
        self,
        user_id: str,
        query: str,
        summary: str,
        thread_id: str,
    ) -> None:
        self.save_calls.append(
            {
                "user_id": user_id,
                "query": query,
                "summary": summary,
                "thread_id": thread_id,
            }
        )


# ---------------------------------------------------------------------------
# 可覆盖 research-supervisor 依赖的应用 fixture
# ---------------------------------------------------------------------------


def _build_test_app(
    graph: _FakeGraph | None,
    *,
    memory: MemoryManager | None = None,
    quota: TokenQuotaManager | None = None,
) -> FastAPI:
    """构建一个仅包含 supervisor 路由的精简 FastAPI 应用。

    不启动生产 lifespan（它会连接 Chroma、Postgres、MCP）。而是只注册我们关心的路由，并通过 ``dependency_overrides``直接注入依赖。
    """
    app = FastAPI()
    app.include_router(supervisor_router)

    if graph is None:
        # 模拟 "MCP 失败，图未能构建" 的情况：不设置 ``app.state.research_supervisor_graph``，让真实依赖抛出 503。
        pass
    else:
        app.dependency_overrides[get_research_supervisor_graph] = lambda: graph

    if memory is not None:
        app.dependency_overrides[get_memory_manager] = lambda: memory

    disabled_quota = quota or TokenQuotaManager(daily_limit=0)
    app.dependency_overrides[get_token_quota] = lambda: disabled_quota

    # minimal-supervisor 依赖在这里不会被执行，但不设置的话测试 URL 中的拼写错误会表现为令人困惑的 500 而非 404；注入一个简单覆盖以确保路由表完整。
    app.dependency_overrides[get_supervisor_graph] = lambda: graph

    return app


# ---------------------------------------------------------------------------
# /api/supervisor/research（非流式 JSON）
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

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/supervisor/research",
                json={"query": "分析宁德时代"},
            )

        assert r.status_code == 200
        body = r.json()
        assert body["reply"].startswith("### 核心发现")
        assert body["thread_id"]  # 解析为一个新的 UUID
        assert body["specialists_reached"] == ["data_expert", "report_expert"]
        assert body["message_count"] >= 5  # 人类消息 + 预设消息

    @pytest.mark.asyncio
    async def test_thread_id_is_echoed_when_supplied(self) -> None:
        graph = _FakeGraph([_supervisor_final("ok")])
        app = _build_test_app(graph)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/supervisor/research",
                json={"query": "hello", "thread_id": "my-fixed-thread"},
            )

        assert r.status_code == 200
        assert r.json()["thread_id"] == "my-fixed-thread"

    @pytest.mark.asyncio
    async def test_token_quota_exceeded_returns_429(self) -> None:
        graph = _FakeGraph([_supervisor_final("ok")])
        quota = TokenQuotaManager(daily_limit=1000)
        quota.check_and_consume("alice", 900)
        app = _build_test_app(graph, quota=quota)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/supervisor/research",
                json={"query": "分析宁德时代", "user_id": "alice"},
            )

        assert r.status_code == 429
        assert "quota" in r.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_503_when_graph_unavailable(self) -> None:
        """如果 lifespan 未能构建 supervisor，路由应返回 503 — 而非 500 — 这样客户端可以直接重试，无需解析堆栈跟踪。"""
        app = _build_test_app(graph=None)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/api/supervisor/research", json={"query": "hi"})

        assert r.status_code == 503
        assert "not available" in r.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_422_on_empty_query(self) -> None:
        graph = _FakeGraph([_supervisor_final("ok")])
        app = _build_test_app(graph)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/api/supervisor/research", json={"query": ""})

        assert r.status_code == 422


# ---------------------------------------------------------------------------
# /api/supervisor/research/stream (SSE)
# ---------------------------------------------------------------------------


def _parse_sse(body: bytes) -> list[dict]:
    """将 SSE 负载解码为解析后的 JSON 事件列表。

    SSE 帧由 ``\\n\\n`` 分隔，每行 ``data:`` 包含一个 JSON 对象。忽略注释/保活行。
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

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/supervisor/research/stream",
                json={"query": "go"},
            )

        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers.get("x-thread-id")

        events = _parse_sse(r.content)
        phases = [e["phase"] for e in events]

        # 1. 第一帧是我们在任何图工作之前发出的"流已打开"更新，以便客户端尽快看到进度。
        assert phases[0] == "update"
        # 2. 每个专家至少有一次移交，按顺序排列。
        handoff_specialists = [
            e["metadata"]["specialist"] for e in events if e["phase"] == "handoff"
        ]
        assert handoff_specialists == ["data_expert", "report_expert"]
        # 3. 恰好一个 ``final`` 阶段（第一条无工具调用的 supervisor纯文本消息）。
        final_frames = [e for e in events if e["phase"] == "final"]
        assert len(final_frames) == 1
        assert "final synthesis body" in final_frames[0]["content"]
        # 4. 最后一个阶段始终是 ``done``。
        assert phases[-1] == "done"

    @pytest.mark.asyncio
    async def test_stream_opening_event_includes_available_specialists(self) -> None:
        """第一个 SSE 帧回显 available_specialists，以便客户端检测降级情况。"""
        graph = _FakeGraph([_supervisor_final("ok")])
        app = _build_test_app(graph)
        app.state.available_specialists = ["data_expert", "news_expert"]

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/supervisor/research/stream",
                json={"query": "go"},
            )

        events = _parse_sse(r.content)
        opening = events[0]
        assert opening["phase"] == "update"
        assert opening["metadata"]["available_specialists"] == [
            "data_expert",
            "news_expert",
        ]

    @pytest.mark.asyncio
    async def test_stream_emits_tool_call_from_subgraph(self) -> None:
        """专家内部的工具调用以 ``tool_call`` SSE 事件的形式出现。"""

        class _SubgraphFakeGraph(_FakeGraph):
            async def astream(
                self, inputs: dict, config: dict | None = None, **kwargs: object
            ) -> AsyncIterator[tuple]:
                # 开场：supervisor 移交
                yield (
                    (),
                    {
                        "supervisor": {
                            "messages": [
                                AIMessage(
                                    content="",
                                    name="supervisor",
                                    tool_calls=[
                                        {
                                            "name": "transfer_to_data_expert",
                                            "args": {},
                                            "id": "h1",
                                        }
                                    ],
                                )
                            ]
                        }
                    },
                )
                # 子图：data_expert 调用工具
                yield (
                    ("data_expert",),
                    {
                        "agent": {
                            "messages": [
                                AIMessage(
                                    content="",
                                    name="agent",
                                    tool_calls=[
                                        {
                                            "name": "fin_get_basic_info",
                                            "args": {"code": "300750"},
                                            "id": "tc1",
                                        }
                                    ],
                                )
                            ]
                        }
                    },
                )
                # supervisor 最终回答
                yield (
                    (),
                    {"supervisor": {"messages": [AIMessage(content="done", name="supervisor")]}},
                )

        graph = _SubgraphFakeGraph([])
        app = _build_test_app(graph)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/supervisor/research/stream",
                json={"query": "test"},
            )

        events = _parse_sse(r.content)
        phases = [e["phase"] for e in events]

        assert "handoff" in phases
        assert "tool_call" in phases
        tool_events = [e for e in events if e["phase"] == "tool_call"]
        assert len(tool_events) == 1
        assert tool_events[0]["metadata"]["specialist"] == "data_expert"
        assert tool_events[0]["metadata"]["tool"] == "fin_get_basic_info"
        assert "300750" in tool_events[0]["content"]

    @pytest.mark.asyncio
    async def test_stream_error_emits_error_phase(self) -> None:
        class _BoomGraph(_FakeGraph):
            async def astream(self, *a: Any, **kw: Any) -> AsyncIterator[dict]:
                # 在抛出异常前不产出任何内容，以便测试 ``_research_event_stream`` 的 except 分支。
                if False:
                    yield {}
                raise RuntimeError("boom")

        graph = _BoomGraph([])
        app = _build_test_app(graph)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/supervisor/research/stream",
                json={"query": "go"},
            )

        assert r.status_code == 200  # 即使异步出错，SSE 也返回 200
        events = _parse_sse(r.content)
        phases = [e["phase"] for e in events]
        assert "error" in phases
        assert phases[-1] == "done"
        err_frame = next(e for e in events if e["phase"] == "error")
        assert "boom" in err_frame["content"]

    @pytest.mark.asyncio
    async def test_stream_503_when_graph_unavailable(self) -> None:
        app = _build_test_app(graph=None)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/api/supervisor/research/stream", json={"query": "hi"})

        assert r.status_code == 503

    @pytest.mark.asyncio
    async def test_stream_emits_heartbeat_when_graph_idles(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """较短的心跳间隔 + 慢速图 ⇒ 至少产生一个 ``heartbeat``。"""

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

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
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

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/supervisor/research/stream",
                json={"query": "go"},
            )

        assert r.status_code == 200
        events = _parse_sse(r.content)
        assert not any(e["phase"] == "heartbeat" for e in events)

    @pytest.mark.asyncio
    async def test_stream_persists_memory_when_user_not_anonymous(self) -> None:
        spy = _SpyMemory()
        graph = _FakeGraph([_supervisor_final("persisted synthesis")])
        app = _build_test_app(graph, memory=spy)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/api/supervisor/research/stream",
                json={
                    "query": "user original query",
                    "user_id": "alice",
                    "thread_id": "tid-stream-1",
                },
            )

        assert len(spy.save_calls) == 1
        call = spy.save_calls[0]
        assert call["user_id"] == "alice"
        assert call["query"] == "user original query"
        assert call["summary"] == "persisted synthesis"
        assert call["thread_id"] == "tid-stream-1"

    @pytest.mark.asyncio
    async def test_stream_skips_memory_for_anonymous(self) -> None:
        spy = _SpyMemory()
        graph = _FakeGraph([_supervisor_final("x")])
        app = _build_test_app(graph, memory=spy)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/api/supervisor/research/stream",
                json={"query": "q"},  # 默认 user_id 为匿名
            )

        assert spy.save_calls == []

    @pytest.mark.asyncio
    async def test_stream_skips_memory_on_graph_error(self) -> None:
        spy = _SpyMemory()

        class _BoomGraph(_FakeGraph):
            async def astream(self, *a: Any, **kw: Any) -> AsyncIterator[dict]:
                if False:
                    yield {}
                raise RuntimeError("boom")

        app = _build_test_app(_BoomGraph([]), memory=spy)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/api/supervisor/research/stream",
                json={"query": "go", "user_id": "bob"},
            )

        assert spy.save_calls == []

    @pytest.mark.asyncio
    async def test_stream_save_query_is_original_not_preamble_injected(self) -> None:
        spy = _SpyMemory(
            fake_recent_research=[{"query": "past", "summary": "past sum"}],
        )
        graph = _FakeGraph([_supervisor_final("answer")])
        app = _build_test_app(graph, memory=spy)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/api/supervisor/research/stream",
                json={"query": "CURRENT_QUESTION_ONLY", "user_id": "u3"},
            )

        assert len(spy.save_calls) == 1
        assert spy.save_calls[0]["query"] == "CURRENT_QUESTION_ONLY"

    @pytest.mark.asyncio
    async def test_stream_memory_summary_prefers_last_reflection_plain(
        self,
    ) -> None:
        spy = _SpyMemory()
        graph = _FakeGraph(
            [
                _supervisor_final("supervisor wording"),
                _reflection_plain("reflection wording"),
            ]
        )
        app = _build_test_app(graph, memory=spy)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/api/supervisor/research/stream",
                json={"query": "q", "user_id": "u4"},
            )

        assert len(spy.save_calls) == 1
        assert spy.save_calls[0]["summary"] == "reflection wording"


# ---------------------------------------------------------------------------
# 移交提取器单元级健全性检查
# ---------------------------------------------------------------------------


class TestSpecialistsReached:
    """守护 JSON 响应中使用的辅助函数的去重/排序契约。
    该辅助函数是公开响应结构的一部分，将其与 HTTP 测试分开锁定。"""

    def test_preserves_first_seen_order(self) -> None:
        from research_agent.api.routes.supervisor import _specialists_reached

        msgs = [
            HumanMessage(content="q"),
            _handoff("report_expert"),
            _handoff("data_expert"),
            _handoff("report_expert"),  # 重复项，必须被去除
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
