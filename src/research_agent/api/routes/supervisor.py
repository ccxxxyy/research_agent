"""Supervisor multi-agent endpoints — Phase-3 demo + Phase-4.5 product path."""

from __future__ import annotations

import uuid
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from loguru import logger

from research_agent.api.dependencies import (
    ResearchSupervisorGraphDep,
    SupervisorGraphDep,
)
from research_agent.api.schemas import (
    ResearchSupervisorRequest,
    ResearchSupervisorResponse,
    ResearchSupervisorSSEEvent,
    ResearchSupervisorSSEPhase,
    SupervisorChatRequest,
    SupervisorChatResponse,
)

router = APIRouter(prefix="/api/supervisor", tags=["supervisor"])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _final_assistant_text(messages: list) -> str:
    """Return the last non-tool-call assistant message content."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            tc = getattr(msg, "tool_calls", None) or []
            if not tc and msg.content:
                return str(msg.content)
    return ""


def _specialists_reached(messages: list) -> list[str]:
    """Extract distinct specialists the supervisor routed to.

    Uses the ``transfer_to_<name>`` tool-call convention enforced by
    ``langgraph_supervisor``. ``transfer_to_supervisor`` (back-hand-off)
    is intentionally stripped so the caller only sees what actually
    did work.

    Order is preserved by first-seen; callers that want a stable set
    can just ``set(...)`` the result.
    """
    seen: list[str] = []
    for m in messages:
        tool_calls = getattr(m, "tool_calls", None) if isinstance(m, AIMessage) else None
        for tc in tool_calls or []:
            name = (
                tc.get("name")
                if isinstance(tc, dict)
                else getattr(tc, "name", None) or ""
            )
            if (
                isinstance(name, str)
                and name.startswith("transfer_to_")
                and name != "transfer_to_supervisor"
            ):
                specialist = name[len("transfer_to_") :]
                if specialist and specialist not in seen:
                    seen.append(specialist)
    return seen


# ---------------------------------------------------------------------------
# Phase-3 minimal supervisor — kept for the handoff teaching demo
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=SupervisorChatResponse)
async def supervisor_chat(
    request: SupervisorChatRequest,
    graph: SupervisorGraphDep,
) -> SupervisorChatResponse:
    """Route a user message through the minimal supervisor + specialists graph.

    The supervisor (``langgraph_supervisor.create_supervisor``) decides
    which single-tool specialist — ``math_expert``, ``time_expert``, or
    ``text_analyst`` — should handle each subtask, then synthesises the
    final user-visible answer.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config: dict = {"configurable": {"thread_id": thread_id}}
    if request.recursion_limit is not None:
        config["recursion_limit"] = request.recursion_limit

    logger.info("Supervisor chat: thread={}", thread_id)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config,
    )
    messages = result.get("messages", [])
    reply = _final_assistant_text(messages)

    return SupervisorChatResponse(
        reply=reply,
        thread_id=thread_id,
        message_count=len(messages),
    )


# ---------------------------------------------------------------------------
# Phase-4.5 research supervisor — data / report / coder team
# ---------------------------------------------------------------------------


@router.post("/research", response_model=ResearchSupervisorResponse)
async def supervisor_research(
    request: ResearchSupervisorRequest,
    graph: ResearchSupervisorGraphDep,
) -> ResearchSupervisorResponse:
    """Invoke the financial-research supervisor synchronously.

    Blocks until the supervisor produces a final answer. For long-
    running queries or UIs that want progressive feedback, use
    ``POST /api/supervisor/research/stream`` instead.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config: dict = {"configurable": {"thread_id": thread_id}}
    if request.recursion_limit is not None:
        config["recursion_limit"] = request.recursion_limit

    logger.info("Research-supervisor invoke: thread={}", thread_id)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=request.query)]},
        config=config,
    )
    messages = result.get("messages", [])

    return ResearchSupervisorResponse(
        reply=_final_assistant_text(messages),
        thread_id=thread_id,
        specialists_reached=_specialists_reached(messages),
        message_count=len(messages),
    )


def _format_sse(event: ResearchSupervisorSSEEvent) -> str:
    """Render one SSE event in the canonical ``data: ...\n\n`` shape."""
    return f"data: {event.model_dump_json()}\n\n"


def _extract_update_snippet(node_update: dict) -> tuple[str, str]:
    """Pick an informative ``(last_tool_call_name, text_snippet)`` pair
    from a ``stream_mode='updates'`` payload.

    ``langgraph_supervisor`` emits updates shaped like::

        {"supervisor": {"messages": [AIMessage(tool_calls=[...])]}}
        {"data_expert": {"messages": [AIMessage(content="...")]}}

    We pull the newest message from the ``messages`` key and classify
    it. An empty snippet means the node produced nothing interesting
    (e.g. an internal tool-response ToolMessage we choose not to
    stream).
    """
    msgs = node_update.get("messages") or []
    if not msgs:
        return ("", "")
    last = msgs[-1]
    if isinstance(last, AIMessage):
        tool_calls = getattr(last, "tool_calls", None) or []
        if tool_calls:
            first = tool_calls[0]
            name = (
                first.get("name")
                if isinstance(first, dict)
                else getattr(first, "name", "") or ""
            )
            return (str(name), str(last.content or ""))
        return ("", str(last.content or ""))
    if isinstance(last, ToolMessage):
        return ("", "")
    return ("", str(getattr(last, "content", "") or ""))


async def _research_event_stream(
    graph, query: str, thread_id: str, recursion_limit: int | None
) -> AsyncIterator[str]:
    """Async generator producing SSE frames for one research invocation.

    Phases emitted (in rough order):
      * ``handoff``  — one per ``transfer_to_<specialist>`` tool call.
      * ``update``   — one per non-empty assistant message update.
      * ``final``    — emitted when ``stream_mode='updates'`` yields
                       an update whose last message is a plain
                       supervisor AIMessage with no tool-calls.
      * ``error``    — if the graph raises. Content is the exception
                       message, truncated.
      * ``done``     — always last (or last-before-error), so clients
                       can detect stream termination without relying
                       on connection close.

    Design note: we use ``stream_mode='updates'`` (state delta per
    node) rather than ``astream_events``, because the supervisor
    topology has many low-level events that would swamp a UI. The
    delta view matches the intuitive "a specialist just spoke"
    mental model.
    """
    config: dict = {"configurable": {"thread_id": thread_id}}
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit

    # Opening event immediately so clients know the stream is live
    # before any LLM round-trip has completed.
    yield _format_sse(
        ResearchSupervisorSSEEvent(
            phase=ResearchSupervisorSSEPhase.UPDATE,
            node="supervisor",
            content="stream opened",
            metadata={"thread_id": thread_id},
        )
    )

    final_emitted = False
    try:
        async for chunk in graph.astream(
            {"messages": [HumanMessage(content=query)]},
            config=config,
            stream_mode="updates",
        ):
            if not isinstance(chunk, dict):
                continue
            for node_name, node_update in chunk.items():
                if not isinstance(node_update, dict):
                    continue
                tool_call_name, snippet = _extract_update_snippet(node_update)

                if tool_call_name.startswith("transfer_to_") and tool_call_name != (
                    "transfer_to_supervisor"
                ):
                    specialist = tool_call_name[len("transfer_to_") :]
                    yield _format_sse(
                        ResearchSupervisorSSEEvent(
                            phase=ResearchSupervisorSSEPhase.HANDOFF,
                            node=str(node_name),
                            content=f"→ {specialist}",
                            metadata={"specialist": specialist},
                        )
                    )
                    continue

                if not snippet:
                    continue

                # Plain assistant message from the supervisor with
                # no outgoing transfer → this is (one of) the final
                # synthesis chunks. We emit at most one ``final``
                # per stream; additional plain supervisor messages
                # (if any) fall through as ``update``.
                is_supervisor_final = (
                    node_name == "supervisor" and not tool_call_name
                )
                if is_supervisor_final and not final_emitted:
                    final_emitted = True
                    yield _format_sse(
                        ResearchSupervisorSSEEvent(
                            phase=ResearchSupervisorSSEPhase.FINAL,
                            node=str(node_name),
                            content=snippet,
                        )
                    )
                    continue

                yield _format_sse(
                    ResearchSupervisorSSEEvent(
                        phase=ResearchSupervisorSSEPhase.UPDATE,
                        node=str(node_name),
                        # Long specialist replies can exceed typical
                        # SSE chunking comfort zones; cap at 4 KiB.
                        # Clients that need the full text can follow
                        # up with the non-streaming endpoint.
                        content=snippet[:4096],
                    )
                )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Research-supervisor streaming crashed: {}", exc)
        yield _format_sse(
            ResearchSupervisorSSEEvent(
                phase=ResearchSupervisorSSEPhase.ERROR,
                node="supervisor",
                content=str(exc)[:1024],
            )
        )

    yield _format_sse(
        ResearchSupervisorSSEEvent(
            phase=ResearchSupervisorSSEPhase.DONE, node="supervisor"
        )
    )


@router.post("/research/stream")
async def supervisor_research_stream(
    request: ResearchSupervisorRequest,
    graph: ResearchSupervisorGraphDep,
) -> StreamingResponse:
    """Stream the research-supervisor workflow via SSE.

    Response is ``text/event-stream``; each frame carries a
    :class:`ResearchSupervisorSSEEvent`. The stream terminates with
    a single ``phase=done`` frame (preceded by ``phase=error`` if
    the graph raised). The ``X-Thread-ID`` response header carries
    the resolved thread id so that clients can reuse it in a
    follow-up call without parsing the first event.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    logger.info("Research-supervisor stream: thread={}", thread_id)

    return StreamingResponse(
        _research_event_stream(
            graph, request.query, thread_id, request.recursion_limit
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Thread-ID": thread_id},
    )
