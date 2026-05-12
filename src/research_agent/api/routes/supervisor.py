"""Supervisor multi-agent endpoints — Phase-3 demo + Phase-4.5 product path."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from loguru import logger

from research_agent.api.dependencies import (
    MemoryDep,
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
from research_agent.config import get_settings
from research_agent.memory.manager import MemoryManager

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


async def _build_user_context_messages(
    memory: MemoryManager,
    user_id: str,
    query: str,
) -> list[BaseMessage]:
    """Build the graph input message list with optional long-term context.

    Shared by both the synchronous and SSE research routes so the LLM
    sees exactly the same preamble regardless of transport.

    Returns ``[SystemMessage, HumanMessage]`` when context exists,
    or ``[HumanMessage]`` for anonymous / empty-context users.
    """
    messages: list[BaseMessage] = []
    if user_id != "anonymous":
        user_ctx = await memory.get_user_context(user_id)
        context_parts: list[str] = []
        if user_ctx.get("preferences"):
            prefs = "; ".join(
                p.get("content", str(p)) for p in user_ctx["preferences"]
            )
            context_parts.append(f"User preferences: {prefs}")
        if user_ctx.get("recent_research"):
            history_lines = [
                f"- {r.get('query', '?')}: {r.get('summary', '')[:100]}"
                for r in user_ctx["recent_research"][:3]
            ]
            context_parts.append(
                "Recent research history:\n" + "\n".join(history_lines)
            )
        if context_parts:
            messages.append(SystemMessage(content="\n\n".join(context_parts)))

    messages.append(HumanMessage(content=query))
    return messages


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
    memory: MemoryDep,
) -> ResearchSupervisorResponse:
    """Invoke the financial-research supervisor synchronously.

    Memory lifecycle:
      1. Load user's long-term context (preferences + recent research
         history) and inject as a system-message preamble.
      2. Execute the research graph (short-term state managed by
         the checkpointer via thread_id).
      3. Save the completed research result to long-term memory
         for cross-session retrieval.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    user_id = request.user_id
    config: dict = {"configurable": {"thread_id": thread_id}}
    if request.recursion_limit is not None:
        config["recursion_limit"] = request.recursion_limit

    logger.info(
        "Research-supervisor invoke: user={}, thread={}", user_id, thread_id
    )

    # --- Long-term memory: load user context ---
    messages_input = await _build_user_context_messages(
        memory, user_id, request.query,
    )

    result = await graph.ainvoke(
        {"messages": messages_input},
        config=config,
    )
    messages = result.get("messages", [])
    reply = _final_assistant_text(messages)

    # --- Long-term memory: save research result ---
    if user_id != "anonymous" and reply:
        try:
            await memory.save_research_result(
                user_id=user_id,
                query=request.query,
                summary=reply,
                thread_id=thread_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save research memory: {}", exc)

    return ResearchSupervisorResponse(
        reply=reply,
        thread_id=thread_id,
        specialists_reached=_specialists_reached(messages),
        message_count=len(messages),
    )


def _sse_heartbeat_interval_seconds() -> float:
    """SSE idle interval before emitting a heartbeat (0 disables).

    Separated from :func:`~research_agent.config.get_settings` reads
    so unit tests can ``monkeypatch`` this helper without flushing the
    global settings LRU cache.
    """
    return float(get_settings().sse_research_heartbeat_seconds)


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


_SYNTH_NODES_FOR_HISTORY = frozenset({"supervisor", "reflection"})


async def _persist_stream_research_to_memory(
    *,
    outcome: dict[str, Any],
    memory: MemoryManager | None,
    persist_user_id: str | None,
    persist_original_query: str | None,
    graph_input_query: str,
    thread_id: str,
) -> None:
    """Persist like ``POST …/research`` when the streamed graph exits cleanly.

    Only saves when LangGraph finishes ``astream`` without raising --- same
    success notion as treating the synchronous route as committed. Keeps the
    last plain synthesis from supervisor or reflection as the summary.
    """
    if memory is None or not persist_user_id or persist_user_id == "anonymous":
        return
    if not outcome.get("graph_astream_ok"):
        return

    reply = outcome.get("last_plain_synthesis")
    if not reply or not str(reply).strip():
        return

    canonical_query = persist_original_query or graph_input_query
    try:
        await memory.save_research_result(
            user_id=persist_user_id,
            query=canonical_query,
            summary=str(reply),
            thread_id=thread_id,
        )
        logger.info(
            "Research stream saved to long-term memory: user={}, thread={}",
            persist_user_id,
            thread_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to persist research stream to long-term memory: {}", exc
        )


async def _research_event_stream(
    graph,
    messages: list[BaseMessage],
    thread_id: str,
    recursion_limit: int | None,
    *,
    memory: MemoryManager | None = None,
    persist_user_id: str | None = None,
    persist_original_query: str | None = None,
) -> AsyncIterator[str]:
    """Async generator producing SSE frames for one research invocation.

    Parameters
    ----------
    messages:
        Pre-built input list from :func:`_build_user_context_messages`
        (``[SystemMessage?, HumanMessage]``). Using the same builder as
        the synchronous route guarantees identical LLM preamble.

    Phases emitted (in rough order):
      * ``handoff``   — one per ``transfer_to_<specialist>`` tool call.
      * ``update``    — one per non-empty assistant message update,
                        plus one synthetic opening frame ``stream opened``.
      * ``final``     — the first plain supervisor AIMessage whose last
                       message carries no outbound tool-call.
      * ``error``     — if the graph raises.
      * ``heartbeat`` — idle keep-alive when graph output pauses; spacing
                        comes from ``sse_research_heartbeat_seconds`` (env
                        ``SSE_RESEARCH_HEARTBEAT_SECONDS``, default ``15``;
                        ``0`` disables).
      * ``done``      — always last.

    Long-term memory: mirrors ``supervisor_research``. When ``astream``
    completes without exceptions, the last supervisor / reflection plain
    reply is written via :meth:`MemoryManager.save_research_result` using
    ``persist_original_query`` (never the preamble-inflated text).
    """
    heartbeat_interval = _sse_heartbeat_interval_seconds()

    outcome: dict[str, Any] = {
        "graph_astream_ok": False,
        "last_plain_synthesis": None,
    }
    cfg: dict = {"configurable": {"thread_id": thread_id}}
    if recursion_limit is not None:
        cfg["recursion_limit"] = recursion_limit

    frames: asyncio.Queue[str | None] = asyncio.Queue()

    async def pump() -> None:
        try:
            await frames.put(
                _format_sse(
                    ResearchSupervisorSSEEvent(
                        phase=ResearchSupervisorSSEPhase.UPDATE,
                        node="supervisor",
                        content="stream opened",
                        metadata={"thread_id": thread_id},
                    )
                )
            )

            final_emitted_local = False
            try:
                async for chunk in graph.astream(
                    {"messages": messages},
                    config=cfg,
                    stream_mode="updates",
                ):
                    if not isinstance(chunk, dict):
                        continue
                    for node_name, node_update in chunk.items():
                        if not isinstance(node_update, dict):
                            continue
                        tool_call_name, snippet = _extract_update_snippet(
                            node_update
                        )

                        if tool_call_name.startswith("transfer_to_") and (
                            tool_call_name != "transfer_to_supervisor"
                        ):
                            specialist = tool_call_name[len("transfer_to_") :]
                            await frames.put(
                                _format_sse(
                                    ResearchSupervisorSSEEvent(
                                        phase=(
                                            ResearchSupervisorSSEPhase.HANDOFF
                                        ),
                                        node=str(node_name),
                                        content=f"→ {specialist}",
                                        metadata={"specialist": specialist},
                                    )
                                )
                            )
                            continue

                        if not snippet:
                            continue

                        if (
                            not tool_call_name
                            and str(node_name) in _SYNTH_NODES_FOR_HISTORY
                        ):
                            outcome["last_plain_synthesis"] = snippet

                        is_supervisor_final = (
                            node_name == "supervisor"
                            and not tool_call_name
                        )
                        if is_supervisor_final and not final_emitted_local:
                            final_emitted_local = True
                            await frames.put(
                                _format_sse(
                                    ResearchSupervisorSSEEvent(
                                        phase=(
                                            ResearchSupervisorSSEPhase.FINAL
                                        ),
                                        node=str(node_name),
                                        content=snippet,
                                    )
                                )
                            )
                            continue

                        await frames.put(
                            _format_sse(
                                ResearchSupervisorSSEEvent(
                                    phase=ResearchSupervisorSSEPhase.UPDATE,
                                    node=str(node_name),
                                    content=snippet[:4096],
                                )
                            )
                        )
                outcome["graph_astream_ok"] = True
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Research-supervisor streaming crashed: {}", exc
                )
                await frames.put(
                    _format_sse(
                        ResearchSupervisorSSEEvent(
                            phase=ResearchSupervisorSSEPhase.ERROR,
                            node="supervisor",
                            content=str(exc)[:1024],
                        )
                    )
                )

            await frames.put(
                _format_sse(
                    ResearchSupervisorSSEEvent(
                        phase=ResearchSupervisorSSEPhase.DONE,
                        node="supervisor",
                    )
                )
            )
        finally:
            await frames.put(None)

    runner = asyncio.create_task(pump())
    try:
        while True:
            if heartbeat_interval > 0:
                try:
                    item = await asyncio.wait_for(
                        frames.get(), timeout=heartbeat_interval
                    )
                except asyncio.TimeoutError:
                    yield _format_sse(
                        ResearchSupervisorSSEEvent(
                            phase=(
                                ResearchSupervisorSSEPhase.HEARTBEAT
                            ),
                            node="sse",
                            content="ping",
                            metadata={"thread_id": thread_id},
                        )
                    )
                    continue
            else:
                item = await frames.get()

            if item is None:
                break
            yield item
    finally:
        _fallback_query = next(
            (str(m.content) for m in messages if isinstance(m, HumanMessage)),
            "",
        )
        await _persist_stream_research_to_memory(
            outcome=outcome,
            memory=memory,
            persist_user_id=persist_user_id,
            persist_original_query=persist_original_query,
            graph_input_query=_fallback_query,
            thread_id=thread_id,
        )
        runner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await runner


@router.post("/research/stream")
async def supervisor_research_stream(
    request: ResearchSupervisorRequest,
    graph: ResearchSupervisorGraphDep,
    memory: MemoryDep,
) -> StreamingResponse:
    """Stream the research-supervisor workflow via SSE.

    Response is ``text/event-stream``; each frame carries a
    :class:`ResearchSupervisorSSEEvent`. The stream terminates with
    a single ``phase=done`` frame (preceded by ``phase=error`` if
    the graph raised). While the LangGraph backend is idle (no deltas
    for ``sse_research_heartbeat_seconds``, default **15**, set to
    ``0`` to disable), the server emits ``phase=heartbeat`` frames so
    reverse proxies retain the SSE connection. The ``X-Thread-ID``
    response header carries the resolved thread id so that clients can
    reuse it in a follow-up call without parsing the first event.

    Long-term memory: user context preamble is built by
    :func:`_build_user_context_messages` — identical to the synchronous
    route. Completed streams also call ``MemoryManager.save_research_result``
    automatically (unless ``user_id`` is ``anonymous``), using the user's
    original ``query`` and the last supervisor / reflection plain reply.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    user_id = request.user_id
    logger.info(
        "Research-supervisor stream: user={}, thread={}", user_id, thread_id
    )

    messages_input = await _build_user_context_messages(
        memory, user_id, request.query,
    )

    return StreamingResponse(
        _research_event_stream(
            graph,
            messages_input,
            thread_id,
            request.recursion_limit,
            memory=memory,
            persist_user_id=user_id,
            persist_original_query=request.query,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Thread-ID": thread_id,
            "X-User-ID": user_id,
        },
    )
