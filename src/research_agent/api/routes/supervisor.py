"""Supervisor multi-agent endpoints — Phase-3 demo + Phase-4.5 product path."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request as FastAPIRequest, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command
from loguru import logger

from research_agent.api.dependencies import (
    MemoryDep,
    ResearchSupervisorGraphDep,
    SupervisorGraphDep,
)
from research_agent.api.schemas import (
    ApproveRequest,
    ResearchSupervisorRequest,
    ResearchSupervisorResponse,
    ResearchSupervisorSSEEvent,
    ResearchSupervisorSSEPhase,
    ResumeRequest,
    SupervisorChatRequest,
    SupervisorChatResponse,
)
from research_agent.config import get_settings
from research_agent.memory.manager import MemoryManager


def _graph_config(
    thread_id: str,
    recursion_limit: int | None,
) -> dict:
    """Build a LangGraph config dict with a safe recursion limit.

    When the caller does not specify a limit, falls back to
    ``Settings.default_recursion_limit`` (default 50) instead of
    LangGraph's built-in 25, which is too low for the 6-specialist
    research supervisor + optional reflection loop.
    """
    cfg: dict = {"configurable": {"thread_id": thread_id}}
    if recursion_limit is not None:
        cfg["recursion_limit"] = recursion_limit
    else:
        cfg["recursion_limit"] = get_settings().default_recursion_limit
    return cfg

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
    config = _graph_config(thread_id, request.recursion_limit)

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
    config = _graph_config(thread_id, request.recursion_limit)

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

_KNOWN_SPECIALISTS = frozenset({
    "data_expert", "report_expert", "coder_expert",
    "knowledge_expert", "news_expert", "sentiment_expert",
    "math_expert", "time_expert", "text_analyst",
})


def _namespace_specialist(namespace: tuple) -> str | None:
    """Extract specialist name from a subgraph namespace tuple.

    ``subgraphs=True`` yields ``(namespace, chunk)`` pairs where
    ``namespace`` is a tuple of strings tracing the nesting path:

    * ``()`` — root / parent graph.
    * ``("supervisor",)`` — inside the wrapped supervisor node
      (when reflection or HITL wraps the supervisor).
    * ``("supervisor", "data_expert")`` or ``("data_expert",)`` —
      inside a specialist subgraph.

    Returns the specialist name if found, ``None`` otherwise.
    """
    if not namespace:
        return None
    for part in namespace:
        base = str(part).split(":")[0]
        if base in _KNOWN_SPECIALISTS:
            return base
    return None


def _emit_specialist_internal(
    specialist: str,
    node_name: str,
    node_update: dict,
    frames: "asyncio.Queue[str | None]",
) -> None:
    """Push SSE frames for a specialist's internal steps.

    Only tool invocations are surfaced (``TOOL_CALL`` phase) to
    keep the stream concise.  Raw ``ToolMessage`` results are
    skipped — they are often verbose JSON payloads that add noise
    without value for the end-user.
    """
    msgs = node_update.get("messages") or []
    if not msgs:
        return
    last = msgs[-1]
    if not isinstance(last, AIMessage):
        return
    tool_calls = getattr(last, "tool_calls", None) or []
    if not tool_calls:
        return
    for tc in tool_calls:
        name = (
            tc.get("name")
            if isinstance(tc, dict)
            else getattr(tc, "name", "") or ""
        )
        args = (
            tc.get("args", {})
            if isinstance(tc, dict)
            else getattr(tc, "args", {}) or {}
        )
        if not name or name.startswith("transfer_to_"):
            continue
        args_preview = ", ".join(
            f"{k}={v!r}" for k, v in (args or {}).items()
        )[:200]
        frames.put_nowait(
            _format_sse(
                ResearchSupervisorSSEEvent(
                    phase=ResearchSupervisorSSEPhase.TOOL_CALL,
                    node=specialist,
                    content=f"{name}({args_preview})",
                    metadata={
                        "specialist": specialist,
                        "tool": str(name),
                    },
                )
            )
        )


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
    available_specialists: list[str] | None = None,
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
                        plus one synthetic opening frame ``stream opened``
                        whose ``metadata`` includes ``thread_id`` and
                        ``available_specialists`` (names compiled at
                        startup; empty if none).
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
    cfg = _graph_config(thread_id, recursion_limit)

    frames: asyncio.Queue[str | None] = asyncio.Queue()

    async def pump() -> None:
        try:
            opening_meta: dict[str, Any] = {"thread_id": thread_id}
            if available_specialists is not None:
                opening_meta["available_specialists"] = available_specialists
            await frames.put(
                _format_sse(
                    ResearchSupervisorSSEEvent(
                        phase=ResearchSupervisorSSEPhase.UPDATE,
                        node="supervisor",
                        content="stream opened",
                        metadata=opening_meta,
                    )
                )
            )

            final_emitted_local = False
            try:
                async for event in graph.astream(
                    {"messages": messages},
                    config=cfg,
                    stream_mode="updates",
                    subgraphs=True,
                ):
                    # With subgraphs=True each event is
                    # (namespace_tuple, chunk_dict).
                    if isinstance(event, tuple) and len(event) == 2:
                        namespace, chunk = event
                    elif isinstance(event, dict):
                        namespace, chunk = (), event
                    else:
                        continue
                    if not isinstance(chunk, dict):
                        continue

                    specialist_ns = _namespace_specialist(namespace)

                    for node_name, node_update in chunk.items():
                        if not isinstance(node_update, dict):
                            continue

                        # --- Specialist internal events ---
                        if specialist_ns:
                            _emit_specialist_internal(
                                specialist_ns,
                                node_name,
                                node_update,
                                frames,
                            )
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

                # --- HITL: detect graph interrupted for human review ---
                try:
                    _state = await graph.aget_state(cfg)
                    if _state and getattr(_state, "next", None):
                        outcome["graph_astream_ok"] = False
                        draft = str(
                            outcome.get("last_plain_synthesis") or ""
                        )
                        await frames.put(
                            _format_sse(
                                ResearchSupervisorSSEEvent(
                                    phase=ResearchSupervisorSSEPhase.REVIEW_REQUESTED,
                                    node="human_review",
                                    content=draft,
                                    metadata={
                                        "thread_id": thread_id,
                                        "action_required": "approve_or_revise",
                                    },
                                )
                            )
                        )
                        logger.info(
                            "HITL review requested: thread={}",
                            thread_id,
                        )
                except Exception:  # noqa: BLE001
                    pass

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
        # Shield the memory write from cancellation: if the client
        # disconnects mid-stream, Uvicorn cancels the handler task.
        # Without shield() the awaited coroutine would be cancelled
        # and the research result would be silently lost.
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.shield(
                _persist_stream_research_to_memory(
                    outcome=outcome,
                    memory=memory,
                    persist_user_id=persist_user_id,
                    persist_original_query=persist_original_query,
                    graph_input_query=_fallback_query,
                    thread_id=thread_id,
                )
            )
        runner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await runner


@router.post("/research/stream")
async def supervisor_research_stream(
    request: ResearchSupervisorRequest,
    raw_request: FastAPIRequest,
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
    The first SSE frame lists ``available_specialists`` when MCP
    tooling degraded at startup so UIs can show reduced capability.

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

    specialists: list[str] = getattr(
        raw_request.app.state, "available_specialists", None
    ) or []

    return StreamingResponse(
        _research_event_stream(
            graph,
            messages_input,
            thread_id,
            request.recursion_limit,
            memory=memory,
            persist_user_id=user_id,
            persist_original_query=request.query,
            available_specialists=specialists,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Thread-ID": thread_id,
            "X-User-ID": user_id,
        },
    )


# ---------------------------------------------------------------------------
# HITL — approve / resume a paused research thread
# ---------------------------------------------------------------------------


async def _verify_thread_interrupted(
    graph, thread_id: str
) -> None:
    """Raise the correct HTTP error if the thread is not awaiting review.

    Status code matrix:

    * ``404`` — checkpointer has *no record* of ``thread_id``
      (``aget_state`` returns ``None`` or a state with empty
      ``values`` and empty ``next``). The thread never existed.
    * ``409`` — the thread exists but is *not paused for review*
      (``state.next`` is empty). The graph has already terminated.
    * ``500`` — the checkpointer itself errored out (DB
      unreachable, schema mismatch, etc.). Caller can retry.
    """
    cfg = _graph_config(thread_id, None)
    try:
        state = await graph.aget_state(cfg)
    except Exception as exc:
        # Underlying checkpointer / DB failure — distinct from a
        # missing thread, so surface as a true 500.
        logger.exception(
            "HITL: checkpointer failed reading thread state: {}", thread_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cannot read graph state: {exc}",
        ) from exc

    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread '{thread_id}' does not exist.",
        )

    state_next = getattr(state, "next", None)
    state_values = getattr(state, "values", None) or {}

    # An "empty" state (no values, no next) means the checkpointer
    # never saw this thread_id — equivalent to "not found".
    if not state_next and not state_values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread '{thread_id}' does not exist.",
        )

    if not state_next:
        # Thread is real but already terminated.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Thread '{thread_id}' is not paused for human review. "
                "It has already been approved or completed."
            ),
        )


@router.post(
    "/research/{thread_id}/approve",
    response_model=ResearchSupervisorResponse,
)
async def supervisor_research_approve(
    thread_id: str,
    request: ApproveRequest,
    graph: ResearchSupervisorGraphDep,
) -> ResearchSupervisorResponse:
    """Approve a HITL-paused research draft and resume the graph.

    The ``human_review`` node receives ``{"action": "approve", ...}``
    via ``Command(resume=...)``, passes through without injecting
    feedback, and the graph continues to reflection (if enabled) or
    terminates.

    If ``feedback`` is non-empty, it is still forwarded as an approve
    action but the downstream reflection critic / writer will see the
    reviewer's notes in the message stream.
    """
    await _verify_thread_interrupted(graph, thread_id)

    cfg = _graph_config(thread_id, None)
    logger.info("HITL approve: thread={}", thread_id)

    result = await graph.ainvoke(
        Command(resume={"action": "approve", "feedback": request.feedback}),
        config=cfg,
    )
    messages = result.get("messages", [])
    reply = _final_assistant_text(messages)

    return ResearchSupervisorResponse(
        reply=reply,
        thread_id=thread_id,
        specialists_reached=_specialists_reached(messages),
        message_count=len(messages),
    )


@router.post(
    "/research/{thread_id}/resume",
    response_model=ResearchSupervisorResponse,
)
async def supervisor_research_resume(
    thread_id: str,
    request: ResumeRequest,
    graph: ResearchSupervisorGraphDep,
) -> ResearchSupervisorResponse:
    """Resume a HITL-paused research with revision feedback.

    The ``human_review`` node receives ``{"action": "revise", ...}``
    via ``Command(resume=...)``.  When ``feedback`` is non-empty, the
    node injects it as a ``HumanMessage`` so the reflection loop (or
    a downstream rewrite step) can address the reviewer's concerns.
    """
    await _verify_thread_interrupted(graph, thread_id)

    action = "revise" if request.feedback else "approve"
    cfg = _graph_config(thread_id, None)
    logger.info("HITL resume: thread={} action={}", thread_id, action)

    result = await graph.ainvoke(
        Command(resume={"action": action, "feedback": request.feedback}),
        config=cfg,
    )
    messages = result.get("messages", [])
    reply = _final_assistant_text(messages)

    return ResearchSupervisorResponse(
        reply=reply,
        thread_id=thread_id,
        specialists_reached=_specialists_reached(messages),
        message_count=len(messages),
    )
