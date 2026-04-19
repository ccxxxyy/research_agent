"""Research task endpoints — submit, stream, resume, approve."""

from __future__ import annotations

import uuid
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from loguru import logger

from research_agent.api.dependencies import GraphDep, MemoryDep, ModelRouterDep
from research_agent.api.schemas import (
    ApproveRequest,
    ResearchRequest,
    ResearchResponse,
    ResearchSSEEvent,
    ResearchStateResponse,
    ResumeRequest,
)

router = APIRouter(prefix="/api/research", tags=["research"])


@router.post("", response_model=ResearchResponse)
async def create_research(
    request: ResearchRequest,
    graph: GraphDep,
    memory: MemoryDep,
    model_router: ModelRouterDep,
) -> ResearchResponse:
    """Submit a new research task or resume an existing one."""
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    logger.info("Research task: thread={}, query='{}'", thread_id, request.query[:80])

    result = await graph.ainvoke(
        {"query": request.query},
        config=config,
    )

    # Persist to long-term memory
    if result.get("final_report"):
        await memory.save_research_result(
            user_id=request.user_id,
            query=request.query,
            summary=result["final_report"][:500],
            thread_id=thread_id,
        )

    return ResearchResponse(
        thread_id=thread_id,
        status=result.get("phase", "completed"),
        final_report=result.get("final_report", ""),
        quality_score=result.get("quality_score", 0.0),
        reflection_rounds=result.get("reflection_count", 0),
        usage=model_router.usage.summary(),
    )


@router.post("/stream")
async def stream_research(
    request: ResearchRequest,
    graph: GraphDep,
) -> StreamingResponse:
    """Submit a research task and stream progress via SSE."""
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator() -> AsyncIterator[str]:
        async for event in graph.astream_events(
            {"query": request.query},
            config=config,
            version="v2",
        ):
            kind = event.get("event", "")
            if kind == "on_chain_start":
                node_name = event.get("name", "")
                sse = ResearchSSEEvent(
                    phase=_map_node_to_phase(node_name),
                    agent=node_name,
                    content=f"Starting {node_name}...",
                )
                yield f"data: {sse.model_dump_json()}\n\n"
            elif kind == "on_chain_end":
                node_name = event.get("name", "")
                sse = ResearchSSEEvent(
                    phase=_map_node_to_phase(node_name),
                    agent=node_name,
                    content=f"Completed {node_name}",
                )
                yield f"data: {sse.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Thread-ID": thread_id},
    )


@router.get("/{thread_id}/state", response_model=ResearchStateResponse)
async def get_research_state(thread_id: str, graph: GraphDep) -> ResearchStateResponse:
    """Query the current execution state of a research task."""
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)

    return ResearchStateResponse(
        thread_id=thread_id,
        current_phase=state.values.get("phase", "unknown"),
        next_nodes=list(state.next) if state.next else [],
        can_resume=bool(state.next),
    )


@router.post("/{thread_id}/resume", response_model=ResearchResponse)
async def resume_research(
    thread_id: str,
    request: ResumeRequest,
    graph: GraphDep,
) -> ResearchResponse:
    """Resume a failed or paused research task from its last checkpoint."""
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)

    if not state.next:
        return ResearchResponse(
            thread_id=thread_id,
            status="already_completed",
        )

    logger.info("Resuming research: thread={}, next={}", thread_id, state.next)
    result = await graph.ainvoke(None, config=config)

    return ResearchResponse(
        thread_id=thread_id,
        status=result.get("phase", "completed"),
        final_report=result.get("final_report", ""),
        quality_score=result.get("quality_score", 0.0),
        reflection_rounds=result.get("reflection_count", 0),
    )


@router.post("/{thread_id}/approve", response_model=ResearchResponse)
async def approve_research(
    thread_id: str,
    request: ApproveRequest,
    graph: GraphDep,
) -> ResearchResponse:
    """Approve a paused task (human-in-the-loop) and optionally inject feedback."""
    config = {"configurable": {"thread_id": thread_id}}

    if request.feedback:
        await graph.aupdate_state(
            config,
            {"human_feedback": request.feedback},
        )

    result = await graph.ainvoke(None, config=config)

    return ResearchResponse(
        thread_id=thread_id,
        status=result.get("phase", "completed"),
        final_report=result.get("final_report", ""),
        quality_score=result.get("quality_score", 0.0),
        reflection_rounds=result.get("reflection_count", 0),
    )


def _map_node_to_phase(node_name: str) -> str:
    mapping = {
        "plan": "planning",
        "retrieve": "retrieving",
        "grade_retrieval": "retrieving",
        "rewrite_query": "retrieving",
        "analyze": "analyzing",
        "write": "writing",
        "reason": "reflecting",
        "finalize": "completed",
    }
    return mapping.get(node_name, "planning")
