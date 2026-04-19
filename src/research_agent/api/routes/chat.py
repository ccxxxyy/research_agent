"""Chat endpoint — conversational interaction with the research agent."""

from __future__ import annotations

import uuid

from fastapi import APIRouter
from loguru import logger

from research_agent.api.dependencies import GraphDep, MemoryDep
from research_agent.api.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    graph: GraphDep,
    memory: MemoryDep,
) -> ChatResponse:
    """Send a message and get a response from the research agent.

    Uses the same LangGraph pipeline but in a lighter conversational mode.
    Thread ID enables multi-turn conversation with memory.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    logger.info("Chat: thread={}, user={}", thread_id, request.user_id)

    # Retrieve user context from long-term memory
    user_context = await memory.get_user_context(request.user_id)

    result = await graph.ainvoke(
        {
            "query": request.message,
            "messages": [{"role": "user", "content": request.message}],
        },
        config=config,
    )

    reply = result.get("final_report") or result.get("analysis_result") or ""

    return ChatResponse(
        reply=reply,
        thread_id=thread_id,
    )
