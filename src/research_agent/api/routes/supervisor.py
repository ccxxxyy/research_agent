"""Supervisor multi-agent endpoint — Phase-3 orchestration demo / product path."""

from __future__ import annotations

import uuid

from fastapi import APIRouter
from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger

from research_agent.api.dependencies import SupervisorGraphDep
from research_agent.api.schemas import SupervisorChatRequest, SupervisorChatResponse

router = APIRouter(prefix="/api/supervisor", tags=["supervisor"])


def _final_assistant_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            tc = getattr(msg, "tool_calls", None) or []
            if not tc and msg.content:
                return str(msg.content)
    return ""


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
