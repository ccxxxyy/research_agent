"""会话历史 API —— 列出、加载和删除已保存的对话记录。"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def _store(request: Request):
    store = getattr(request.app.state, "conversation_store", None)
    if store is None:
        raise HTTPException(503, "Conversation store not initialised")
    return store


@router.get("")
async def list_conversations(
    request: Request,
    user_id: str = Query(default="anonymous"),
) -> list[dict[str, Any]]:
    store = _store(request)
    return await asyncio.to_thread(store.list_conversations, user_id)


@router.get("/{thread_id}/messages")
async def get_conversation_messages(
    thread_id: str,
    request: Request,
) -> list[dict[str, Any]]:
    store = _store(request)
    return await asyncio.to_thread(store.get_messages, thread_id)


@router.delete("/{thread_id}")
async def delete_conversation(
    thread_id: str,
    request: Request,
    user_id: str = Query(default="anonymous"),
) -> dict[str, bool]:
    store = _store(request)
    ok = await asyncio.to_thread(store.delete_conversation, thread_id, user_id)
    if not ok:
        raise HTTPException(404, "Conversation not found")
    return {"deleted": True}
