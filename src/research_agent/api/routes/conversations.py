"""会话历史 API —— 列出、加载、重命名、置顶和删除已保存的对话记录。"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def _store(request: Request):
    store = getattr(request.app.state, "conversation_store", None)
    if store is None:
        raise HTTPException(503, "Conversation store not initialised")
    return store


class RenameBody(BaseModel):
    title: str = Field(..., min_length=1, max_length=80)


class PinBody(BaseModel):
    pinned: bool = True


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
    user_id: str = Query(default="anonymous"),
) -> list[dict[str, Any]]:
    store = _store(request)
    return await asyncio.to_thread(store.get_messages, thread_id, user_id)


@router.patch("/{thread_id}/title")
async def rename_conversation(
    thread_id: str,
    body: RenameBody,
    request: Request,
    user_id: str = Query(default="anonymous"),
) -> dict[str, Any]:
    store = _store(request)
    ok = await asyncio.to_thread(store.rename_conversation, thread_id, user_id, body.title)
    if not ok:
        raise HTTPException(404, "Conversation not found or empty title")
    return {"thread_id": thread_id, "title": body.title.strip()[:80]}


@router.patch("/{thread_id}/pin")
async def pin_conversation(
    thread_id: str,
    body: PinBody,
    request: Request,
    user_id: str = Query(default="anonymous"),
) -> dict[str, Any]:
    store = _store(request)
    ok = await asyncio.to_thread(store.set_pinned, thread_id, user_id, body.pinned)
    if not ok:
        raise HTTPException(404, "Conversation not found")
    return {"thread_id": thread_id, "pinned": body.pinned}


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
