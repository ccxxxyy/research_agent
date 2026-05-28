"""长期记忆管理端点。

将 MemoryManager 以 REST 接口暴露，使前端可以：
- 查看用户的研究历史
- 设置 / 更新用户偏好
- 查询积累的领域知识

这些端点是主管路由中自动记忆生命周期（研究完成时自动保存结果）的补充。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

from research_agent.memory.manager import MemoryNamespace

if TYPE_CHECKING:
    from research_agent.api.dependencies import MemoryDep

router = APIRouter(prefix="/api/memory", tags=["memory"])


# =====================================================================
# 请求 / 响应模型
# =====================================================================


class SavePreferenceRequest(BaseModel):
    key: str = Field(..., min_length=1, max_length=128)
    content: str = Field(..., min_length=1, max_length=2000)


class SavePreferenceResponse(BaseModel):
    user_id: str
    key: str
    saved: bool = True


class MemoryItem(BaseModel):
    key: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class ResearchHistoryResponse(BaseModel):
    user_id: str
    items: list[dict[str, Any]]


class UserContextResponse(BaseModel):
    user_id: str
    preferences: list[dict[str, Any]]
    recent_research: list[dict[str, Any]]


# =====================================================================
# 端点
# =====================================================================


@router.get("/context", response_model=UserContextResponse)
async def get_user_context(
    memory: MemoryDep,
    x_user_id: str = Header(..., alias="X-User-ID"),
) -> UserContextResponse:
    """获取完整的用户上下文（偏好 + 近期研究历史）。

    与研究主管在每次调用前注入的前导上下文相同。
    """
    ctx = await memory.get_user_context(x_user_id)
    return UserContextResponse(
        user_id=x_user_id,
        preferences=ctx.get("preferences", []),
        recent_research=ctx.get("recent_research", []),
    )


@router.get("/history", response_model=ResearchHistoryResponse)
async def get_research_history(
    memory: MemoryDep,
    x_user_id: str = Header(..., alias="X-User-ID"),
    limit: int = 10,
) -> ResearchHistoryResponse:
    """列出用户保存的研究结果（最新优先）。"""
    items = await memory.search_memories(
        user_id=x_user_id,
        namespace=MemoryNamespace.RESEARCH_HISTORY,
        limit=min(limit, 50),
    )
    return ResearchHistoryResponse(user_id=x_user_id, items=items)


@router.post(
    "/preferences",
    response_model=SavePreferenceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def save_preference(
    body: SavePreferenceRequest,
    memory: MemoryDep,
    x_user_id: str = Header(..., alias="X-User-ID"),
) -> SavePreferenceResponse:
    """保存或更新用户偏好。

    偏好会注入主管提示词中，以便个性化回答（如首选语言、分析深度、关注行业等）。
    """
    await memory.save_memory(
        user_id=x_user_id,
        namespace=MemoryNamespace.USER_PREFERENCES,
        key=body.key,
        value={"content": body.content},
    )
    return SavePreferenceResponse(user_id=x_user_id, key=body.key)


@router.delete("/preferences/{key}")
async def delete_preference(
    key: str,
    memory: MemoryDep,
    x_user_id: str = Header(..., alias="X-User-ID"),
) -> dict[str, Any]:
    """按键删除用户偏好。"""
    try:
        ns = (x_user_id, MemoryNamespace.USER_PREFERENCES)
        await memory._store.adelete(ns, key)
        return {"user_id": x_user_id, "key": key, "deleted": True}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete preference: {exc}",
        ) from exc
