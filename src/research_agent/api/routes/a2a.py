"""A2A (Agent-to-Agent) 协议端点。

实现 Google A2A 协议规范的核心子集：
- Agent Card 发现（/.well-known/agent.json）
- 任务提交（POST /a2a/tasks/send）
- 任务状态查询（GET /a2a/tasks/{task_id}）
- 任务取消（POST /a2a/tasks/{task_id}/cancel）

A2A 使外部 Agent 能通过标准化协议调用本系统的金融研究能力，实现跨服务、跨信任域的 Agent-to-Agent 水平协作。
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from loguru import logger
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# A2A Data Models (based on Google A2A spec) 数据模型（基于 Google A2A 规范）
# ---------------------------------------------------------------------------


class TaskState(StrEnum):
    """A2A 任务状态机。"""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class AgentSkill(BaseModel):
    """Agent Card 中声明的单项能力。"""

    id: str
    name: str
    description: str
    tags: list[str] = []
    examples: list[str] = []


class AgentCard(BaseModel):
    """A2A Agent Card — 能力声明。"""

    name: str
    description: str
    url: str
    version: str = "1.0.0"
    protocol_version: str = "0.2"
    capabilities: dict[str, Any] = Field(default_factory=dict)
    skills: list[AgentSkill] = []
    default_input_modes: list[str] = Field(default_factory=lambda: ["text/plain"])
    default_output_modes: list[str] = Field(default_factory=lambda: ["text/plain"])


class TextPart(BaseModel):
    """A2A 消息中的文本部分。"""

    type: str = "text"
    text: str


class Message(BaseModel):
    """A2A 消息。"""

    role: str
    parts: list[TextPart]


class TaskArtifact(BaseModel):
    """任务产出物。"""

    name: str = "research_report"
    parts: list[TextPart]


class TaskStatus(BaseModel):
    """任务状态快照。"""

    state: TaskState
    message: Message | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


class Task(BaseModel):
    """A2A 任务实体。"""

    id: str
    status: TaskStatus
    artifacts: list[TaskArtifact] = []
    history: list[Message] = []
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskSendRequest(BaseModel):
    """发送任务请求体。"""

    id: str | None = None
    message: Message
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskSendResponse(BaseModel):
    """发送任务响应体。"""

    id: str
    jsonrpc: str = "2.0"
    result: Task


# ---------------------------------------------------------------------------
# In-memory task store (production would use persistent storage) 内存任务存储（生产环境应用持久化存储）
# ---------------------------------------------------------------------------

_tasks: dict[str, Task] = {}
_task_futures: dict[str, asyncio.Task] = {}


# ---------------------------------------------------------------------------
# Router 路由定义
# ---------------------------------------------------------------------------

router = APIRouter(tags=["a2a"])


def _build_agent_card(base_url: str = "") -> AgentCard:
    """构建 Agent Card。"""
    return AgentCard(
        name="Research Agent",
        description=(
            "基于 LangGraph 的多智能体金融研究系统。"
            "支持 A 股财务数据分析、年报解读、新闻舆情分析、知识库检索等深度研究任务。"
        ),
        url=f"{base_url}/a2a",
        version="0.1.0",
        protocol_version="0.2",
        capabilities={
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        skills=[
            AgentSkill(
                id="financial_research",
                name="金融研究分析",
                description=(
                    "对 A 股上市公司进行深度财务分析，涵盖财务指标、年报解读、行业对比和投资建议。"
                ),
                tags=["finance", "research", "analysis", "A-share"],
                examples=[
                    "分析宁德时代2023年业绩表现",
                    "对比贵州茅台和五粮液的财务指标",
                    "总结比亚迪最新年报的风险因素",
                ],
            ),
            AgentSkill(
                id="knowledge_retrieval",
                name="知识库检索",
                description=(
                    "从已导入的研究文档知识库中检索相关信息，"
                    "支持混合检索（BM25+向量）和交叉编码器重排序。"
                ),
                tags=["rag", "retrieval", "knowledge"],
                examples=[
                    "搜索关于ESG披露的研究报告",
                    "查找碳中和相关政策文件",
                ],
            ),
            AgentSkill(
                id="news_sentiment",
                name="新闻舆情分析",
                description=(
                    "获取并分析上市公司相关新闻，提供舆情评分和趋势判断。"
                ),
                tags=["news", "sentiment", "nlp"],
                examples=[
                    "最近一周关于宁德时代的新闻舆情如何",
                    "分析比亚迪近期负面新闻",
                ],
            ),
        ],
    )


@router.get("/.well-known/agent.json", response_model=AgentCard)
async def get_agent_card(request: Request) -> AgentCard:
    """A2A Agent Card 发现端点。

    外部 Agent 通过此端点了解本系统的能力、支持的输入/输出模式，以及可调用的技能列表。
    """
    base_url = str(request.base_url).rstrip("/")
    return _build_agent_card(base_url)


@router.post("/a2a/tasks/send", response_model=TaskSendResponse)
async def send_task(
    request: Request,
    body: TaskSendRequest,
) -> TaskSendResponse:
    """提交 A2A 任务。

    接收外部 Agent 的研究请求，创建任务并异步执行。
    任务状态遵循 A2A 生命周期：submitted → working → completed/failed。
    """
    task_id = body.id or str(uuid.uuid4())

    if task_id in _tasks:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Task '{task_id}' already exists.",
        )

    user_text = " ".join(part.text for part in body.message.parts)
    if not user_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message must contain non-empty text parts.",
        )

    task = Task(
        id=task_id,
        status=TaskStatus(
            state=TaskState.SUBMITTED,
            message=Message(
                role="agent",
                parts=[TextPart(text="Task received, queuing for execution.")],
            ),
        ),
        history=[body.message],
        metadata=body.metadata,
    )
    _tasks[task_id] = task

    bg_task = asyncio.create_task(_execute_task(task_id, user_text, request.app))
    _task_futures[task_id] = bg_task

    return TaskSendResponse(id=task_id, result=task)


@router.get("/a2a/tasks/{task_id}", response_model=Task)
async def get_task(task_id: str) -> Task:
    """查询 A2A 任务状态。"""
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )
    return task


@router.post("/a2a/tasks/{task_id}/cancel", response_model=Task)
async def cancel_task(task_id: str) -> Task:
    """取消 A2A 任务。"""
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )

    if task.status.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Task '{task_id}' is already in terminal state: {task.status.state}.",
        )

    future = _task_futures.get(task_id)
    if future and not future.done():
        future.cancel()

    task.status = TaskStatus(
        state=TaskState.CANCELED,
        message=Message(
            role="agent",
            parts=[TextPart(text="Task canceled by caller.")],
        ),
    )
    return task


# ---------------------------------------------------------------------------
# Background task execution 后台任务执行逻辑
# ---------------------------------------------------------------------------


async def _execute_task(task_id: str, query: str, app: Any) -> None:
    """异步执行研究任务并更新状态。"""
    task = _tasks[task_id]

    task.status = TaskStatus(
        state=TaskState.WORKING,
        message=Message(
            role="agent",
            parts=[TextPart(text="Research in progress...")],
        ),
    )

    try:
        graph = getattr(app.state, "research_supervisor_graph", None)
        if graph is None:
            raise RuntimeError("Research supervisor graph not available")

        from langchain_core.messages import HumanMessage

        thread_id = f"a2a-{task_id}"
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 50,
        }

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config=config,
        )

        messages = result.get("messages", [])
        reply = ""
        from langchain_core.messages import AIMessage
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                tc = getattr(msg, "tool_calls", None) or []
                if not tc and msg.content:
                    reply = str(msg.content)
                    break

        task.artifacts = [
            TaskArtifact(
                name="research_report",
                parts=[TextPart(text=reply or "No output generated.")],
            )
        ]
        task.status = TaskStatus(
            state=TaskState.COMPLETED,
            message=Message(
                role="agent",
                parts=[TextPart(text="Research completed successfully.")],
            ),
        )
        logger.info("A2A task completed: {}", task_id)

    except asyncio.CancelledError:
        task.status = TaskStatus(
            state=TaskState.CANCELED,
            message=Message(
                role="agent",
                parts=[TextPart(text="Task was canceled.")],
            ),
        )
    except Exception as exc:
        logger.exception("A2A task failed: {}", task_id)
        task.status = TaskStatus(
            state=TaskState.FAILED,
            message=Message(
                role="agent",
                parts=[TextPart(text=f"Task failed: {exc}")],
            ),
        )
