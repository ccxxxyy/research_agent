"""所有 API 端点的 Pydantic 请求/响应模型。"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ---------- Supervisor（最小化版） ----------


class SupervisorChatRequest(BaseModel):
    """最小化多 Agent supervisor 的请求体。"""

    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: str | None = Field(
        None,
        description="会话线程；省略则开启新的独立会话。",
    )
    recursion_limit: int | None = Field(
        None,
        ge=4,
        le=50,
        description="可选的 LangGraph 递归上限（默认使用框架默认值）。",
    )


class SupervisorChatResponse(BaseModel):
    reply: str
    thread_id: str
    message_count: int = 0


# ---------- 研究 Supervisor----------


class ResearchSupervisorRequest(BaseModel):
    """金融研究 supervisor 的请求体。

    接受自由格式的中文或英文问题。supervisor 内部决定将子任务路由到哪些 specialist（``data_expert``、``report_expert``、``coder_expert``）。
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="自然语言研究问题。",
    )
    user_id: str = Field(
        default="anonymous",
        min_length=1,
        max_length=64,
        description=("用于长期记忆隔离的用户标识。省略则为匿名（无跨会话持久化）。"),
    )
    thread_id: str | None = Field(
        None,
        description=(
            "会话线程；省略则开启新的独立会话。跨调用复用同一 thread_id 可从检查点恢复之前的对话。"
        ),
    )
    recursion_limit: int | None = Field(
        None,
        ge=4,
        le=80,
        description=(
            "可选的 LangGraph 递归上限。默认值（框架值 25）对 3 个 specialist"
            " 路由已足够宽裕；仅在需要 >5 次连续交接时提高。"
        ),
    )
    market: str | None = Field(
        None,
        description=(
            "可选市场覆盖：``CN_A`` / ``US`` / ``MIXED`` / ``auto``（默认）。"
            "省略或 auto 时按问句信号 → 会话粘性 thread_market → 用户 preferred_market → 产品默认 CN_A 解析。"
        ),
    )
    thread_market: str | None = Field(
        None,
        description=(
            "同会话上一轮解析市场（``CN_A`` / ``US`` / ``MIXED``）。"
            "仅当问句无明确市场信号且未设 market 覆盖时作为粘性回退，避免跟进句被默认成 A 股。"
        ),
    )


class ResearchSupervisorResponse(BaseModel):
    """非流式研究端点的 JSON 响应。

    ``specialists_reached`` 来源于外层状态的 ``transfer_to_*`` 交接轨迹。
    由于图编译时使用 ``output_mode="last_message"``，specialist 内部的``fin_*``/``pdf_*``/``code_*`` 工具调用停留在其子图中，不会暴露于此；
    回复内容本身就是 specialist 完成工作的权威证据。

    若 ``cache_hit=True``，表示请求被静态知识语义缓存短路，未调用 supervisor / LLM。
    """

    reply: str = Field(..., description="Supervisor 的最终回答。")
    thread_id: str
    specialists_reached: list[str] = Field(
        default_factory=list,
        description="Supervisor 路由到的不同 specialist 列表。",
    )
    message_count: int = 0
    cache_hit: bool = Field(
        default=False,
        description="是否命中静态知识语义缓存（glossary/FAQ 等，未调用 LLM）。",
    )
    cache_domain: str | None = Field(
        default=None,
        description="命中时的缓存域：glossary / methodology / template / faq / macro / historical_event。",
    )
    market: str | None = Field(
        default=None,
        description="本次请求解析出的市场：CN_A / US / MIXED。",
    )
    market_source: str | None = Field(
        default=None,
        description="市场判定来源：query_signal / user_preference / default / request_override。",
    )


class ResearchSupervisorSSEPhase(StrEnum):
    """流式传输期间发出的 SSE 事件阶段标签。

    * ``handoff``           — supervisor 调用了 ``transfer_to_<specialist>``。
    * ``update``            — 某节点产生了状态更新（specialist 或 supervisor）。内容为该更新中最新的 assistant 消息（已截断）。
    * ``final``             — supervisor 产生了最终面向用户的回答。包含完整内容。
    * ``review_requested``  — HITL 中断：图暂停等待人工审核。内容为草稿；metadata 携带 ``thread_id`` 和 ``action_required``。
    * ``tool_call``         — specialist 调用了其 MCP 工具之一。
                              ``metadata.specialist`` 标识 Agent；
                              ``metadata.tool`` 为工具名称；
                              ``content`` 为参数的简短预览。
    * ``tool_done``         — specialist 的某次工具调用已返回（成功或错误）。
                              用于清除前端「转圈」状态，避免最后一项一直显示处理中。
    * ``token``             — 最终回答的增量 token（逐字流式输出）。``content`` 为单次 chunk 文本。
    * ``error``             — 图调用抛出异常。内容为简短错误消息；客户端应停止消费。
    * ``heartbeat``         — 合成保活事件，当图在配置的空闲窗口内无新消息时发出，以便反向代理保持 SSE 连接。
    * ``done``              — 流终止符。始终最后发出。
    """

    HANDOFF = "handoff"
    UPDATE = "update"
    TOOL_CALL = "tool_call"
    TOOL_DONE = "tool_done"
    TOKEN = "token"
    FINAL = "final"
    REVIEW_REQUESTED = "review_requested"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    DONE = "done"


class ResearchSupervisorSSEEvent(BaseModel):
    """研究 supervisor 流式传输的单条 SSE 事件负载。"""

    phase: ResearchSupervisorSSEPhase
    node: str = Field(
        "",
        description="产生该更新的图节点（如 supervisor / data_expert）。",
    )
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------- 健康检查 ----------


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = ""
    services: dict[str, str] = {}


# ---------- 恢复 / 批准（HITL）----------


class ApproveRequest(BaseModel):
    """批准 HITL 暂停的研究草稿并恢复执行。

    若 ``feedback`` 非空，会作为 ``HumanMessage`` 注入，使下游节点（反思/写作者）可以纳入审阅者的意见。空字符串表示"照原样发布"。
    """

    feedback: str = Field(
        default="",
        max_length=4000,
        description="可选的修改意见；空值 = 照原样批准。",
    )


class ResumeRequest(BaseModel):
    """带修订反馈恢复 HITL 暂停的研究任务。

    与 ``ApproveRequest`` 不同，``feedback`` 应包含可操作的修订指令。
    图将此视为 ``revise`` 操作 — human_review 节点注入反馈，
    下游反思/写作者循环据此改写。
    """

    feedback: str = Field(
        default="",
        max_length=4000,
        description="对 supervisor 草稿的修订反馈。",
    )
