"""主管多智能体端点 —— FastAPI 的 HTTP 路由文件，负责接收前端请求然后调用 graph 包里的图来处理。是 Web API 层，把 HTTP 请求转换成对图的调用。它和 graph 包下的 supervisor 是"调用者"和"被调用者"的关系。"""

from __future__ import annotations

import asyncio
import contextlib
import re
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, status
from fastapi import Request as FastAPIRequest
from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import Command
from loguru import logger

from research_agent.api.dependencies import (
    MemoryDep,
    ResearchSupervisorGraphDep,
    SupervisorGraphDep,
    TokenQuotaDep,
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
from research_agent.security.prompt_guard import FINANCIAL_DISCLAIMER, PromptGuard, ThreatLevel

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from research_agent.cache.semantic_cache import SemanticHit
    from research_agent.market.types import MarketResolution
    from research_agent.memory.manager import MemoryManager
    from research_agent.security.token_quota import TokenQuotaManager


def _graph_config(
    thread_id: str,
    recursion_limit: int | None,
    *,
    user_id: str = "anonymous",
) -> dict:
    """构建 LangGraph 配置字典，设置安全的递归上限。

    当调用方未指定上限时，回退到 ``Settings.default_recursion_limit``（默认 50），而非 LangGraph 内置的 25 —— 后者对 6 个专家的研究主管 + 可选反思循环而言过低。
    """
    cfg: dict = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    if recursion_limit is not None:
        cfg["recursion_limit"] = recursion_limit
    else:
        cfg["recursion_limit"] = get_settings().default_recursion_limit
    return cfg


router = APIRouter(prefix="/api/supervisor", tags=["supervisor"])
_prompt_guard = PromptGuard()

_ESTIMATED_TOKENS_PER_RESEARCH = 4000


def _check_token_quota(quota: TokenQuotaManager, user_id: str) -> None:
    """Pre-flight token quota check; raises 429 if budget exhausted. 如果用户的每日配额已用尽，则引发 HTTP 429 错误。"""
    if quota.daily_limit <= 0:
        return
    ok, remaining = quota.check_and_consume(user_id, _ESTIMATED_TOKENS_PER_RESEARCH)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Daily token quota exhausted (limit={quota.daily_limit}, "
                f"remaining={remaining}). Try again tomorrow."
            ),
            headers={"Retry-After": "3600"},
        )


async def _try_semantic_knowledge_cache(
    query: str,
    *,
    market: str | None = None,
) -> SemanticHit | None:
    """静态知识语义缓存短路：命中则跳过整个 research graph / LLM。

    仅对 glossary / FAQ / 口径 / 模板等非时效问题生效；
    含股票代码或「最新行情」类动态语义的问题直接返回 ``None``。
    ``market`` 用于 US / CN_A 域过滤（见 ADR-0006 语义缓存 US 域）。
    """
    from research_agent.cache.semantic_cache import get_semantic_cache

    cache = get_semantic_cache()
    if not cache.enabled:
        return None
    try:
        return await asyncio.to_thread(cache.lookup, query, market=market)
    except Exception as exc:  # noqa: BLE001 — 缓存故障不得阻断研究主路径
        logger.warning("semantic_cache lookup failed: {}", exc)
        return None


# ---------------------------------------------------------------------------
# 共享辅助函数
# ---------------------------------------------------------------------------


def _final_assistant_text(messages: list) -> str:
    """返回最后一条不含工具调用的助手消息内容。"""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            tc = getattr(msg, "tool_calls", None) or []
            if not tc and msg.content:
                return str(msg.content)
    return ""


def _specialists_reached(messages: list) -> list[str]:
    """提取主管路由到的去重专家列表。

    使用 ``langgraph_supervisor`` 强制的 ``transfer_to_<name>`` 工具调用约定。
    ``transfer_to_supervisor``（回交）被有意剥离，调用方仅看到实际执行工作的专家。

    保持首次出现顺序；需要稳定集合的调用方可直接 ``set(...)``。
    """
    seen: list[str] = []
    for m in messages:
        tool_calls = getattr(m, "tool_calls", None) if isinstance(m, AIMessage) else None
        for tc in tool_calls or []:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None) or ""
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
    *,
    market_override: str | None = None,
) -> tuple[list[BaseMessage], MarketResolution]:
    """构建包含可选长期上下文的图输入消息列表。

    同步与 SSE 研究路由共用此函数，确保 LLM 无论传输方式如何都看到完全相同的前导内容。

    有上下文时返回 ``[SystemMessage, HumanMessage]``，
    匿名 / 无上下文用户至少也有 MarketResolution 前导。

    同时解析市场（问句信号 → 用户偏好 → 默认），写入 SystemMessage，
    并返回 ``MarketResolution`` 供响应头 / JSON 字段使用。
    """
    from research_agent.market import format_market_preamble, resolve_market

    override = (
        None if (not market_override or market_override.lower() == "auto") else market_override
    )
    resolution = await resolve_market(
        query,
        memory=memory if user_id != "anonymous" else None,
        user_id=user_id,
        override=override,
    )

    messages: list[BaseMessage] = []
    context_parts: list[str] = [format_market_preamble(resolution, query=query)]

    if user_id != "anonymous":
        user_ctx = await memory.get_user_context(user_id)
        if user_ctx.get("preferences"):
            prefs = "; ".join(p.get("content", str(p)) for p in user_ctx["preferences"])
            context_parts.append(f"User preferences: {prefs}")
        if user_ctx.get("recent_research"):
            history_lines = [
                f"- {r.get('query', '?')}: {r.get('summary', '')[:100]}"
                for r in user_ctx["recent_research"][:3]
            ]
            context_parts.append("Recent research history:\n" + "\n".join(history_lines))

    messages.append(SystemMessage(content="\n\n".join(context_parts)))
    messages.append(HumanMessage(content=query))
    return messages, resolution


# ---------------------------------------------------------------------------
# 最小主管 —— 为交接教学演示保留
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=SupervisorChatResponse)
async def supervisor_chat(
    request: SupervisorChatRequest,
    graph: SupervisorGraphDep,
) -> SupervisorChatResponse:
    """将用户消息路由到最小主管 + 专家图。

    主管（``langgraph_supervisor.create_supervisor``）决定由哪个单工具
    专家 —— ``math_expert``、``time_expert`` 或 ``text_analyst`` ——
    处理每个子任务，然后合成最终面向用户的回答。
    """
    thread_id = request.thread_id or str(uuid.uuid4())

    verdict = _prompt_guard.check_input(request.message)
    if verdict.level == ThreatLevel.BLOCKED:
        logger.warning(
            "Prompt injection blocked in chat: thread={}, rules={}",
            thread_id,
            verdict.triggered_rules,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request blocked by security filter.",
        )

    config = _graph_config(thread_id, request.recursion_limit)

    logger.info("Supervisor chat: thread={}", thread_id)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config,
    )
    messages = result.get("messages", [])
    reply = _final_assistant_text(messages)

    # --- 输出安全过滤 ---
    out_verdict = _prompt_guard.check_output(reply)
    if out_verdict.level == ThreatLevel.BLOCKED:
        logger.warning(
            "Output leak blocked in chat: thread={}, rules={}",
            thread_id,
            out_verdict.triggered_rules,
        )
        reply = "[输出已过滤：检测到敏感信息泄漏风险]"

    return SupervisorChatResponse(
        reply=reply,
        thread_id=thread_id,
        message_count=len(messages),
    )


# ---------------------------------------------------------------------------
# 研究主管 —— 数据 / 报告 / 编码团队
# ---------------------------------------------------------------------------


@router.post("/research", response_model=ResearchSupervisorResponse)
async def supervisor_research(
    request: ResearchSupervisorRequest,
    graph: ResearchSupervisorGraphDep,
    memory: MemoryDep,
    quota: TokenQuotaDep,
) -> ResearchSupervisorResponse:
    """同步调用金融研究主管。

    记忆生命周期：
      1. 加载用户的长期上下文（偏好 + 近期研究历史）并作为系统消息前导注入。
      2. 执行研究图（短期状态由检查点器通过 thread_id 管理）。
      3. 将完成的研究结果保存到长期记忆，以支持跨会话检索。
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    user_id = request.user_id

    _check_token_quota(quota, user_id)

    verdict = _prompt_guard.check_input(request.query)
    if verdict.level == ThreatLevel.BLOCKED:
        logger.warning(
            "Prompt injection blocked in research: user={}, thread={}, rules={}",
            user_id,
            thread_id,
            verdict.triggered_rules,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request blocked by security filter.",
        )

    # --- 静态知识语义缓存：先解析市场，再按域短路 ---
    from research_agent.market import resolve_market

    resolution = await resolve_market(
        request.query,
        memory=memory if user_id != "anonymous" else None,
        user_id=user_id,
        override=(
            None if (not request.market or request.market.lower() == "auto") else request.market
        ),
    )
    cache_hit = await _try_semantic_knowledge_cache(request.query, market=resolution.market.value)
    if cache_hit is not None:
        reply = _clean_markdown(cache_hit.answer) + FINANCIAL_DISCLAIMER
        logger.info(
            "Research short-circuited by semantic_cache: domain={}, market={}, exact={}, user={}, thread={}",
            cache_hit.cache_domain,
            cache_hit.market,
            cache_hit.exact,
            user_id,
            thread_id,
        )
        return ResearchSupervisorResponse(
            reply=reply,
            thread_id=thread_id,
            specialists_reached=[],
            message_count=0,
            cache_hit=True,
            cache_domain=cache_hit.cache_domain,
            market=resolution.market.value,
            market_source=resolution.source,
        )

    config = _graph_config(thread_id, request.recursion_limit, user_id=user_id)

    logger.info("Research-supervisor invoke: user={}, thread={}", user_id, thread_id)

    # --- 长期记忆：加载用户上下文 ---
    messages_input, market_resolution = await _build_user_context_messages(
        memory,
        user_id,
        request.query,
        market_override=request.market,
    )

    result = await graph.ainvoke(
        {"messages": messages_input},
        config=config,
    )
    messages = result.get("messages", [])
    reply = _final_assistant_text(messages)

    # --- 输出安全过滤 ---
    out_verdict = _prompt_guard.check_output(reply)
    if out_verdict.level == ThreatLevel.BLOCKED:
        logger.warning(
            "Output leak blocked in research: user={}, thread={}, rules={}",
            user_id,
            thread_id,
            out_verdict.triggered_rules,
        )
        reply = "[输出已过滤：检测到敏感信息泄漏风险]"

    if out_verdict.level == ThreatLevel.SUSPICIOUS:
        logger.info(
            "Suspicious output in research: user={}, rules={}",
            user_id,
            out_verdict.triggered_rules,
        )

    # --- 清理 markdown 表格 + 金融免责声明 ---
    reply = _clean_markdown(reply) + FINANCIAL_DISCLAIMER

    # --- 长期记忆：保存研究结果 ---
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
        cache_hit=False,
        cache_domain=None,
        market=market_resolution.market.value,
        market_source=market_resolution.source,
    )


def _sse_heartbeat_interval_seconds() -> float:
    """发送心跳前的 SSE 空闲间隔（0 表示禁用）。

    从 :func:`~research_agent.config.get_settings` 读取中分离出来，
    以便单元测试可以 ``monkeypatch`` 此辅助函数而无需刷新全局配置的LRU 缓存。
    """
    return float(get_settings().sse_research_heartbeat_seconds)


def _format_sse(event: ResearchSupervisorSSEEvent) -> str:
    """将一个 SSE 事件渲染为规范的 ``data: ...\\n\\n`` 格式。"""
    return f"data: {event.model_dump_json()}\n\n"


def _clean_markdown(text: str) -> str:
    """对 LLM 输出做轻量清洗，保留 markdown 格式供前端渲染。

    仅执行：压缩连续空行为最多 1 个。
    表格、标题、粗体等 markdown 语法由前端 renderMarkdown 负责渲染。
    """
    result = re.sub(r"\n{3,}", "\n\n", text)
    return result.strip()


# Supervisor 路由阶段的「请稍候」类过渡话，绝不能当成 FINAL 发给前端。
_INTERIM_SUPERVISOR_RE = re.compile(
    r"(请稍候|请稍等|稍等一下|正在查询|正在获取|数据正在获取|"
    r"已将.*交给|交给.*专家|任务已移交|马上为您|结果出来后|"
    r"我会立即|稍后呈现|正在分析中)",
    re.IGNORECASE,
)
_REAL_SYNTHESIS_RE = re.compile(
    r"(\d+\.?\d*\s*%|[+\-−]\s*\d+\.?\d*\s*%|##\s|数据来源|涨跌幅|收盘|点位)",
)


def _looks_like_interim_supervisor_text(text: str) -> bool:
    """判断是否为移交前的占位话（不应作为最终回答）。"""
    t = (text or "").strip()
    if not t:
        return True
    return bool(_INTERIM_SUPERVISOR_RE.search(t))


def _looks_like_real_synthesis(text: str) -> bool:
    """判断是否像带数据的综合分析（可安全作为 FINAL）。"""
    t = (text or "").strip()
    if len(t) < 40:
        return False
    if _INTERIM_SUPERVISOR_RE.search(t):
        return False
    return bool(_REAL_SYNTHESIS_RE.search(t))


def _extract_update_snippet(node_update: dict) -> tuple[str, str]:
    """从 ``stream_mode='updates'`` 的载荷中提取有用的``(last_tool_call_name, text_snippet)`` 对。

    ``langgraph_supervisor`` 发出的更新形如::

        {"supervisor": {"messages": [AIMessage(tool_calls=[...])]}}
        {"data_expert": {"messages": [AIMessage(content="...")]}}

    从 ``messages`` 键中取出最新消息并分类。空 snippet 表示该节点未产生有意义的输出（如选择不流式传输的内部工具响应 ToolMessage）。
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
                first.get("name") if isinstance(first, dict) else getattr(first, "name", "") or ""
            )
            return (str(name), str(last.content or ""))
        return ("", str(last.content or ""))
    if isinstance(last, ToolMessage):
        return ("", "")
    return ("", str(getattr(last, "content", "") or ""))


_SYNTH_NODES_FOR_HISTORY = frozenset({"supervisor", "reflection"})

_KNOWN_SPECIALISTS = frozenset(
    {
        "data_expert",
        "us_data_expert",
        "us_filing_expert",
        "us_news_expert",
        "us_sentiment_expert",
        "fund_expert",
        "report_expert",
        "coder_expert",
        "knowledge_expert",
        "news_expert",
        "sentiment_expert",
        "math_expert",
        "time_expert",
        "text_analyst",
    }
)


def _namespace_specialist(namespace: tuple) -> str | None:
    """从子图命名空间元组中提取专家名称。

    ``subgraphs=True`` 产出 ``(namespace, chunk)`` 对，其中 ``namespace`` 是一个追踪嵌套路径的字符串元组：

    * ``()`` —— 根 / 父图。
    * ``("supervisor",)`` —— 在被包装的主管节点内（当反思或 HITL 包装了主管时）。
    * ``("supervisor", "data_expert")`` 或 ``("data_expert",)`` ——在专家子图内。

    找到时返回专家名称，否则返回 ``None``。
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
    frames: asyncio.Queue[str | None],
) -> None:
    """为专家的内部步骤推送 SSE 帧。

    展示 ``TOOL_CALL``；工具返回时发 ``TOOL_DONE``（不推送冗长 JSON 正文），避免最后一次调用（如 QQQ）在 UI 上一直转圈。
    """
    msgs = node_update.get("messages") or []
    if not msgs:
        return
    last = msgs[-1]
    if isinstance(last, ToolMessage):
        tool_name = str(getattr(last, "name", "") or "")
        frames.put_nowait(
            _format_sse(
                ResearchSupervisorSSEEvent(
                    phase=ResearchSupervisorSSEPhase.TOOL_DONE,
                    node=specialist,
                    content=tool_name or "done",
                    metadata={
                        "specialist": specialist,
                        "tool": tool_name,
                    },
                )
            )
        )
        return
    if not isinstance(last, AIMessage):
        return
    tool_calls = getattr(last, "tool_calls", None) or []
    if not tool_calls:
        # 专家已开始写结论：清掉仍挂着的工具转圈
        frames.put_nowait(
            _format_sse(
                ResearchSupervisorSSEEvent(
                    phase=ResearchSupervisorSSEPhase.TOOL_DONE,
                    node=specialist,
                    content="specialist_text",
                    metadata={"specialist": specialist, "tool": ""},
                )
            )
        )
        return
    for tc in tool_calls:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "") or ""
        args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {}) or {}
        if not name or name.startswith("transfer_to_"):
            continue
        args_preview = ", ".join(f"{k}={v!r}" for k, v in (args or {}).items())[:200]
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
    """当流式图正常退出时，执行与 ``POST …/research`` 相同的持久化。

    仅在 LangGraph 的 ``astream`` 未抛异常时保存 —— 与同步路由视为已提交的成功定义一致。保留主管或反思的最后一条纯文本合成作为摘要。
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
        logger.warning("Failed to persist research stream to long-term memory: {}", exc)


async def _research_event_stream(
    graph,
    messages: list[BaseMessage],
    thread_id: str,
    recursion_limit: int | None,
    *,
    user_id: str = "anonymous",
    memory: MemoryManager | None = None,
    persist_user_id: str | None = None,
    persist_original_query: str | None = None,
    available_specialists: list[str] | None = None,
    conversation_store: Any | None = None,
) -> AsyncIterator[str]:
    """为单次研究调用生成 SSE 帧的异步生成器。

    Parameters
    ----------
    messages:
        由 :func:`_build_user_context_messages` 预构建的输入列表（``[SystemMessage?, HumanMessage]``）。使用与同步路由相同的构建器以保证 LLM 前导内容一致。

    发出的阶段（大致顺序）：
      * ``handoff``   —— 每次 ``transfer_to_<specialist>`` 工具调用各一条。
      * ``update``    —— 每条非空助手消息更新各一条，外加一条合成的开启帧 ``stream opened``，其 ``metadata`` 含 ``thread_id`` 和 ``available_specialists`` （启动时编译的名称列表，无则为空）。
      * ``final``     —— 主管发出的首条不含出站工具调用的纯文本 AIMessage。
      * ``error``     —— 图抛出异常时发出。
      * ``heartbeat`` —— 图输出暂停时的空闲保活帧；间隔由``sse_research_heartbeat_seconds``（环境变量``SSE_RESEARCH_HEARTBEAT_SECONDS``，默认 ``15``；``0`` 禁用）控制。
      * ``done``      —— 始终为最后一条。

    长期记忆：与 ``supervisor_research`` 镜像。当 ``astream`` 无异常完成时，通过 :meth:`MemoryManager.save_research_result` 将主管 /反思的最后一条纯文本回复写入，
    使用 ``persist_original_query``（而非经前导膨胀的文本）。
    """
    heartbeat_interval = _sse_heartbeat_interval_seconds()

    outcome: dict[str, Any] = {
        "graph_astream_ok": False,
        "last_plain_synthesis": None,
    }
    cfg = _graph_config(thread_id, recursion_limit, user_id=user_id)

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
            final_token_buf: list[str] = []
            streaming_final = False
            # supervisor 在路由阶段会生成 "好的，..." 之类的过渡文本然后
            # 才发 tool_call（transfer_to_*）。为避免这些无用文本泄露到前端，
            # 先缓冲 supervisor 的 token，仅当确认不是路由消息后再发送。
            _sup_pending: list[str] = []
            _sup_flushing = False
            _sup_flush_threshold = 120
            handoff_count = 0

            try:
                logger.info("Research graph astream starting: thread={}", thread_id)
                _first_graph_event = True
                async for event in graph.astream(
                    {"messages": messages},
                    config=cfg,
                    stream_mode=["messages", "updates"],
                    subgraphs=True,
                ):
                    # subgraphs=True + list stream_mode → 3-tuple: (namespace, mode, data)
                    # subgraphs=False or test fakes  → 2-tuple: (mode, data)
                    if not isinstance(event, tuple) or len(event) < 2:
                        continue
                    if _first_graph_event:
                        _first_graph_event = False
                        _mode0 = event[1] if len(event) == 3 else event[0]
                        logger.info(
                            "Research graph first event: thread={} mode={}",
                            thread_id,
                            _mode0,
                        )

                    if len(event) == 3:
                        namespace_raw, mode, data = event
                    else:
                        mode, data = event[0], event[1]
                        namespace_raw = ()

                    # --- messages 模式：逐 token 流式 ---
                    if mode == "messages":
                        msg_chunk = data[0] if isinstance(data, tuple) and len(data) >= 1 else data

                        if not isinstance(msg_chunk, AIMessageChunk):
                            continue

                        chunk_text = str(msg_chunk.content or "")
                        tool_calls = getattr(msg_chunk, "tool_calls", None) or []

                        if tool_calls:
                            # supervisor 发了 tool_call → 当前轮是路由，丢弃缓冲
                            _sup_pending.clear()
                            _sup_flushing = False
                            continue
                        if not chunk_text:
                            continue

                        namespace_m = namespace_raw if isinstance(namespace_raw, tuple) else ()
                        is_root = not namespace_m
                        is_supervisor_ns = any(
                            str(p).split(":")[0] == "supervisor" for p in (namespace_m or ())
                        )
                        specialist_in_ns = (
                            _namespace_specialist(namespace_m) if namespace_m else None
                        )
                        is_supervisor_level = is_root or (is_supervisor_ns and not specialist_in_ns)

                        if is_supervisor_level:
                            if _sup_flushing:
                                streaming_final = True
                                final_token_buf.append(chunk_text)
                                await frames.put(
                                    _format_sse(
                                        ResearchSupervisorSSEEvent(
                                            phase=ResearchSupervisorSSEPhase.TOKEN,
                                            node="supervisor",
                                            content=chunk_text,
                                        )
                                    )
                                )
                            else:
                                _sup_pending.append(chunk_text)
                                pending_text = "".join(_sup_pending)
                                total_pending = len(pending_text)
                                # 「请稍候/已交给专家」类过渡话再长也不能刷成最终回答
                                if _looks_like_interim_supervisor_text(pending_text):
                                    if total_pending > 2000:
                                        _sup_pending.clear()
                                    continue
                                if total_pending >= _sup_flush_threshold:
                                    _sup_flushing = True
                                    streaming_final = True
                                    for buffered in _sup_pending:
                                        final_token_buf.append(buffered)
                                        await frames.put(
                                            _format_sse(
                                                ResearchSupervisorSSEEvent(
                                                    phase=ResearchSupervisorSSEPhase.TOKEN,
                                                    node="supervisor",
                                                    content=buffered,
                                                )
                                            )
                                        )
                                    _sup_pending.clear()
                        continue

                    # --- updates 模式：节点级事件（handoff / tool_call / 完整消息） ---
                    if mode != "updates":
                        continue

                    namespace = namespace_raw if isinstance(namespace_raw, tuple) else ()
                    chunk = data
                    if not isinstance(chunk, dict):
                        continue

                    specialist_ns = _namespace_specialist(namespace)

                    for node_name, node_update in chunk.items():
                        if not isinstance(node_update, dict):
                            continue

                        if specialist_ns:
                            _emit_specialist_internal(
                                specialist_ns,
                                node_name,
                                node_update,
                                frames,
                            )
                            continue

                        tool_call_name, snippet = _extract_update_snippet(node_update)

                        if tool_call_name.startswith("transfer_to_") and (
                            tool_call_name != "transfer_to_supervisor"
                        ):
                            specialist = tool_call_name[len("transfer_to_") :]
                            handoff_count += 1
                            # 移交开始：丢弃路由过渡缓冲，避免误当 FINAL
                            _sup_pending.clear()
                            _sup_flushing = False
                            streaming_final = False
                            final_token_buf.clear()
                            await frames.put(
                                _format_sse(
                                    ResearchSupervisorSSEEvent(
                                        phase=(ResearchSupervisorSSEPhase.HANDOFF),
                                        node=str(node_name),
                                        content=f"→ {specialist}",
                                        metadata={"specialist": specialist},
                                    )
                                )
                            )
                            continue

                        if not snippet:
                            continue

                        if not tool_call_name and str(node_name) in _SYNTH_NODES_FOR_HISTORY:
                            if not _looks_like_interim_supervisor_text(snippet):
                                outcome["last_plain_synthesis"] = snippet

                        is_supervisor_final = node_name == "supervisor" and not tool_call_name
                        if is_supervisor_final and not final_emitted_local:
                            candidate = snippet
                            if _sup_pending:
                                candidate = "".join(_sup_pending) + (
                                    snippet if snippet not in "".join(_sup_pending) else ""
                                )
                            if final_token_buf:
                                candidate = "".join(final_token_buf) or candidate

                            # 占位「请稍候」绝不能结束流；继续等移交后的综合回答
                            if _looks_like_interim_supervisor_text(candidate):
                                logger.info(
                                    "Skip interim supervisor FINAL: thread={} chars={}",
                                    thread_id,
                                    len(candidate),
                                )
                                _sup_pending.clear()
                                _sup_flushing = False
                                streaming_final = False
                                final_token_buf.clear()
                                continue

                            # 尚未移交且无实质数据时，视为路由空转，继续等待
                            if (
                                handoff_count == 0
                                and available_specialists
                                and not _looks_like_real_synthesis(candidate)
                                and len(candidate.strip()) < 280
                            ):
                                logger.info(
                                    "Skip pre-handoff weak FINAL: thread={} chars={}",
                                    thread_id,
                                    len(candidate),
                                )
                                _sup_pending.clear()
                                continue

                            final_emitted_local = True

                            # 若 pending 缓冲中还有未发送的 token，先追加到 final buf
                            if _sup_pending:
                                final_token_buf.extend(_sup_pending)
                                _sup_pending.clear()

                            if not streaming_final and not final_token_buf:
                                clean = _clean_markdown(snippet) + FINANCIAL_DISCLAIMER
                                await frames.put(
                                    _format_sse(
                                        ResearchSupervisorSSEEvent(
                                            phase=(ResearchSupervisorSSEPhase.FINAL),
                                            node=str(node_name),
                                            content=clean,
                                        )
                                    )
                                )
                            else:
                                await frames.put(
                                    _format_sse(
                                        ResearchSupervisorSSEEvent(
                                            phase=ResearchSupervisorSSEPhase.TOKEN,
                                            node="supervisor",
                                            content=FINANCIAL_DISCLAIMER,
                                        )
                                    )
                                )
                                raw_final = "".join(final_token_buf) + FINANCIAL_DISCLAIMER
                                clean_final = _clean_markdown(raw_final)
                                await frames.put(
                                    _format_sse(
                                        ResearchSupervisorSSEEvent(
                                            phase=(ResearchSupervisorSSEPhase.FINAL),
                                            node=str(node_name),
                                            content=clean_final,
                                        )
                                    )
                                )
                            continue

                        if not final_emitted_local:
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

                # --- HITL：检测图是否因人工审核而中断 ---
                hitl_pending = False
                try:
                    _state = await graph.aget_state(cfg)
                    if _state and getattr(_state, "next", None):
                        hitl_pending = True
                        outcome["graph_astream_ok"] = False
                        draft = str(outcome.get("last_plain_synthesis") or "")
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

                # 图已结束却从未发出 FINAL：用实质综合兜底，否则报错
                # （避免「请稍候/已交给专家」被当成回答完成）
                if not final_emitted_local and not hitl_pending:
                    last = str(outcome.get("last_plain_synthesis") or "").strip()
                    if last and _looks_like_real_synthesis(last):
                        final_emitted_local = True
                        clean = _clean_markdown(last) + FINANCIAL_DISCLAIMER
                        await frames.put(
                            _format_sse(
                                ResearchSupervisorSSEEvent(
                                    phase=ResearchSupervisorSSEPhase.FINAL,
                                    node="supervisor",
                                    content=clean,
                                )
                            )
                        )
                    else:
                        err = (
                            "研究未完成：主管未成功移交专家或仅输出了过渡话，请重新提问一次。"
                            if handoff_count == 0
                            else "研究未完成：专家已执行但未生成综合分析，请重新提问一次。"
                        )
                        logger.warning(
                            "Research stream ended without FINAL: thread={} handoffs={}",
                            thread_id,
                            handoff_count,
                        )
                        await frames.put(
                            _format_sse(
                                ResearchSupervisorSSEEvent(
                                    phase=ResearchSupervisorSSEPhase.ERROR,
                                    node="supervisor",
                                    content=err,
                                )
                            )
                        )

            except Exception as exc:  # noqa: BLE001
                logger.exception("Research-supervisor streaming crashed: {}", exc)
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
                    item = await asyncio.wait_for(frames.get(), timeout=heartbeat_interval)
                except TimeoutError:
                    logger.info(
                        "Research stream idle heartbeat: thread={} (waiting for LLM/tools)",
                        thread_id,
                    )
                    yield _format_sse(
                        ResearchSupervisorSSEEvent(
                            phase=(ResearchSupervisorSSEPhase.HEARTBEAT),
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
        # 将记忆写入操作从取消中屏蔽：如果客户端在流传输中途断开，Uvicorn 会取消处理器任务。若不使用 shield()，被等待的协程将被取消，研究结果会静默丢失。
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
        # 将最终助手回复写入会话历史
        if conversation_store is not None and outcome.get("last_plain_synthesis"):
            with contextlib.suppress(Exception):
                await asyncio.to_thread(
                    conversation_store.add_message,
                    thread_id,
                    user_id,
                    "assistant",
                    str(outcome["last_plain_synthesis"]),
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
    quota: TokenQuotaDep,
) -> StreamingResponse:
    """通过 SSE 流式传输研究主管工作流。

    响应类型为 ``text/event-stream``；每帧携带一个:class:`ResearchSupervisorSSEEvent`。流以单条 ``phase=done`` 帧终止（若图抛出异常则在其之前发出 ``phase=error``）。
    当 LangGraph 后端空闲（在 ``sse_research_heartbeat_seconds`` 秒内无增量，默认 **15**，设为 ``0`` 可禁用）时，
    服务器发出 ``phase=heartbeat`` 帧以保持反向代理的 SSE 连接。
    ``X-Thread-ID`` 响应头携带解析后的 thread id，客户端可复用它进行后续调用而无需解析首个事件。
    首条 SSE帧列出 ``available_specialists``， 当 MCP 工具在启动时降级时 UI 可据此展示缩减的能力。

    长期记忆：用户上下文前导由 :func:`_build_user_context_messages`构建 —— 与同步路由一致。完成的流也会自动调用
    ``MemoryManager.save_research_result``（``user_id`` 为 ``anonymous`` 时除外），使用用户原始 ``query`` 和主管 / 反思的最后一条纯文本回复。
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    user_id = request.user_id

    _check_token_quota(quota, user_id)

    verdict = _prompt_guard.check_input(request.query)
    if verdict.level == ThreatLevel.BLOCKED:
        logger.warning(
            "Prompt injection blocked in research stream: user={}, thread={}, rules={}",
            user_id,
            thread_id,
            verdict.triggered_rules,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request blocked by security filter.",
        )

    logger.info("Research-supervisor stream: user={}, thread={}", user_id, thread_id)

    # --- 静态知识语义缓存短路（流式：先解析市场，再按域短路）---
    from research_agent.market import resolve_market

    resolution = await resolve_market(
        request.query,
        memory=memory if user_id != "anonymous" else None,
        user_id=user_id,
        override=(
            None if (not request.market or request.market.lower() == "auto") else request.market
        ),
    )
    cache_hit = await _try_semantic_knowledge_cache(request.query, market=resolution.market.value)
    if cache_hit is not None:
        reply = _clean_markdown(cache_hit.answer) + FINANCIAL_DISCLAIMER

        async def _cached_event_stream() -> AsyncIterator[str]:
            yield _format_sse(
                ResearchSupervisorSSEEvent(
                    phase=ResearchSupervisorSSEPhase.FINAL,
                    node="semantic_cache",
                    content=reply,
                    metadata={
                        "cache_hit": True,
                        "cache_domain": cache_hit.cache_domain,
                        "cache_market": cache_hit.market,
                        "exact": cache_hit.exact,
                        "score": cache_hit.score,
                        "matched_question": cache_hit.matched_question,
                        "market": resolution.market.value,
                        "market_source": resolution.source,
                    },
                )
            )
            yield _format_sse(ResearchSupervisorSSEEvent(phase=ResearchSupervisorSSEPhase.DONE))

        logger.info(
            "Research stream short-circuited by semantic_cache: domain={}, cache_market={}, thread={}",
            cache_hit.cache_domain,
            cache_hit.market,
            thread_id,
        )
        return StreamingResponse(
            _cached_event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Thread-ID": thread_id,
                "X-User-ID": user_id,
                "X-Cache-Hit": "1",
                "X-Cache-Domain": cache_hit.cache_domain,
                "X-Market": resolution.market.value,
                "X-Market-Source": resolution.source,
            },
        )

    # -- 将用户消息写入会话历史 --
    conv_store = getattr(raw_request.app.state, "conversation_store", None)
    if conv_store is not None:
        title_hint = request.query[:40].replace("\n", " ")
        await asyncio.to_thread(
            conv_store.add_message,
            thread_id,
            user_id,
            "user",
            request.query,
            title_hint=title_hint,
        )

    messages_input, market_resolution = await _build_user_context_messages(
        memory,
        user_id,
        request.query,
        market_override=request.market,
    )

    specialists: list[str] = getattr(raw_request.app.state, "available_specialists", None) or []

    return StreamingResponse(
        _research_event_stream(
            graph,
            messages_input,
            thread_id,
            request.recursion_limit,
            user_id=user_id,
            memory=memory,
            persist_user_id=user_id,
            persist_original_query=request.query,
            available_specialists=specialists,
            conversation_store=conv_store,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Thread-ID": thread_id,
            "X-User-ID": user_id,
            "X-Market": market_resolution.market.value,
            "X-Market-Source": market_resolution.source,
        },
    )


# ---------------------------------------------------------------------------
# HITL —— 审批 / 恢复已暂停的研究线程
# ---------------------------------------------------------------------------


async def _verify_thread_interrupted(graph, thread_id: str) -> None:
    """若线程未处于待审核状态，则抛出正确的 HTTP 错误。
    状态码矩阵：
    * ``404`` —— 检查点器无 ``thread_id`` 记录（``aget_state`` 返回 ``None`` 或 ``values`` 和 ``next`` 均为空的状态）。线程从未存在。
    * ``409`` —— 线程存在但 *未暂停待审核*（``state.next`` 为空）。图已终止。
    * ``500`` —— 检查点器自身出错（数据库不可达、schema 不匹配等）。调用方可重试。
    """
    cfg = _graph_config(thread_id, None)
    try:
        state = await graph.aget_state(cfg)
    except Exception as exc:
        # 底层检查点器 / 数据库故障 —— 与缺失线程不同，
        # 因此作为真正的 500 返回。
        logger.exception("HITL: checkpointer failed reading thread state: {}", thread_id)
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

    # "空"状态（无 values、无 next）意味着检查点器从未见过此
    # thread_id —— 等同于"未找到"。
    if not state_next and not state_values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread '{thread_id}' does not exist.",
        )

    if not state_next:
        # 线程存在但已终止。
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
    """审批 HITL 暂停的研究草稿并恢复图执行。审核通过，让研究继续往下跑。Chat UI 里点 Approve 按钮。

    ``human_review`` 节点通过 ``Command(resume=...)`` 收到``{"action": "approve", ...}``，直接通过而不注入反馈，图继续进入反思（若启用）或终止。

    若 ``feedback`` 非空，仍作为审批动作转发，但下游反思批评者 /撰写者将在消息流中看到审核者的备注。
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
    """带修订反馈恢复 HITL 暂停的研究。带修订意见恢复；或没意见时等价 approve。Chat UI 点 Revise 按钮并输入反馈。

    ``human_review`` 节点通过 ``Command(resume=...)`` 收到 ``{"action": "revise", ...}``。
    当 ``feedback`` 非空时，节点将其作为 ``HumanMessage`` 注入，使反思循环（或下游重写步骤）能够处理审核者的意见。

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
