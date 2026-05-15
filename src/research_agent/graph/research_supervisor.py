"""Phase-4.4 research supervisor — the financial-research workflow.

Why a SEPARATE graph from ``minimal_supervisor.py``?
----------------------------------------------------
``minimal_supervisor.py`` is a teaching scaffold: three toy
``@tool``-backed specialists plus an optional MCP coder. It exists to
demonstrate the supervisor pattern itself, in isolation, with no
network dependencies.

This graph is the **real product**. It orchestrates three
MCP-delivered specialists that share the same A-share / 巨潮资讯
data surface the Agent will hit in production:

              ┌───────────────────────────────────────────────────┐
              │              research_supervisor                     │ ← HEAVY tier LLM
              └─┬────────┬───────────┬─────────┬───────┬──────────┬─┘
                │        │           │         │       │          │
                ▼        ▼           ▼         ▼       ▼          ▼
          data_expert report_expert coder  news_expert knowledge sentiment
          (fin_data)  (pdf_report)  _expert (news_srv) _expert   _expert
                                   (code)              (knowledge)(sentiment)

Typical flow for "分析 宁德时代 2023 年业绩 + ESG 披露中提到的碳中和承诺":

  1. supervisor → data_expert : pull financial abstract + indicators
  2. supervisor → report_expert: locate 2023 annual-report PDF and
     extract the 经营情况 / 风险因素 sections
  3. supervisor → coder_expert: compute derived ratios or sanity-
     check numbers from the two previous outputs
  4. supervisor → knowledge_expert: search the user's previously
     ingested ESG library for "碳中和" mentions (corrective-RAG:
     the agent reads the per-call ``quality`` signal and rewrites
     the query if hits are weak, up to 3 attempts)
  5. supervisor writes a final synthesis.

Design choices that matter for interview-grade storytelling
-----------------------------------------------------------
- Each specialist owns a **disjoint** toolset. No overlap means the
  supervisor's routing choice is unambiguous — a common failure mode
  in naive "one agent with all tools" designs.
- Specialists run on :class:`ModelTier.MEDIUM` (via
  :attr:`AgentName.ANALYST`), supervisor on HEAVY. The supervisor
  does the hard reasoning (planning + synthesis); specialists do
  targeted tool-calling only.
- The supervisor prompt lists **exactly** the tools each specialist
  owns. This is critical: the supervisor decides routing by reading
  its own system prompt, not by peeking at specialist toolbelts.
- ``output_mode="last_message"`` keeps the shared state compact.
  Switch to ``full_history`` only when debugging a routing loop.
- Building a specialist is lazy — if the caller didn't supply its
  MCP tools, that specialist is simply left out of the team and the
  supervisor prompt is trimmed accordingly. This lets unit tests
  compile a 1- or 2-specialist graph without spawning the full
  subprocess fleet.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from langgraph_supervisor import create_supervisor
from loguru import logger

from research_agent.agents.specialists import (
    build_coder_expert,
    build_data_expert,
    build_knowledge_expert,
    build_news_expert,
    build_report_expert,
    build_sentiment_expert,
)
from research_agent.graph.reflection import build_reflection_subgraph
from research_agent.llm.provider import ModelRouter
from research_agent.llm.tier import ModelTier


SUPERVISOR_PROMPT_BASE = """\
You are the Financial Research Supervisor. You coordinate a small
team of specialists to produce concise, cited answers about
A-share-listed companies. Your default language follows the user's —
if the user writes in Chinese, answer in Chinese.

Team roster:
"""

SUPERVISOR_PROMPT_DATA = """\
  - data_expert   : A-share market & fundamentals via akshare MCP.
      Toolbelt (tool names may be prefixed by the MCP server key):
        * fin_search_stock_by_name    — name → 6-digit ticker lookup
        * fin_get_stock_basic_info    — company profile / latest price
        * fin_get_stock_price_history — OHLCV + summary stats
        * fin_get_financial_abstract  — revenue / profit / cash flow
        * fin_get_financial_indicators — ROE / margins / leverage
      Delegate when the user asks for: latest price / market cap /
      industry classification, OHLCV history, quarterly/annual
      financials, key ratios, or name→ticker resolution.
"""

SUPERVISOR_PROMPT_REPORT = """\
  - report_expert : 巨潮资讯 disclosure PDFs.
      Toolbelt:
        * pdf_search_announcements    — list annual / quarterly /
                                        disclosure filings in a date range
        * pdf_download_pdf            — cache-aware fetch
        * pdf_extract_pdf_metadata    — num_pages / title / author
        * pdf_parse_pdf_pages         — page-windowed text extraction
                                        (max 20 pages per call)
      Delegate when the user asks for: annual/quarterly report
      content, 经营情况讨论与分析, 风险因素, 业绩预告 / 预增 / 预减,
      or any excerpt from a 巨潮资讯 PDF.
"""

SUPERVISOR_PROMPT_CODER = """\
  - coder_expert  : sandboxed Python execution via MCP. Safe builtins
      only (math / statistics / json / collections pre-imported).
      Delegate when the user needs a derived metric (mean / std /
      growth rate), a sort / filter over returned rows, or any other
      computation that the other specialists did not pre-compute.
"""

SUPERVISOR_PROMPT_NEWS = """\
  - news_expert   : A-share news & sentiment via 东方财富 / 财联社 /
      百度财经 / 雪球. Toolbelt:
        * news_get_stock_news        — recent news for a 6-digit
                                       ticker (东方财富 individual feed)
        * news_get_market_telegraph  — real-time market flashes from
                                       财联社 (filter category: only
                                       "全部" or "重点" are supported)
        * news_get_hot_keywords      — trending themes / keywords
                                       co-occurring with a ticker
        * news_get_economic_news     — daily macro / policy digest
                                       (百度财经 早晚报)
        * news_get_xueqiu_discussion_hot_rank — 雪球讨论热度个股榜
                                       (``ranking``: ``"最热门"`` or
                                       ``"本周新增"``; wraps
                                       ``stock_hot_tweet_xq`` — rows
                                       are stocks, not post threads)
      Delegate when the user asks: "<公司>最近的新闻 / 舆情 / 热度",
      "今天 A 股有什么大事 / 重要快讯", "市场对 <公司> 的情绪如何",
      "最近的宏观 / 政策 / 央行新闻", "雪球讨论榜 / 雪球最热标的".
      Do NOT route here for raw numerical / fundamentals data, or
      for official disclosures like 年报 / 公告 — those belong to
      other specialists.
"""

SUPERVISOR_PROMPT_KNOWLEDGE = """\
  - knowledge_expert : the USER's private PDF library, indexed in a
      persistent FAISS vector store with hybrid (vector + BM25 +
      cross-encoder rerank) retrieval. Toolbelt:
        * knowledge_list_collections — enumerate user collections
        * knowledge_ingest_pdf       — chunk + embed a local PDF
                                       into a collection
        * knowledge_search           — hybrid search with reranking;
                                       returns a ``quality`` label
                                       and per-hit ``rerank_score``
                                       so the expert runs an internal
                                       corrective-RAG loop
        * knowledge_delete_collection
      Delegate when the user asks something only the user's own
      uploaded documents could answer: "我之前上传的 ESG 报告里
      关于碳中和怎么写的"、"我那份招股说明书里的募投项目"、
      或追问"把这份报告灌进我的知识库并按xx检索". Do NOT route
      generic A-share market or public-disclosure questions here —
      this expert only sees what the user has personally uploaded.
"""

SUPERVISOR_PROMPT_SENTIMENT = """\
  - sentiment_expert : 结构化新闻情感量化分析（SnowNLP + 金融关键词
      词典，确定性模型，不走大模型打分）。Toolbelt:
        * sentiment_get_stock_sentiment_report — 一站式个股舆情报告：
            拉东财新闻 → 逐条打分 → 聚合。返回每条新闻的
            ``sentiment_score ∈ [-1, 1]``、标签（正面/中性/负面）、
            命中关键词、文本指纹 + 聚合统计（正/负/中性比例、均分、
            样本量）+ 审计元数据（模型版本 + 时间戳）。
        * sentiment_analyze_text_sentiment — 纯文本批量打分。传入
            任意中文文本列表，返回逐条分数 + 聚合。可用于对其他
            专家返回的文本做二次情感标注。
      Delegate when the user asks for: 个股舆情量化（"宁德时代最近
      舆情如何 / 市场情绪"）、新闻情感打分（"帮我分析这几条新闻的
      情绪"）、批量文本情感标注。与 news_expert 的区别：news_expert
      获取原始新闻文本，sentiment_expert 对文本做可复现的量化评分。
      二者配合使用效果最佳。
"""

# NOTE: These rules are *invariant* across team compositions. They
# must NEVER mention a specific specialist by name, because the team
# is assembled at runtime and absent specialists would otherwise leak
# into the prompt as phantom routing targets — causing failing
# ``transfer_to_<missing>`` tool calls. Per-specialist guidance lives
# in the ``*_PROMPT_*`` sections above.
SUPERVISOR_PROMPT_RULES = """\
Your job
--------
1. READ the user's request carefully. Identify every distinct
   sub-question it contains (e.g. "基本资料 + 最近披露 + 算均值"
   is three sub-questions, not one). A user request that uses
   numbered steps (1) (2) (3) ... or bullet points is GIVING YOU
   the decomposition explicitly — every numbered step counts as
   ONE distinct sub-question and DEMANDS ITS OWN hand-off.
2. PLAN a minimal sequence of hand-offs. For each sub-question,
   pick the single specialist whose toolbelt (described above) is
   the best fit. If the user gave a company name but a subsequent
   step needs a 6-digit ticker, resolve the ticker FIRST via the
   specialist that owns the name-lookup tool.
3. HAND OFF ONE SUBTASK AT A TIME by calling the appropriate
   ``transfer_to_<name>`` tool. Wait for that specialist's result
   before routing the next subtask. Never issue two hand-offs in
   parallel — the shared state assumes serial turns.
4. WRITE the final answer yourself ONLY AFTER every sub-question
   has been delegated and answered. A useful self-check before
   you produce the final answer: re-read the user's original
   request and verify that EACH numbered / bullet sub-task was
   handled by an actual ``transfer_to_<name>`` hand-off (not by
   you). If any sub-task was skipped, route it now.
   Required structure for multi-step research requests:
     - ### 核心发现 / Key findings  (3-5 bullet points, with concrete
       numbers and short quotations from the PDF where relevant)
     - ### 数据来源 / Sources (list the specialists called and what
       each contributed)
5. Never invent numbers or quotes. If a specialist returned a dict
   with an ``"error"`` key, say so plainly and do NOT fabricate a
   substitute.
6. Do NOT call specialist tools yourself. You have no direct access
   to ``fin_*``, ``pdf_*``, ``code_*``, ``news_*``, ``knowledge_*``,
   or ``sentiment_*`` — only to the ``transfer_to_*`` hand-off tools.

CRITICAL anti-hallucination rules
---------------------------------
A. NEVER claim a tool, specialist, or capability is "unavailable",
   "tool-restricted", "无法访问", "由于工具限制", "暂不支持", or
   any equivalent. Every specialist enumerated in the team roster
   above IS available right now. If you catch yourself writing
   such a phrase, STOP — re-read the roster and issue the correct
   ``transfer_to_<name>`` hand-off instead.
B. NEVER substitute your own knowledge for a specialist's output.
   If the user asks for content from a PDF, the user's knowledge
   base, the latest stock price, or a numerical computation, you
   MUST route to the specialist that owns that capability — even
   if you "could answer it yourself". Routing IS the deliverable;
   the specialists' outputs are what the user is paying for.
C. NEVER perform arithmetic / statistics / data transformations
   yourself when a coder specialist is on the team. Even simple
   means and standard deviations go through the coder via a
   ``transfer_to_<coder>`` hand-off — that is how we guarantee
   reproducibility.
D. If a sub-task SHOULD have a hand-off but you find yourself
   reaching for self-generated text, that is the bug — fix it by
   issuing the missing ``transfer_to_<name>`` call BEFORE writing
   any prose for that sub-task.
"""


def _build_supervisor_prompt(
    *,
    has_data: bool,
    has_report: bool,
    has_coder: bool,
    has_knowledge: bool,
    has_news: bool,
    has_sentiment: bool,
) -> str:
    """Assemble the supervisor prompt to match the actual team roster.

    Listing a specialist the graph does NOT contain would produce
    ``transfer_to_<missing>`` tool calls that fail at runtime. The
    prompt therefore enumerates only the specialists we actually
    compiled.
    """
    parts = [SUPERVISOR_PROMPT_BASE]
    if has_data:
        parts.append(SUPERVISOR_PROMPT_DATA)
    if has_report:
        parts.append(SUPERVISOR_PROMPT_REPORT)
    if has_coder:
        parts.append(SUPERVISOR_PROMPT_CODER)
    if has_news:
        parts.append(SUPERVISOR_PROMPT_NEWS)
    if has_knowledge:
        parts.append(SUPERVISOR_PROMPT_KNOWLEDGE)
    if has_sentiment:
        parts.append(SUPERVISOR_PROMPT_SENTIMENT)
    parts.append("\n" + SUPERVISOR_PROMPT_RULES)
    return "".join(parts)


class _ResearchState(TypedDict, total=False):
    """Parent-graph state for the supervisor + reflection wrapper.

    Single field that matters: the message stream. The ``add_messages``
    reducer dedupes by message id, so when the inner supervisor
    returns its full transcript (input + new messages), only the
    newly produced messages actually get appended to parent state —
    which is the behaviour we want for clean SSE streaming.
    """

    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Human-in-the-Loop review node
# ---------------------------------------------------------------------------

def _build_human_review_node():
    """Create a graph node that pauses execution for human review.

    The node extracts the supervisor's draft from the message stream
    and calls ``interrupt()`` — LangGraph persists the graph state to
    the checkpointer and halts execution.  The SSE layer detects the
    pause and emits a ``review_requested`` event.

    When the reviewer calls ``/approve`` or ``/resume``, the graph is
    resumed with ``Command(resume=value)``.  The ``interrupt()`` call
    returns that value:

    * ``{"action": "approve", ...}`` — node passes through; draft
      proceeds to reflection / END unchanged.
    * ``{"action": "revise", "feedback": "..."}`` — node injects the
      feedback as a ``HumanMessage`` so downstream nodes (reflection
      or a potential supervisor re-run) can incorporate it.
    """

    async def human_review_node(state: _ResearchState) -> dict[str, list[BaseMessage]]:
        messages = state.get("messages", [])
        draft = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                content = msg.content
                if isinstance(content, str) and content.strip():
                    draft = content
                    break

        decision = interrupt({
            "draft": draft,
            "action_required": "approve_or_revise",
        })

        if isinstance(decision, dict) and decision.get("action") == "revise":
            feedback = decision.get("feedback", "")
            if feedback:
                return {
                    "messages": [
                        HumanMessage(
                            content=f"[REVIEWER FEEDBACK]\n{feedback}"
                        )
                    ]
                }

        return {"messages": []}

    return human_review_node


def _wrap_with_hitl_only(
    supervisor: CompiledStateGraph,
    *,
    checkpointer: BaseCheckpointSaver | None,
) -> CompiledStateGraph:
    """Wrap supervisor with a human-review interrupt (no reflection)."""

    async def supervisor_node(state: _ResearchState) -> dict[str, list[BaseMessage]]:
        result = await supervisor.ainvoke(
            {"messages": state.get("messages", [])},
        )
        return {"messages": result.get("messages", [])}

    parent: StateGraph = StateGraph(_ResearchState)
    parent.add_node("supervisor", supervisor_node)
    parent.add_node("human_review", _build_human_review_node())
    parent.add_edge(START, "supervisor")
    parent.add_edge("supervisor", "human_review")
    parent.add_edge("human_review", END)
    compiled = parent.compile(checkpointer=checkpointer)
    logger.info("Research supervisor wrapped with HITL review (no reflection).")
    return compiled


def _wrap_with_reflection(
    supervisor: CompiledStateGraph,
    *,
    model_router: ModelRouter,
    pass_threshold: float,
    max_iterations: int,
    checkpointer: BaseCheckpointSaver | None,
    enable_hitl: bool = False,
) -> CompiledStateGraph:
    """Wrap a compiled supervisor in a parent graph that runs reflection.

    Why a parent graph instead of inline post-processing?
    -----------------------------------------------------
    We could simply call ``supervisor.ainvoke`` and then run
    ``reflection.ainvoke`` on its output in pure Python. We don't,
    because:

      1. LangGraph's tracing / LangSmith integration loses the
         per-node visualisation if part of the pipeline runs
         outside the graph. Keeping reflection as a graph node
         keeps the full DAG visible in studio.
      2. The checkpointer is attached to the OUTER graph, so the
         supervisor + reflection are atomic with respect to
         resume-after-crash: a thread that crashed mid-reflection
         picks up at the critic node, not by re-running the whole
         specialist team.
      3. Adding more post-supervisor stages later (e.g. fact-
         checking against a citation index) is a 1-node graph edit
         rather than a Python orchestration rewrite.
    """
    reflection = build_reflection_subgraph(
        model_router=model_router,
        pass_threshold=pass_threshold,
        max_iterations=max_iterations,
    )

    async def supervisor_node(state: _ResearchState) -> dict[str, list[BaseMessage]]:
        """Run the inner supervisor graph and pipe its output upward."""
        result = await supervisor.ainvoke(
            {"messages": state.get("messages", [])},
        )
        return {"messages": result.get("messages", [])}

    async def reflection_node(state: _ResearchState) -> dict[str, list[BaseMessage]]:
        """Run the reflection subgraph over the supervisor's output."""
        result = await reflection.ainvoke(
            {"messages": state.get("messages", [])},
        )
        return {"messages": result.get("messages", [])}

    parent: StateGraph = StateGraph(_ResearchState)
    parent.add_node("supervisor", supervisor_node)
    parent.add_node("reflection", reflection_node)

    if enable_hitl:
        parent.add_node("human_review", _build_human_review_node())
        parent.add_edge(START, "supervisor")
        parent.add_edge("supervisor", "human_review")
        parent.add_edge("human_review", "reflection")
        parent.add_edge("reflection", END)
    else:
        parent.add_edge(START, "supervisor")
        parent.add_edge("supervisor", "reflection")
        parent.add_edge("reflection", END)

    compiled = parent.compile(checkpointer=checkpointer)
    logger.info(
        "Research supervisor wrapped with reflection: "
        "pass_threshold={:.2f} max_iterations={} hitl={}",
        pass_threshold,
        max_iterations,
        enable_hitl,
    )
    return compiled


def build_research_supervisor(
    *,
    model_router: ModelRouter,
    data_tools: Sequence[BaseTool] | None = None,
    report_tools: Sequence[BaseTool] | None = None,
    coder_tools: Sequence[BaseTool] | None = None,
    knowledge_tools: Sequence[BaseTool] | None = None,
    news_tools: Sequence[BaseTool] | None = None,
    sentiment_tools: Sequence[BaseTool] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    supervisor_tier: ModelTier = ModelTier.HEAVY,
    enable_reflection: bool = False,
    reflection_pass_threshold: float = 0.85,
    reflection_max_iterations: int = 2,
    enable_hitl: bool = False,
) -> CompiledStateGraph:
    """Compile the Phase-4.7 financial-research supervisor graph.

    The graph includes exactly the specialists whose tool lists were
    supplied (non-empty). A supervisor with zero specialists would be
    useless, so at least one tool list must be non-empty — we fail
    loudly otherwise.

    Args:
        model_router: Shared router (supervisor uses
            ``supervisor_tier``; specialists use MEDIUM via the
            ``ANALYST`` / ``RETRIEVER`` agent-name mapping inside
            their builders).
        data_tools: ``fin_*`` tools. Omit/empty → no ``data_expert``.
        report_tools: ``pdf_*`` tools. Omit/empty → no ``report_expert``.
        coder_tools: ``code_*`` tools. Omit/empty → no ``coder_expert``.
        knowledge_tools: ``knowledge_*`` tools. Omit/empty → no ``knowledge_expert``.
        news_tools: ``news_*`` tools. Omit/empty → no ``news_expert``.
        sentiment_tools: ``sentiment_*`` tools. Omit/empty → no ``sentiment_expert``.
        checkpointer: Optional LangGraph checkpointer.
        supervisor_tier: Defaults to HEAVY.
        enable_reflection: When True, wraps the supervisor in a parent
            graph that runs a critic+writer reflection loop over the
            supervisor's final synthesis. The loop terminates as soon
            as the critic scores the draft at or above
            ``reflection_pass_threshold`` OR after
            ``reflection_max_iterations`` rewrites — whichever comes
            first. The wrapper preserves the supervisor's
            ``ainvoke`` / ``astream`` contract; the only difference
            visible to callers is one extra ``AIMessage`` at the end
            of the transcript whose ``additional_kwargs['reflection']``
            carries the audit trail.
        reflection_pass_threshold: Passed to ``build_reflection_subgraph``.
        reflection_max_iterations: Passed to ``build_reflection_subgraph``.
        enable_hitl: When True, inserts a ``human_review`` node that
            calls ``interrupt()`` after the supervisor draft is
            produced. The graph pauses for human approval; callers
            resume via ``Command(resume=...)``.

    Returns:
        A compiled LangGraph app.

    Raises:
        ValueError: If every tool list was empty.
    """
    has_data = bool(data_tools)
    has_report = bool(report_tools)
    has_coder = bool(coder_tools)
    has_knowledge = bool(knowledge_tools)
    has_news = bool(news_tools)
    has_sentiment = bool(sentiment_tools)

    if not (has_data or has_report or has_coder or has_knowledge or has_news or has_sentiment):
        raise ValueError(
            "build_research_supervisor needs at least one specialist's "
            "tools. All six were empty."
        )

    agents: list = []
    roster: list[str] = []

    if has_data:
        agents.append(build_data_expert(model_router, data_tools or []))
        roster.append("data_expert")
    if has_report:
        agents.append(build_report_expert(model_router, report_tools or []))
        roster.append("report_expert")
    if has_coder:
        agents.append(build_coder_expert(model_router, coder_tools or []))
        roster.append("coder_expert")
    if has_news:
        agents.append(build_news_expert(model_router, news_tools or []))
        roster.append("news_expert")
    if has_knowledge:
        agents.append(build_knowledge_expert(model_router, knowledge_tools or []))
        roster.append("knowledge_expert")
    if has_sentiment:
        agents.append(build_sentiment_expert(model_router, sentiment_tools or []))
        roster.append("sentiment_expert")

    supervisor_model = model_router.get_model(supervisor_tier)
    prompt = _build_supervisor_prompt(
        has_data=has_data,
        has_report=has_report,
        has_coder=has_coder,
        has_knowledge=has_knowledge,
        has_news=has_news,
        has_sentiment=has_sentiment,
    )

    workflow = create_supervisor(
        agents=agents,
        model=supervisor_model,
        prompt=prompt,
        output_mode="last_message",
    )

    # When reflection is enabled, the parent (wrapper) graph holds the
    # checkpointer — the inner supervisor compiles statelessly so we
    # don't get two layers fighting over the same thread_id. When
    # reflection is OFF, the supervisor itself holds the checkpointer
    # exactly as before (zero behavioural change).
    if enable_reflection:
        inner = workflow.compile()
        compiled = _wrap_with_reflection(
            inner,
            model_router=model_router,
            pass_threshold=reflection_pass_threshold,
            max_iterations=reflection_max_iterations,
            checkpointer=checkpointer,
            enable_hitl=enable_hitl,
        )
    elif enable_hitl:
        inner = workflow.compile()
        compiled = _wrap_with_hitl_only(inner, checkpointer=checkpointer)
    else:
        compiled = workflow.compile(checkpointer=checkpointer)

    logger.info(
        "Research supervisor compiled: tier={} specialists={} reflection={} hitl={}",
        supervisor_tier.value,
        roster,
        enable_reflection,
        enable_hitl,
    )
    return compiled


__all__ = [
    "build_research_supervisor",
    "SUPERVISOR_PROMPT_BASE",
    "SUPERVISOR_PROMPT_DATA",
    "SUPERVISOR_PROMPT_REPORT",
    "SUPERVISOR_PROMPT_CODER",
    "SUPERVISOR_PROMPT_NEWS",
    "SUPERVISOR_PROMPT_KNOWLEDGE",
    "SUPERVISOR_PROMPT_SENTIMENT",
    "SUPERVISOR_PROMPT_RULES",
]
