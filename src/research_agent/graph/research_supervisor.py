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

              ┌──────────────────────────────────────────┐
              │           research_supervisor            │  ← HEAVY tier LLM
              └─┬────────┬───────────┬─────────┬───────┬─┘
                │        │           │         │       │
                ▼        ▼           ▼         ▼       ▼
          data_expert report_expert coder_expert news_expert knowledge_expert
          (fin_data)  (pdf_report)  (code_server) (news_server) (knowledge_server)

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

from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph_supervisor import create_supervisor
from loguru import logger

from research_agent.agents.specialists import (
    build_coder_expert,
    build_data_expert,
    build_knowledge_expert,
    build_news_expert,
    build_report_expert,
)
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
   to ``fin_*``, ``pdf_*``, ``code_*``, ``news_*``, or ``knowledge_*`` —
   only to the ``transfer_to_*`` hand-off tools.

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
    parts.append("\n" + SUPERVISOR_PROMPT_RULES)
    return "".join(parts)


def build_research_supervisor(
    *,
    model_router: ModelRouter,
    data_tools: Sequence[BaseTool] | None = None,
    report_tools: Sequence[BaseTool] | None = None,
    coder_tools: Sequence[BaseTool] | None = None,
    knowledge_tools: Sequence[BaseTool] | None = None,
    news_tools: Sequence[BaseTool] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    supervisor_tier: ModelTier = ModelTier.HEAVY,
) -> CompiledStateGraph:
    """Compile the Phase-4.4 / 4.6 financial-research supervisor graph.

    The graph includes exactly the specialists whose tool lists were
    supplied (non-empty). A supervisor with zero specialists would be
    useless, so at least one of ``data_tools`` / ``report_tools`` /
    ``coder_tools`` / ``knowledge_tools`` / ``news_tools`` must be
    non-empty — we fail loudly otherwise.

    Args:
        model_router: Shared router (supervisor uses
            ``supervisor_tier``; specialists use MEDIUM via the
            ``ANALYST`` / ``RETRIEVER`` agent-name mapping inside
            their builders).
        data_tools: Output of
            :func:`research_agent.mcp_servers.client_factory.load_fin_data_server_tools`.
            When omitted or empty, ``data_expert`` is not added.
        report_tools: Output of
            :func:`research_agent.mcp_servers.client_factory.load_pdf_report_server_tools`.
            When omitted or empty, ``report_expert`` is not added.
        coder_tools: Output of
            :func:`research_agent.mcp_servers.client_factory.load_code_server_tools`.
            When omitted or empty, ``coder_expert`` is not added.
        knowledge_tools: Output of
            :func:`research_agent.mcp_servers.client_factory.load_knowledge_server_tools`.
            When omitted or empty, ``knowledge_expert`` is not added.
        news_tools: Output of
            :func:`research_agent.mcp_servers.client_factory.load_news_server_tools`.
            When omitted or empty, ``news_expert`` is not added.
        checkpointer: Optional LangGraph checkpointer for per-``thread_id``
            conversation persistence.
        supervisor_tier: Override supervisor model tier. Defaults to
            HEAVY; LIGHT is insufficient for the planning prompt.

    Returns:
        A compiled LangGraph app. Invoke with
        ``await app.ainvoke({"messages": [HumanMessage(...)]})``.

    Raises:
        ValueError: If every tool list was empty.
    """
    has_data = bool(data_tools)
    has_report = bool(report_tools)
    has_coder = bool(coder_tools)
    has_knowledge = bool(knowledge_tools)
    has_news = bool(news_tools)

    if not (has_data or has_report or has_coder or has_knowledge or has_news):
        raise ValueError(
            "build_research_supervisor needs at least one specialist's "
            "tools. Supply data_tools and/or report_tools and/or "
            "coder_tools and/or knowledge_tools and/or news_tools — "
            "all five were empty."
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

    supervisor_model = model_router.get_model(supervisor_tier)
    prompt = _build_supervisor_prompt(
        has_data=has_data,
        has_report=has_report,
        has_coder=has_coder,
        has_knowledge=has_knowledge,
        has_news=has_news,
    )

    workflow = create_supervisor(
        agents=agents,
        model=supervisor_model,
        prompt=prompt,
        output_mode="last_message",
    )
    compiled = workflow.compile(checkpointer=checkpointer)
    logger.info(
        "Research supervisor compiled: tier={} specialists={}",
        supervisor_tier.value,
        roster,
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
    "SUPERVISOR_PROMPT_RULES",
]
