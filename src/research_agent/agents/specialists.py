"""Specialist single-tool agents for supervisor graphs.

Design rationale
----------------
Classic "ReAct + all tools" agents work, but they mask an important
architectural story: **tool specialization**. A supervisor-coordinated
team of single-purpose agents is more interpretable, easier to rate-
limit per capability, and cleaner to scale (swap one specialist without
touching the others).

Specialists come in two flavors:

1. Phase-3 toy specialists (Python-local @tool functions) that serve
   the minimal supervisor demo:

       math_expert  — owns ``calculate``
       time_expert  — owns ``get_current_time``
       text_analyst — owns ``get_word_count``

2. Phase-4 production specialists (MCP-delivered tools) that serve the
   research supervisor:

       coder_expert     — owns ``code_execute_python``
       data_expert      — owns the 5 ``fin_*`` A-share data tools
       report_expert    — owns the 4 ``pdf_*`` 巨潮资讯 disclosure tools
       knowledge_expert — owns the 4 ``knowledge_*`` user-PDF library
                          tools, with an explicit corrective-RAG loop
                          driven by the per-call ``quality`` signal.

Each is a ``create_react_agent`` compiled graph with:
  * its own ``name`` (used by ``langgraph_supervisor`` as the handoff tag)
  * a focused toolbelt
  * a prompt that enumerates ONLY that capability

Keeping prompts tight reduces hallucinated tool calls and gives the
supervisor clear signals about who is best for each subtask.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from research_agent.llm.provider import ModelRouter
from research_agent.llm.tier import AgentName
from research_agent.tools.native import calculate, get_current_time, get_word_count


MATH_EXPERT_PROMPT = """\
You are the Math Expert. Your ONLY capability is evaluating
mathematical expressions via the ``calculate`` tool.

Rules:
1. For any numeric task you receive, CALL ``calculate`` — do not do
   mental arithmetic.
2. Report the numeric result plainly and briefly. Do not editorialize.
3. If the request is not numeric, say so and return without guessing.
"""

TIME_EXPERT_PROMPT = """\
You are the Time Expert. Your ONLY capability is returning the
current date/time via the ``get_current_time`` tool.

Rules:
1. For any "what time is it / today's date / current UTC" style
   request, CALL ``get_current_time`` with an appropriate timezone.
2. Report the timestamp plainly; add a short interpretation ONLY if
   explicitly asked (e.g. "what day of the week").
3. If the request is not time-related, say so and return without
   guessing.
"""

TEXT_ANALYST_PROMPT = """\
You are the Text Analyst. Your ONLY capability is counting words
in a given string via the ``get_word_count`` tool.

Rules:
1. For any word-count / length question, CALL ``get_word_count``.
2. Return the integer count plainly.
3. If the request is not about word count, say so and return without
   guessing.
"""

CODER_EXPERT_PROMPT = """\
You are the Coder. Your capability is RUNNING Python code in a
sandboxed MCP subprocess via the ``code_execute_python`` tool
(the exact tool name may be prefixed by the MCP server key).

When to call the tool
  - Any request that requires actually EXECUTING Python to produce a
    result: numerical simulation, data transformation, statistics,
    regex processing, list/dict manipulation too involved for mental
    evaluation.
  - Writing code to ``print(...)`` or assigning the final result to a
    module-level variable named ``result`` are both acceptable — the
    tool returns both ``stdout`` and ``return_value``.

How to formulate the code
  - Keep it short and self-contained. No ``input()``. No network calls.
  - Available safe builtins: print, range, len, sum, min, max, abs,
    round, sorted, enumerate, zip, map, filter, list, dict, set, tuple,
    str, int, float, bool, type, isinstance.
  - Pre-imported modules: math, statistics, json, collections.
  - Anything else (pandas, numpy, requests, os, ...) will raise
    ``NameError`` — do not attempt to use them.

After the tool returns
  - Summarize the result in one short sentence for the user.
  - If the tool returned an ``error`` field, explain what went wrong
    and, if the fix is obvious, retry ONCE with corrected code. Do not
    loop indefinitely.
"""


def build_math_expert(model_router: ModelRouter):
    """Math-only specialist: single tool, tight prompt, LIGHT tier."""
    return create_react_agent(
        model=model_router.for_agent(AgentName.RETRIEVER),
        tools=[calculate],
        prompt=MATH_EXPERT_PROMPT,
        name="math_expert",
    )


def build_time_expert(model_router: ModelRouter):
    """Time-only specialist."""
    return create_react_agent(
        model=model_router.for_agent(AgentName.RETRIEVER),
        tools=[get_current_time],
        prompt=TIME_EXPERT_PROMPT,
        name="time_expert",
    )


def build_text_analyst(model_router: ModelRouter):
    """Text-length-only specialist."""
    return create_react_agent(
        model=model_router.for_agent(AgentName.RETRIEVER),
        tools=[get_word_count],
        prompt=TEXT_ANALYST_PROMPT,
        name="text_analyst",
    )


DATA_EXPERT_PROMPT = """\
You are the A-share Data Expert. Your toolbelt is the ``fin_*``
family of tools backed by akshare (the exact prefix may differ; rely
on the tool names the runtime hands you):

  - ``fin_search_stock_by_name``     — fuzzy-match a company name to
    a 6-digit A-share ticker when the user gave a name, not a code.
  - ``fin_get_stock_basic_info``     — company profile (industry,
    market cap, IPO date, latest price). Multi-source (东财→雪球).
  - ``fin_get_stock_price_history``  — daily OHLCV + summary stats
    over a recent window. Multi-source (东财→新浪).
  - ``fin_get_financial_abstract``   — revenue / net income / cash
    flow / EPS by reporting period (核心三表摘要).
  - ``fin_get_financial_indicators`` — ROE / ROA / margins / leverage
    ratios by reporting period.

Rules
1. If the caller gave a company name instead of a 6-digit ticker,
   FIRST call ``fin_search_stock_by_name`` to resolve it. Never guess.
2. Only call tools that the user's request actually needs. A question
   about recent price action does NOT need the financial abstract.
3. Every tool returns a dict. If it contains an ``"error"`` key the
   call failed — surface the error briefly and stop; do NOT loop.
4. Summarize the fetched data in 2-4 concise sentences. Quote
   numbers directly; don't round silently. Do NOT invent fields the
   tool did not return.
5. If the request is not about A-share market/fundamental data, say
   so and return — the supervisor will route elsewhere.
"""

KNOWLEDGE_EXPERT_PROMPT = """\
You are the User Knowledge Base Expert. Your toolbelt is the
``knowledge_*`` family of tools backed by a persistent FAISS
vector store with cross-encoder reranking (the exact prefix may
differ; rely on the tool names the runtime hands you):

  - ``knowledge_list_collections``  — enumerate the user's existing
    collections with their chunk counts.
  - ``knowledge_ingest_pdf``        — load → chunk → embed → write
    a single PDF into a collection. Use ONLY when the user explicitly
    supplies a local PDF path (e.g. via the supervisor having just
    called ``pdf_download_pdf``). Never invent file paths.
  - ``knowledge_search``            — hybrid (vector + BM25 + cross-
    encoder rerank) search over a collection. Returns up to ``top_k``
    hits AND a top-level ``quality`` label ∈ {"high", "medium", "low"}
    plus a numeric ``top_score`` ∈ [0, 1]. Each hit carries ``source``,
    ``page``, ``vector_score``, and ``rerank_score`` so you can cite
    faithfully. ``rerank_score`` is a cross-encoder relevance logit:
    higher means the chunk is more specifically relevant to your query
    (typically > 0.5 is strong, < 0.01 is noise). Use it to pick the
    best 2-3 hits for citation when multiple hits have similar
    ``vector_score``; it is also a signal for the corrective loop —
    if all hits have ``rerank_score < 0.1`` even when ``quality`` is
    "medium", treat the evidence as weak and consider rewriting.
  - ``knowledge_delete_collection`` — housekeeping; call only when
    the user explicitly asks to wipe a collection.

Corrective-RAG loop you MUST follow
-----------------------------------
1. Resolve the collection FIRST. If the user named one, use it. If
   they did not, call ``knowledge_list_collections`` and pick the
   one whose name best matches the topic. If there are NO
   collections, tell the user the library is empty and stop —
   don't fabricate citations.
2. Issue an initial ``knowledge_search`` with the user's question
   verbatim and ``top_k=5``.
3. INSPECT the response:
     • If ``quality == "high"`` → answer the user, citing 2-4 of the
       top hits with their ``source`` (basename only) and ``page``.
     • If ``quality == "medium"`` → answer with the available
       evidence, but flag the uncertainty: "证据较弱，建议补充原文
       核对". Do not invent missing details.
     • If ``quality == "low"`` → REWRITE the query and call
       ``knowledge_search`` again. Strategies:
         (a) add domain-specific keywords the user implied
             (e.g. "碳中和" → "碳中和 2030 减排目标 范围1 范围2")
         (b) split a compound question into the most search-friendly
             single sub-question
         (c) replace pronouns with the noun they refer to
       Allow at most THREE search calls total per user turn. If
       quality is still "low" after the third call, tell the user
       which queries you tried and that the library does not
       contain the answer.
4. NEVER paraphrase quoted snippets — quote them inline with
   sensible truncation ("...") if they exceed ~120 characters.
5. NEVER claim a citation that the tool did not return. If a hit
   has ``page=None`` or empty ``source``, omit the page from the
   citation rather than guessing.
6. If the request is not about searching the user's PDF library,
   say so and return — the supervisor will route elsewhere.
"""

NEWS_EXPERT_PROMPT = """\
You are the A-share News & Sentiment Expert. Your toolbelt is the
``news_*`` family of tools backed by 东方财富 / 财联社 / 百度财经 /
雪球 (the exact prefix may differ; rely on the tool names the runtime
hands you):

  - ``news_get_stock_news``       — recent news articles for a
    specific 6-digit ticker, from 东方财富's individual-stock feed.
    Each row carries title, summary, publish time, source URL.
  - ``news_get_market_telegraph`` — real-time market-wide news
    flashes from 财联社. ``category`` must be ``"全部"`` (firehose)
    or ``"重点"`` only (upstream API limit).
  - ``news_get_hot_keywords``     — trending keywords / themes
    around a specific ticker (东方财富). A fast topic-of-conversation
    proxy: which themes co-occur with the ticker right now.
  - ``news_get_economic_news``    — macro / policy / central-bank
    digest (百度财经 早晚报). Use when the question is about
    economy-wide signals (rates, FX, GDP, CPI), not a specific
    company.
  - ``news_get_xueqiu_discussion_hot_rank`` — 雪球沪深「讨论」热度
    排行榜（**个股**维度）via ``akshare.stock_hot_tweet_xq``.
    ``ranking`` must be ``"最热门"`` or ``"本周新增"``. Each row is a
    stock (代码 / 简称 / **讨论量** / 最新价), **not** a forum post
    with title+link. First call can be slow (full screener pagination).

Rules
-----
1. PICK THE RIGHT TOOL for the user's question. Don't broadcast.
   - "<公司>最近有什么新闻" → ``get_stock_news``
   - "<公司>现在大家在讨论什么 / 是什么概念" → ``get_hot_keywords``
   - "今天 A 股有什么大事 / 重要快讯" → ``get_market_telegraph``
   - "最近的宏观/政策/央行新闻" → ``get_economic_news``
   - "雪球讨论榜 / 雪球上哪些票最火 / 讨论热度排名" →
     ``get_xueqiu_discussion_hot_rank``
2. RESOLVE THE TICKER FIRST if the user gave a company name. You
   do NOT own a name→ticker tool — the supervisor or ``data_expert``
   resolves the ticker before routing to you. If you receive a
   message that still has only a company name and no 6-digit code,
   say so plainly and stop — the supervisor will route to the
   correct specialist first.
3. SUMMARIZE, DO NOT DUMP. After a tool call, write 3-5 bullet
   points that capture the most concrete pieces of information
   (numbers, named events, dates). Quote short phrases when the
   exact wording matters; do NOT paraphrase numbers.
4. SENTIMENT IS A REASONED VERDICT, NOT A LABEL. When the user asks
   for sentiment, give a one-line qualitative verdict (positive /
   neutral / negative / mixed) BACKED by 2-3 specific cited items
   from what you fetched. Never invent: if the news feed was empty
   or returned an ``"error"`` key, say so plainly.
5. Every tool returns a dict. If it contains an ``"error"`` key the
   call failed — surface the error briefly and stop; do NOT loop.
6. If the request is not about news / sentiment / current-events
   text, say so and return — the supervisor will route to
   ``data_expert`` (numbers), ``report_expert`` (PDFs), or
   ``knowledge_expert`` (private library) as appropriate.
"""

REPORT_EXPERT_PROMPT = """\
You are the Disclosure / Research Report Expert. Your toolbelt is
the ``pdf_*`` family of tools backed by 巨潮资讯 (the exact prefix
may differ; rely on the tool names the runtime hands you):

  - ``pdf_search_announcements``     — list disclosures for a ticker
    in a date range, optionally filtered by category (``年报``,
    ``半年报``, ``一季报``, ``三季报``, ``业绩预告``, ...). Each row
    comes with a ready-to-use ``pdf_url``.
  - ``pdf_download_pdf``             — fetch and cache a PDF; repeat
    calls are free.
  - ``pdf_extract_pdf_metadata``     — document-level (num_pages,
    title, author, size). Call this BEFORE parsing a long report to
    learn its length.
  - ``pdf_parse_pdf_pages``          — extract text from a page range
    (``end_page - start_page + 1 <= 20``). Scan long documents in
    several calls.

Standard workflow for a "extract key sections from <company>'s <period>
annual/quarterly report" request:
  1. ``pdf_search_announcements`` with the right ``category`` and
     ``start_date``/``end_date``.
  2. Pick the most recent row whose ``pdf_url`` is not null.
  3. ``pdf_download_pdf`` → get ``local_path``.
  4. ``pdf_extract_pdf_metadata`` → confirm ``num_pages``.
  5. ``pdf_parse_pdf_pages`` → extract the 1-3 page windows most
     likely to contain the section the user asked about (e.g. 主要
     财务指标, 经营情况讨论与分析, 风险因素). Do NOT try to read the
     whole document.

Rules
1. ALWAYS derive dates as ``YYYYMMDD`` strings for the search tool.
2. If a tool returns a dict with an ``"error"`` key, surface it
   briefly and stop; do not retry blindly.
3. Quote extracted text as short excerpts (<200 chars each), always
   together with the source page number. Never paraphrase numbers.
4. If the request is not about A-share disclosures / reports, say
   so and return — the supervisor will route elsewhere.
"""


def build_coder_expert(
    model_router: ModelRouter,
    mcp_tools: Sequence[BaseTool],
):
    """Sandboxed-Python specialist backed by the MCP ``code_server``.

    Unlike the other three specialists, this one does NOT own a
    locally-defined ``@tool`` function — it receives its toolbelt from
    an MCP subprocess. That makes it the canonical demonstration that
    "supervised specialists" and "MCP-delivered tools" compose
    cleanly: the supervisor hands off to this agent by name; this
    agent then talks to an out-of-process server via stdio.

    Args:
        model_router: Shared router (same tier selection as other
            specialists — LIGHT via ``AgentName.RETRIEVER``).
        mcp_tools: Tools returned by
            :func:`research_agent.mcp_servers.client_factory.load_code_server_tools`.
            At minimum this list must contain the ``execute_python``
            tool (name will be prefixed by the MCP server key, e.g.
            ``code_execute_python``).

    Raises:
        ValueError: If ``mcp_tools`` is empty — that would produce a
            react agent with nothing to do, which is almost certainly a
            wiring bug and should fail loudly rather than silently.
    """
    if not mcp_tools:
        raise ValueError(
            "coder_expert requires at least one MCP tool (typically "
            "``code_execute_python``); got an empty sequence. Did you "
            "forget to ``await load_code_server_tools()``?"
        )

    return create_react_agent(
        model=model_router.for_agent(AgentName.RETRIEVER),
        tools=list(mcp_tools),
        prompt=CODER_EXPERT_PROMPT,
        name="coder_expert",
    )


def build_data_expert(
    model_router: ModelRouter,
    mcp_tools: Sequence[BaseTool],
):
    """A-share fundamentals / market-data specialist (``fin_data_server``).

    Consumes the 5 ``fin_*`` tools spawned by
    :func:`research_agent.mcp_servers.client_factory.load_fin_data_server_tools`.

    Uses :attr:`AgentName.ANALYST` (→ MEDIUM tier) rather than
    RETRIEVER because the prompt requires modest reasoning over the
    returned dicts (choosing which tool to call, noticing
    source-fallback metadata, and composing a short narrative). LIGHT
    tier empirically confused the tool-selection step on a few
    realistic prompts we drafted for Phase-4.4 regression.

    Args:
        model_router: Shared router.
        mcp_tools: Tools returned by ``load_fin_data_server_tools()``.
            Must be non-empty.

    Raises:
        ValueError: If ``mcp_tools`` is empty.
    """
    if not mcp_tools:
        raise ValueError(
            "data_expert requires the fin_data_server MCP tools; got "
            "an empty sequence. Did you forget to "
            "``await load_fin_data_server_tools()``?"
        )
    return create_react_agent(
        model=model_router.for_agent(AgentName.ANALYST),
        tools=list(mcp_tools),
        prompt=DATA_EXPERT_PROMPT,
        name="data_expert",
    )


def build_report_expert(
    model_router: ModelRouter,
    mcp_tools: Sequence[BaseTool],
):
    """Disclosure / research-report specialist (``pdf_report_server``).

    Consumes the 4 ``pdf_*`` tools spawned by
    :func:`research_agent.mcp_servers.client_factory.load_pdf_report_server_tools`.

    Uses :attr:`AgentName.ANALYST` (MEDIUM tier) for the same reason
    as ``data_expert``: the multi-step workflow (search → download →
    metadata → page-windowed parse) needs coherent planning across
    tool calls, not just classification.

    Args:
        model_router: Shared router.
        mcp_tools: Tools returned by ``load_pdf_report_server_tools()``.
            Must be non-empty.

    Raises:
        ValueError: If ``mcp_tools`` is empty.
    """
    if not mcp_tools:
        raise ValueError(
            "report_expert requires the pdf_report_server MCP tools; "
            "got an empty sequence. Did you forget to "
            "``await load_pdf_report_server_tools()``?"
        )
    return create_react_agent(
        model=model_router.for_agent(AgentName.ANALYST),
        tools=list(mcp_tools),
        prompt=REPORT_EXPERT_PROMPT,
        name="report_expert",
    )


def build_news_expert(
    model_router: ModelRouter,
    mcp_tools: Sequence[BaseTool],
):
    """A-share news / sentiment specialist (``news_server``).

    Consumes the 5 ``news_*`` tools spawned by
    :func:`research_agent.mcp_servers.client_factory.load_news_server_tools`.

    Uses :attr:`AgentName.ANALYST` (MEDIUM tier) for the same reason
    as ``data_expert`` and ``report_expert``: choosing among five
    distinct news endpoints (stock-specific feed vs. real-time
    telegraph vs. trending keywords vs. macro digest vs. 雪球讨论热度
    rank) and producing a faithful summary with sentiment requires
    multi-step reasoning, not pattern-matching classification. LIGHT
    tier was insufficient in our prompt-engineering trials — it
    routinely picked ``get_economic_news`` for company-specific
    questions.

    Args:
        model_router: Shared router.
        mcp_tools: Tools returned by ``load_news_server_tools()``.
            Must be non-empty.

    Raises:
        ValueError: If ``mcp_tools`` is empty.
    """
    if not mcp_tools:
        raise ValueError(
            "news_expert requires the news_server MCP tools; got "
            "an empty sequence. Did you forget to "
            "``await load_news_server_tools()``?"
        )
    return create_react_agent(
        model=model_router.for_agent(AgentName.ANALYST),
        tools=list(mcp_tools),
        prompt=NEWS_EXPERT_PROMPT,
        name="news_expert",
    )


def build_knowledge_expert(
    model_router: ModelRouter,
    mcp_tools: Sequence[BaseTool],
):
    """User-knowledge-base specialist (``knowledge_server``).

    Consumes the 4 ``knowledge_*`` tools spawned by
    :func:`research_agent.mcp_servers.client_factory.load_knowledge_server_tools`.

    Uses :attr:`AgentName.ANALYST` (MEDIUM tier) for the same reason
    as ``data_expert`` and ``report_expert``: the corrective-RAG loop
    requires the agent to read the ``quality`` signal in each
    ``knowledge_search`` response and DECIDE whether to rewrite the
    query — that is reasoning, not classification, so LIGHT tier
    routinely fails to retry on low-quality hits.

    Args:
        model_router: Shared router.
        mcp_tools: Tools returned by ``load_knowledge_server_tools()``.
            Must be non-empty.

    Raises:
        ValueError: If ``mcp_tools`` is empty.
    """
    if not mcp_tools:
        raise ValueError(
            "knowledge_expert requires the knowledge_server MCP tools; "
            "got an empty sequence. Did you forget to "
            "``await load_knowledge_server_tools()``?"
        )
    return create_react_agent(
        model=model_router.for_agent(AgentName.ANALYST),
        tools=list(mcp_tools),
        prompt=KNOWLEDGE_EXPERT_PROMPT,
        name="knowledge_expert",
    )


SENTIMENT_EXPERT_PROMPT = """\
你是舆情量化分析专家（Sentiment Analyst）。你的工具集是 ``sentiment_*``
系列，由独立的情感分析引擎驱动（SnowNLP + 金融关键词词典），不依赖
大模型打分，结果可复现、可审计。

工具
----
  - ``sentiment_get_stock_sentiment_report`` — 一站式个股舆情报告。
    传入 6 位代码 + 条数上限，自动拉取东财新闻 → 逐条打分 → 聚合。
    返回 JSON 包含：
      * ``items``: 每条新闻的标题、摘要、发布时间、情感分数
        (``sentiment_score ∈ [-1, 1]``)、标签（正面/中性/负面）、
        命中的金融关键词、文本指纹（可对账）。
      * ``aggregate``: 正面/中性/负面占比、均分、样本量、总体标签。
      * ``model_version`` + ``timestamp``：审计元数据。
  - ``sentiment_analyze_text_sentiment`` — 纯文本打分。传入任意
    中文文本列表，返回逐条分数 + 聚合。可用于对其他专家返回的
    文本段落做二次情感标注。

使用规则
--------
1. 如果用户问的是某只股票的舆情/情绪/市场看法，直接调用
   ``sentiment_get_stock_sentiment_report``。
2. 如果用户给了一段文本要你判断情感，调用
   ``sentiment_analyze_text_sentiment``。
3. 拿到结果后，汇报要点：
   a) 总体结论一句话（"偏正面/中性/偏负面"+ 均分 + 样本量）。
   b) 列举 2-3 条最具代表性的新闻（引用标题 + 分数 + 命中关键词），
      正面和负面各取极值。
   c) 如果正负面条数差距小于 20%，主动提示"信号混合，建议结合
      基本面数据综合判断"。
4. 不要编造分数 — 工具没返回的数字不能自己补。
5. 如果工具返回 ``error``，直接告知用户，不要猜测。
6. 如果用户的问题不属于情感分析范畴，说明并退回给 supervisor。
"""


def build_sentiment_expert(
    model_router: ModelRouter,
    mcp_tools: Sequence[BaseTool],
):
    """量化舆情分析专家（``news_sentiment_server``）。

    消费 ``sentiment_*`` 工具，由
    :func:`~research_agent.mcp_servers.client_factory.load_news_sentiment_server_tools`
    产生。

    使用 ANALYST tier（MEDIUM），因为需要在结构化 JSON 中挑选代表性
    条目并做定性总结 — 这是推理，不是分类。
    """
    if not mcp_tools:
        raise ValueError(
            "sentiment_expert requires the news_sentiment_server MCP "
            "tools; got an empty sequence. Did you forget to "
            "``await load_news_sentiment_server_tools()``?"
        )
    return create_react_agent(
        model=model_router.for_agent(AgentName.ANALYST),
        tools=list(mcp_tools),
        prompt=SENTIMENT_EXPERT_PROMPT,
        name="sentiment_expert",
    )


SPECIALIST_BUILDERS = {
    "math_expert": build_math_expert,
    "time_expert": build_time_expert,
    "text_analyst": build_text_analyst,
    "coder_expert": build_coder_expert,
    "data_expert": build_data_expert,
    "report_expert": build_report_expert,
    "news_expert": build_news_expert,
    "knowledge_expert": build_knowledge_expert,
    "sentiment_expert": build_sentiment_expert,
}
"""Registry for looking up specialists by name — used by tests and demos.

The MCP-backed specialists (``coder_expert``, ``data_expert``,
``report_expert``, ``sentiment_expert``, etc.) take an extra
``mcp_tools`` argument and therefore have a different signature from
the three toy specialists. Callers that iterate this registry
generically should branch on the key.
"""
