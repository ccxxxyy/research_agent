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

       coder_expert  — owns ``code_execute_python``
       data_expert   — owns the 5 ``fin_*`` A-share data tools
       report_expert — owns the 4 ``pdf_*`` 巨潮资讯 disclosure tools

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


SPECIALIST_BUILDERS = {
    "math_expert": build_math_expert,
    "time_expert": build_time_expert,
    "text_analyst": build_text_analyst,
    "coder_expert": build_coder_expert,
    "data_expert": build_data_expert,
    "report_expert": build_report_expert,
}
"""Registry for looking up specialists by name — used by tests and demos.

The three MCP-backed specialists (``coder_expert``, ``data_expert``,
``report_expert``) take an extra ``mcp_tools`` argument and therefore
have a different signature from the three toy specialists. Callers
that iterate this registry generically should branch on the key.
"""
