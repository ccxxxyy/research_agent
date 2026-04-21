"""Specialist single-tool agents for the minimal supervisor demo.

Design rationale
----------------
Classic "ReAct + all tools" agents work, but they mask an important
architectural story: **tool specialization**. A supervisor-coordinated
team of single-purpose agents is more interpretable, easier to rate-
limit per capability, and cleaner to scale (swap one specialist without
touching the others).

This module builds three such specialists, all using the Phase-1
native toolset:

    math_expert  — owns ``calculate``
    time_expert  — owns ``get_current_time``
    text_analyst — owns ``get_word_count``

Each is a ``create_react_agent`` compiled graph with:
  * its own ``name`` (used by ``langgraph_supervisor`` as the handoff tag)
  * a single tool in its toolbox
  * a prompt that describes ONLY that capability

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


SPECIALIST_BUILDERS = {
    "math_expert": build_math_expert,
    "time_expert": build_time_expert,
    "text_analyst": build_text_analyst,
    "coder_expert": build_coder_expert,
}
"""Registry for looking up specialists by name — used by tests and demos.

Note that ``coder_expert`` takes an extra ``mcp_tools`` argument and
therefore has a different signature from the others. Callers that
iterate this registry generically should branch on the key.
"""
