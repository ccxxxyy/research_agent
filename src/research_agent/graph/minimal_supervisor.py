"""Minimal supervisor-pattern graph — the Phase-3 entry point.

Why a SEPARATE graph from ``supervisor.py``?
The ``supervisor.py`` module in this package wires a heavyweight
Corrective-RAG + Reflection pipeline and is the long-term home of the
full research agent. For Phase 3 we want to isolate and demonstrate the
**supervisor-of-specialists** pattern itself, without any retrieval,
grading, or reflection noise. This gives us a clean, fast, fully-
testable minimal example to iterate on before growing into the bigger
graph later.

Topology:

              ┌─────────────────────────┐
              │       supervisor        │   <-- LLM chooses who handles next
              └────┬───────┬───────┬────┘
                   │       │       │
                   ▼       ▼       ▼
              math_expert  time_expert  text_analyst
                   │       │       │
                   └───────┴───────┘
                           │
                           ▼
                        END (when supervisor emits final answer)

Handoffs are performed via auto-generated ``transfer_to_<name>`` tools
injected into the supervisor by ``langgraph_supervisor``. On each step
the supervisor either calls a transfer tool (routing) or produces a
final assistant message (terminating).
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
    build_math_expert,
    build_text_analyst,
    build_time_expert,
)
from research_agent.llm.provider import ModelRouter
from research_agent.llm.tier import ModelTier


SUPERVISOR_PROMPT_BASE = """\
You are the Supervisor of a small team of specialists:

  - math_expert   : evaluates arithmetic expressions using a calculator tool.
  - time_expert   : returns the current date/time for a timezone.
  - text_analyst  : counts the number of words in a given string.
"""

SUPERVISOR_PROMPT_CODER_EXTENSION = """\
  - coder_expert  : runs arbitrary sandboxed Python via an MCP subprocess.
                    Use this for tasks that need real code execution
                    (statistics, sorting, set algebra, regex, JSON
                    reshaping) — anything beyond a single expression
                    that ``math_expert`` can handle.
"""

SUPERVISOR_PROMPT_RULES = """\
Your job:
1. READ the user's request carefully.
2. If the request needs a specialist, HAND IT OFF by calling the
   appropriate ``transfer_to_<name>`` tool. Delegate ONE subtask at a
   time; wait for the result before routing the next subtask.
3. For COMPOSITE requests (e.g. "count the words in X then multiply by
   Y"), break them into subtasks and delegate to specialists in order.
4. Once all needed specialist results are collected, WRITE a final
   concise answer for the user. Do NOT delegate a subtask that you can
   trivially answer yourself (e.g. reformatting).
5. Never attempt arithmetic, time lookups, word counts, or code
   execution yourself — always delegate these.
"""


def _build_supervisor_prompt(*, include_coder: bool) -> str:
    """Assemble the supervisor system prompt, adapting to the team roster.

    We only advertise ``coder_expert`` in the roster when it actually
    exists in the compiled graph. Listing a specialist the supervisor
    cannot hand off to would cause ``transfer_to_coder_expert`` calls
    to fail at runtime.
    """
    parts = [SUPERVISOR_PROMPT_BASE]
    if include_coder:
        parts.append(SUPERVISOR_PROMPT_CODER_EXTENSION)
    parts.append("\n" + SUPERVISOR_PROMPT_RULES)
    return "".join(parts)


def build_minimal_supervisor(
    *,
    model_router: ModelRouter,
    checkpointer: BaseCheckpointSaver | None = None,
    supervisor_tier: ModelTier = ModelTier.MEDIUM,
    coder_tools: Sequence[BaseTool] | None = None,
) -> CompiledStateGraph:
    """Build and compile the Phase-3 supervisor graph.

    The graph always contains three Python-local specialists
    (``math_expert``, ``time_expert``, ``text_analyst``). When
    ``coder_tools`` is supplied, a fourth MCP-backed specialist
    (``coder_expert``) is added and the supervisor prompt is extended
    to advertise it. This split keeps ``build_minimal_supervisor``
    usable by pure-Python unit tests (which don't want to spawn an
    MCP subprocess) while letting the full smoke test / production
    FastAPI lifespan opt into the real MCP link.

    Args:
        model_router: Shared model router used for both supervisor and
            specialists. The supervisor is given a stronger tier (default
            :attr:`ModelTier.MEDIUM`) because routing decisions benefit
            from better reasoning; specialists run on LIGHT.
        checkpointer: Optional LangGraph checkpointer. When supplied,
            the entire supervisor conversation (including specialist
            hand-offs) is persisted per ``thread_id``, so the Phase-2
            multi-turn / resume guarantees extend to this graph too.
        supervisor_tier: Override the supervisor model tier. HEAVY is
            appropriate if routing errors become noticeable in practice.
        coder_tools: Optional pre-loaded MCP tool list (typically the
            return value of
            :func:`research_agent.mcp_servers.client_factory.load_code_server_tools`).
            When ``None`` or empty, the ``coder_expert`` specialist is
            NOT added to the team.

    Returns:
        A compiled LangGraph app invoked via ``ainvoke`` / ``astream``.
    """
    math = build_math_expert(model_router)
    time_ = build_time_expert(model_router)
    text = build_text_analyst(model_router)

    agents: list = [math, time_, text]
    specialist_names = ["math_expert", "time_expert", "text_analyst"]

    include_coder = bool(coder_tools)
    if include_coder:
        coder = build_coder_expert(model_router, coder_tools or [])
        agents.append(coder)
        specialist_names.append("coder_expert")

    supervisor_model = model_router.get_model(supervisor_tier)
    prompt = _build_supervisor_prompt(include_coder=include_coder)

    workflow = create_supervisor(
        agents=agents,
        model=supervisor_model,
        prompt=prompt,
        # ``last_message`` keeps the shared state compact: only the last
        # specialist reply is appended back to the supervisor context.
        # Switch to ``full_history`` when you need the supervisor to see
        # the specialist's chain-of-thought too.
        output_mode="last_message",
    )

    compiled = workflow.compile(checkpointer=checkpointer)
    logger.info(
        "Minimal supervisor compiled: tier={} specialists={}",
        supervisor_tier.value,
        specialist_names,
    )
    return compiled
