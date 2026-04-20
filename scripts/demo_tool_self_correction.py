"""Demonstrate an agent recovering from tool errors without human help.

Run:
    uv run python scripts/demo_tool_self_correction.py

The ReAct loop has a property that is easy to state but surprisingly
useful: **a tool's error message is fed back to the LLM as a ToolMessage,
and the LLM can read it and retry with different inputs**.

We showcase this in two deterministic probes. Each probe uses a custom
tool that is guaranteed to fail in a specific, instructive way the
first time, then accept a corrected call. This avoids depending on
stochastic LLM reasoning to manufacture the failure.

Probe 1 — Transient service failure (retry with SAME arguments)
    We inject ``flaky_calculator`` that returns an error on its first
    call and succeeds on the second. The LLM must read the error and
    call the tool again with the same expression.

Probe 2 — Strict-validation failure (retry with FIXED arguments)
    We inject ``length_in_cm`` that rejects any unit outside the
    enumerated set {m, mm, cm, inch}. The user's question deliberately
    uses the noisy word "meters" in the prose; the LLM must read the
    tool's error message and re-invoke with unit="m".

Every invocation is bounded by a tight ``recursion_limit`` so the
process cannot hang in a pathological retry loop.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from loguru import logger

from research_agent.agents.simple import build_simple_agent
from research_agent.config import get_settings
from research_agent.llm.provider import ModelRouter


# ---------------------------------------------------------------------------
# Observable tools for the demo
# ---------------------------------------------------------------------------

FLAKY_CALLS = 0
STRICT_CALLS: list[dict[str, Any]] = []


@tool
def flaky_calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression. This calculator is flaky.

    The service is known to be unreliable: transient errors are possible.
    If the returned content starts with "Error:", RETRY the call once
    with the same arguments.

    Args:
        expression: Arithmetic expression, e.g. "12 * 11".
    """
    global FLAKY_CALLS
    FLAKY_CALLS += 1
    logger.debug("flaky_calculator call #{} expression={!r}", FLAKY_CALLS, expression)
    if FLAKY_CALLS == 1:
        return "Error: ServiceTemporarilyUnavailable. Please retry the same call once."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@tool
def length_in_cm(value: float, unit: str) -> str:
    """Convert a length to centimetres.

    The ``unit`` argument expects an internal unit code string, not a
    human-readable name. The exact set of accepted codes is not listed
    here; if the code you pass is not recognised, the tool's error
    message will report which codes ARE valid.

    Args:
        value: Numeric length (non-negative).
        unit: Internal unit code string. Consult the tool's error
            message if unsure of the valid values.
    """
    STRICT_CALLS.append({"value": value, "unit": unit})
    logger.debug("length_in_cm call #{} value={} unit={!r}", len(STRICT_CALLS), value, unit)
    # Deliberately obscure codes (UNIT_* namespace). A model that has not
    # seen them before will almost certainly guess "m", "meter", or "meters"
    # on its first attempt, triggering the corrective-feedback path.
    factors = {
        "UNIT_METRE": 100.0,
        "UNIT_MM": 0.1,
        "UNIT_CM": 1.0,
        "UNIT_INCH": 2.54,
    }
    if unit not in factors:
        return (
            f"Error: unit code '{unit}' is not recognised. "
            f"Valid codes are exactly: {sorted(factors)}. "
            f"Retry with one of these codes."
        )
    return f"{value * factors[unit]:.4f} cm"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _trace(messages: list[Any]) -> dict[str, Any]:
    tool_calls: list[str] = []
    tool_results: list[str] = []
    final = ""

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(f"{tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            preview = str(msg.content).splitlines()[0][:110]
            tool_results.append(f"{msg.name} -> {preview}")
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            final = str(msg.content)

    return {
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "final": final,
        "n_messages": len(messages),
    }


async def run_probe(agent, label: str, question: str, success_predicate) -> None:
    _banner(label)
    print(f"  user question : {question}")

    # Tight recursion bound so a runaway retry loop fails fast rather than
    # hanging. A ReAct step is ~1 LLM call; 10 steps comfortably covers
    # "one retry" scenarios.
    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": 10},
        )
    except Exception as e:
        print(f"  [FAIL] agent raised {type(e).__name__}: {e}")
        return

    trace = _trace(result["messages"])

    print("\n  tool calls emitted:")
    for tc in trace["tool_calls"]:
        print(f"    - {tc}")
    print("\n  tool results observed:")
    for tr in trace["tool_results"]:
        print(f"    - {tr}")
    print(f"\n  total tool calls  : {len(trace['tool_calls'])}")
    print(f"  final answer      : {trace['final'][:160]}")

    ok = success_predicate(trace)
    verdict = "PASS" if ok else "WARN"
    print(f"  [{verdict}] self-correction predicate")


def probe1_predicate(trace: dict[str, Any]) -> bool:
    """Success: flaky_calculator called >=2 times AND final answer contains 132."""
    flaky_count = sum(1 for tc in trace["tool_calls"] if tc.startswith("flaky_calculator"))
    had_error = any("Error:" in r for r in trace["tool_results"])
    return flaky_count >= 2 and had_error and "132" in trace["final"]


def probe2_predicate(trace: dict[str, Any]) -> bool:
    """Success: length_in_cm called >=2 times with DIFFERENT unit codes; result contains 100."""
    units_tried = {c["unit"] for c in STRICT_CALLS}
    had_error = any("Error:" in r for r in trace["tool_results"])
    return (
        len(STRICT_CALLS) >= 2
        and len(units_tried) >= 2
        and had_error
        and "100" in trace["final"]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SELF_CORRECTION_PROMPT = """\
You are a careful assistant with access to several tools.

Self-correction protocol — follow it strictly:
- If any tool returns content starting with "Error:", READ the error
  message, decide whether to retry with the same arguments (for transient
  errors) or adjust the arguments (for validation errors).
- Retry AT MOST ONCE per tool. After one retry, if the error persists,
  answer the user truthfully about the failure.
- Always produce a concise final user-visible answer when you stop.
"""


async def main() -> None:
    global FLAKY_CALLS
    settings = get_settings()
    logger.info("Light model: {}", settings.llm.light_model)

    router = ModelRouter(settings.llm)
    agent = build_simple_agent(
        router,
        tools=[flaky_calculator, length_in_cm],
        prompt=SELF_CORRECTION_PROMPT,
    )

    # ------ Probe 1 — transient error, retry with SAME args ------
    FLAKY_CALLS = 0
    await run_probe(
        agent,
        "Probe 1 — Transient error, retry with SAME arguments",
        "Use flaky_calculator to compute 12 * 11. Report the final number.",
        probe1_predicate,
    )

    # ------ Probe 2 — validation error, retry with FIXED args ------
    STRICT_CALLS.clear()
    await run_probe(
        agent,
        "Probe 2 — Validation error, retry with CORRECTED arguments",
        "Convert 1 metre into centimetres using length_in_cm. "
        "You don't know the tool's expected unit code up front — "
        "make an initial reasonable guess, read the tool's error if any, "
        "and correct your unit argument accordingly.",
        probe2_predicate,
    )

    _banner("Takeaway")
    print(
        """
  The ReAct loop is remarkably fault-tolerant WITHOUT special error-
  handling code on the application side. The ingredients are:

    1. Tools return structured, descriptive errors (never raise).
    2. Errors become ToolMessages in the graph state.
    3. The LLM sees those ToolMessages on its next turn and decides
       whether to retry, adjust arguments, or give up.
    4. A recursion_limit caps runaway loops so a dishonest tool or a
       confused model cannot burn tokens indefinitely.

  In production, pair this with a supervisor node that escalates to a
  stronger model after N consecutive failures, and with structured
  telemetry so failing tools show up in dashboards before they degrade
  user experience.
        """.rstrip()
    )


if __name__ == "__main__":
    asyncio.run(main())
