"""Reflection subgraph — Writer / Reasoner self-critique loop.

Problem this module solves
--------------------------
The Phase-4.7 ``research_supervisor`` produces a final synthesis at
the end of every hand-off chain, but that synthesis is generated in a
single LLM pass — the same pass that just consumed half-a-dozen
specialist outputs. Two failure modes show up at evaluation time:

1. **Under-cited claims**: the supervisor paraphrases a specialist
   finding without preserving the source / page citation, blurring
   the boundary between "evidence" and "interpretation".
2. **Skipped sub-questions**: a multi-step user request "(1)…(2)…(3)…"
   gets a confident summary that quietly drops one of the steps,
   even though the supervisor's anti-hallucination prompt warned
   against it.

A second-pass **reflection loop** catches both failure modes cheaply:
a small LIGHT-tier critic scores the draft, and only when it falls
below the threshold do we burn a HEAVY-tier rewrite token. The
canonical "Self-RAG / Reflexion" pattern, applied at the synthesis
boundary rather than every retrieval step.

Topology
--------
This is a critic-first subgraph — we score before rewriting, so a
high-quality first draft costs ONE LIGHT-tier call instead of one
LIGHT + one HEAVY::

    ┌──────────────────────────────────────────────────────────┐
    │                       START                               │
    │                         │                                 │
    │                         ▼                                 │
    │                    critic_node       ◄─────┐              │
    │                         │                  │              │
    │                         ▼                  │              │
    │              ┌────────route?──────┐        │              │
    │              │                    │        │              │
    │     pass / max_iter           else (fail) │              │
    │              │                    │        │              │
    │              ▼                    ▼        │              │
    │         finalize_node         writer_node──┘              │
    │              │                                            │
    │              ▼                                            │
    │             END                                           │
    └──────────────────────────────────────────────────────────┘

State semantics
---------------
``iteration`` counts critic invocations:

* ``iteration = 0`` → the supervisor's ORIGINAL draft being critiqued.
* ``iteration = 1`` → writer's 1st rewrite being critiqued.
* ``iteration = N`` → writer's Nth rewrite being critiqued.

The loop is bounded by ``max_iterations`` (default 2 rewrites, so at
most 3 critiques). The bound is invariant: if every iteration fails
the quality bar, we still terminate, returning the highest-scoring
draft we saw — that's better than returning nothing or busy-looping.

Why a SUBGRAPH instead of two extra nodes in ``research_supervisor``?
--------------------------------------------------------------------
Three concrete benefits:

1. **Independent testability** — the subgraph runs against any
   ``messages`` list; we don't need to spin up the supervisor +
   six MCP subprocesses to test reflection logic in isolation.
2. **Composable on/off switch** — ``build_research_supervisor``
   takes ``enable_reflection: bool``; when False the parent graph
   is identical to the legacy supervisor and there is zero
   reflection overhead.
3. **Visible in tracing** — LangSmith / LangGraph studio render
   the subgraph as its own collapsed node, so the per-iteration
   write→critic edges are obvious in the visualisation rather than
   buried in a flat supervisor node.

Why no tools, no react agent?
-----------------------------
The writer and critic are pure transformations (text in → text out).
Wrapping them in ``create_react_agent`` would buy us nothing except
a tool-calling envelope they never need, plus one extra LLM round
trip per call. We invoke the underlying LangChain Runnable directly.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from research_agent.llm.provider import ModelRouter
from research_agent.llm.tier import ModelTier


# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------
CRITIC_SYSTEM_PROMPT = """\
You are the Critic in a self-reflection loop over a multi-agent
financial-research supervisor's final synthesis. The synthesis was
generated AFTER a team of specialists already returned their
findings, so your job is NOT to re-do research — it is to grade
whether the synthesis faithfully reflects what was returned.

Evaluate on FIVE dimensions; weight them equally:
  - faithfulness   : does every claim trace to a specialist's output
                     or to the user-provided context? No new facts.
  - citation       : when the synthesis quotes a number or a passage,
                     is the source named (e.g. "data_expert", "p.12 of
                     the annual report", or [Source N])?
  - completeness   : did the synthesis answer EVERY sub-question the
                     user asked? Numbered or bulleted user requests
                     are explicit sub-questions and MUST each be
                     addressed.
  - structure      : are the required sections present (Key findings
                     / Sources / explicit conclusions) and is the
                     ordering coherent?
  - clarity        : would a busy analyst be able to skim and act on
                     this in under a minute?

Output a SINGLE JSON object — no prose before or after — with
exactly these keys:

{
  "quality_score": <float in [0.0, 1.0]>,
  "reasoning":     "<one short paragraph justifying the score>",
  "feedback":      "<concrete, actionable bullet list (newline-separated)
                    of improvements the writer should apply; empty
                    string if quality_score >= 0.85>",
  "issues":        ["<concise issue label 1>", "<concise issue label 2>"]
}

Score calibration:
  >= 0.90 — production-quality, ship as-is.
  0.75-0.89 — competent, minor issues, ship after a light rewrite.
  0.50-0.74 — material gaps, REWRITE required.
  < 0.50  — broken, REWRITE required (likely missing whole sub-questions).
"""

WRITER_SYSTEM_PROMPT = """\
You are the Writer in a self-reflection loop. Your input is:

  1. The original user question.
  2. The transcript of specialist outputs the supervisor saw.
  3. The supervisor's CURRENT draft answer.
  4. The Critic's feedback on that draft (newline-separated bullets).

Your job: produce a REVISED final answer that addresses every
critic bullet WITHOUT inventing new facts. Specifically:

  * If the critic flagged a missing sub-question, find the
    relevant specialist output in the transcript and add a
    response section for it (with the specialist named).
  * If the critic flagged missing citations, add them — quote the
    specialist by role name (data_expert, report_expert, ...) or
    by source attribution the specialist provided (page number,
    file basename, etc.).
  * If the critic flagged structure issues, restructure to use
    the required sections:
        ### 核心发现 / Key findings  (3-5 bullets with concrete
        numbers and short quotations from the PDF where relevant)
        ### 数据来源 / Sources       (list of specialists called
        and what each contributed)
  * Preserve every numeric value that came from a specialist — do
    NOT round, re-state, or "tidy up" numbers.
  * Write in the user's language (Chinese if the user wrote in
    Chinese).
  * Output ONLY the revised final answer text, no preamble or
    JSON envelope.

Hard rule: if the transcript has no evidence for a claim, REMOVE
the claim from the revision. Better to omit than to hallucinate.
"""


# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------
class ReflectionState(TypedDict, total=False):
    """Internal state for the reflection subgraph.

    Why TypedDict + ``total=False``: most fields are *populated by
    nodes as the graph runs*; declaring them all required would force
    every node to default-fill keys it doesn't own, which violates
    the single-responsibility intent of each node.

    Why ``messages`` uses ``add_messages``: identical to the rest of
    the LangGraph project — the reducer dedupes by message id so
    re-entrant nodes don't duplicate transcripts.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    """Input transcript: user query + specialist outputs + supervisor's draft.

    The very last non-tool-call ``AIMessage`` is treated as the
    supervisor's draft and is what the critic + writer operate on.
    """

    draft: str
    """The text under critique — supervisor's original on iteration 0,
    or the writer's most recent rewrite on later iterations."""

    critique: dict[str, Any]
    """Latest Critic verdict: ``{quality_score, reasoning, feedback, issues}``."""

    iteration: int
    """How many critic invocations have run so far. Starts at 0."""

    history: list[dict[str, Any]]
    """Per-iteration audit trail of ``{iteration, draft, critique}``.

    Useful in LangSmith traces and for the "show me what you tried"
    debugging story when reflection plateaus."""

    best_draft: str
    """Highest-scoring draft observed across iterations.

    Reflection terminates by returning *the best* draft, not necessarily
    the latest. If a 2nd rewrite scores LOWER than the 1st (the LLM
    "over-corrected" on a feedback bullet), we should not regress.
    """

    best_score: float
    """Score corresponding to ``best_draft``."""


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_BRACE = re.compile(r"(\{.*\})", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any]:
    """Return the first parseable JSON object in ``text``.

    LLMs frequently wrap JSON in fenced code blocks or add a
    one-line preamble ("Here is the JSON:") despite the system
    prompt asking otherwise. We try three strategies in order:

      1. Parse the whole string (best case — strict prompt obeyed).
      2. Pull the contents of a fenced ``json`` block if present.
      3. Pull the *first* ``{...}`` substring greedily.

    Returns an empty dict on total failure rather than raising —
    the critic node treats an unparseable critique as score 0.0
    (forcing a rewrite) instead of crashing the whole pipeline.
    """
    candidate = text.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    m = _JSON_FENCE.search(candidate)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    m = _JSON_BRACE.search(candidate)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    return {}


def _normalise_critique(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce a raw LLM-emitted critique dict into our canonical shape.

    Defends against three common deformations seen in practice:

      - ``quality_score`` returned as a string ("0.85" or "85%").
      - ``feedback`` returned as a list of strings instead of a single
        newline-joined string.
      - ``issues`` missing entirely (some models inline issues into
        ``feedback`` instead).
    """
    score = raw.get("quality_score", 0.0)
    if isinstance(score, str):
        cleaned = score.strip().rstrip("%")
        try:
            score = float(cleaned)
        except ValueError:
            score = 0.0
        # "85%" → 0.85
        if score > 1.0:
            score /= 100.0
    elif not isinstance(score, (int, float)):
        score = 0.0
    score = max(0.0, min(1.0, float(score)))

    feedback = raw.get("feedback", "")
    if isinstance(feedback, list):
        feedback = "\n".join(str(item) for item in feedback)
    elif not isinstance(feedback, str):
        feedback = str(feedback)

    issues = raw.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)] if issues else []

    return {
        "quality_score": score,
        "reasoning": str(raw.get("reasoning", "")),
        "feedback": feedback,
        "issues": issues,
    }


def _extract_supervisor_draft(messages: list[BaseMessage]) -> str:
    """Pull the supervisor's final synthesis out of the transcript.

    The supervisor's final answer is the LAST ``AIMessage`` whose
    ``tool_calls`` list is empty / absent (every hand-off is itself an
    ``AIMessage`` carrying a ``transfer_to_<name>`` tool call).

    Returns an empty string when no such message exists — the
    reflection subgraph will still run (the critic will score 0.0)
    but downstream code can detect "nothing to reflect on" by
    checking whether ``draft`` is empty.
    """
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if getattr(msg, "tool_calls", None):
            continue
        content = msg.content
        if isinstance(content, str) and content.strip():
            return content
    return ""


def _format_transcript(messages: list[BaseMessage], *, max_chars: int = 8000) -> str:
    """Render the supervisor transcript for the writer's context window.

    We label each message by role so the writer can name specialists
    by their LangGraph node name when adding citations. The total
    output is hard-capped at ``max_chars`` — long supervisor sessions
    can otherwise eat the writer's context budget. We keep the TAIL
    of the transcript (the most recent + most relevant messages) and
    drop the head, since the synthesis is built from later messages.
    """
    parts: list[str] = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else (
            "system" if isinstance(msg, SystemMessage) else (
                getattr(msg, "name", None) or "assistant"
            )
        )
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if not content.strip():
            continue
        parts.append(f"[{role}]\n{content}")
    rendered = "\n\n".join(parts)
    if len(rendered) <= max_chars:
        return rendered
    # Tail-keep with a head marker so the writer knows truncation
    # happened.
    return "... (transcript truncated) ...\n\n" + rendered[-max_chars:]


# ---------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------
def _build_critic_node(
    model_router: ModelRouter,
    *,
    pass_threshold: float,
):
    """Create the critic node closure. Captured kwargs become invariants."""

    critic_model = model_router.get_model(ModelTier.LIGHT)

    async def critic_node(state: ReflectionState) -> dict[str, Any]:
        """Score the current draft on the five reflection dimensions.

        On the first invocation we initialise ``draft`` /
        ``best_draft`` from the supervisor's last AIMessage. Later
        invocations score whatever the writer just produced.
        """
        iteration = state.get("iteration", 0)
        messages = state.get("messages", [])

        # First entry: seed ``draft`` from the supervisor's output.
        draft = state.get("draft", "")
        if not draft:
            draft = _extract_supervisor_draft(messages)

        if not draft.strip():
            # Nothing to critique — emit a zero-score critique so the
            # router falls through to finalize with an empty answer
            # rather than spinning forever.
            empty_critique = {
                "quality_score": 0.0,
                "reasoning": "no supervisor draft available to critique",
                "feedback": "",
                "issues": ["empty_draft"],
            }
            return {
                "draft": "",
                "critique": empty_critique,
                "iteration": iteration + 1,
                "history": [
                    *state.get("history", []),
                    {"iteration": iteration, "draft": "", "critique": empty_critique},
                ],
                "best_draft": state.get("best_draft", ""),
                "best_score": state.get("best_score", 0.0),
            }

        prompt_messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "## User question (original)\n"
                    f"{_format_transcript([m for m in messages if isinstance(m, HumanMessage)], max_chars=2000)}\n\n"
                    "## Draft answer to evaluate\n"
                    f"{draft}\n\n"
                    "Return the JSON verdict now."
                )
            ),
        ]

        response = await critic_model.ainvoke(prompt_messages)
        raw_text = response.content if isinstance(response.content, str) else str(response.content)
        critique = _normalise_critique(_extract_json(raw_text))

        score = critique["quality_score"]
        best_score = state.get("best_score", -1.0)
        best_draft = state.get("best_draft", "")
        if score > best_score:
            best_score = score
            best_draft = draft

        logger.info(
            "Reflection critic iter={} score={:.2f} threshold={:.2f}",
            iteration,
            score,
            pass_threshold,
        )

        return {
            "draft": draft,
            "critique": critique,
            "iteration": iteration + 1,
            "history": [
                *state.get("history", []),
                {"iteration": iteration, "draft": draft, "critique": critique},
            ],
            "best_draft": best_draft,
            "best_score": best_score,
        }

    return critic_node


def _build_writer_node(model_router: ModelRouter):
    """Create the writer node closure that consumes critic feedback."""

    writer_model = model_router.get_model(ModelTier.HEAVY)

    async def writer_node(state: ReflectionState) -> dict[str, Any]:
        """Produce a revised draft based on the latest critique."""
        messages = state.get("messages", [])
        prev_draft = state.get("draft", "")
        critique = state.get("critique", {})
        feedback = critique.get("feedback", "") if isinstance(critique, dict) else ""

        prompt_messages = [
            SystemMessage(content=WRITER_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "## Specialist transcript\n"
                    f"{_format_transcript(messages)}\n\n"
                    "## Current draft\n"
                    f"{prev_draft}\n\n"
                    "## Critic feedback (act on each bullet)\n"
                    f"{feedback or '(no feedback — polish the draft for clarity and citation density)'}\n\n"
                    "Output ONLY the revised answer text. No JSON, no preamble."
                )
            ),
        ]

        response = await writer_model.ainvoke(prompt_messages)
        new_draft = response.content if isinstance(response.content, str) else str(response.content)

        return {"draft": new_draft.strip()}

    return writer_node


def _build_finalize_node():
    """Append the chosen final draft back into the messages stream."""

    async def finalize_node(state: ReflectionState) -> dict[str, Any]:
        """Emit the BEST observed draft as a new ``AIMessage``.

        Returning the best (not the latest) draft is intentional —
        the LLM sometimes over-corrects on critique feedback and
        produces a regression on iteration N+1; we keep iteration N.
        """
        best = state.get("best_draft", "") or state.get("draft", "")
        critique = state.get("critique", {})
        if isinstance(critique, dict):
            score = critique.get("quality_score", 0.0)
        else:
            score = 0.0

        final_msg = AIMessage(
            content=best,
            name="reflection",
            additional_kwargs={
                "reflection": {
                    "iterations_run": state.get("iteration", 0),
                    "final_score": state.get("best_score", score),
                    "history_summary": [
                        {
                            "iteration": h["iteration"],
                            "score": h["critique"].get("quality_score", 0.0),
                            "issues": h["critique"].get("issues", []),
                        }
                        for h in state.get("history", [])
                    ],
                }
            },
        )
        return {"messages": [final_msg]}

    return finalize_node


def _build_router(
    *,
    pass_threshold: float,
    max_iterations: int,
):
    """Return the conditional-edge function that decides write vs finalize."""

    def route(state: ReflectionState) -> str:
        critique = state.get("critique", {})
        score = critique.get("quality_score", 0.0) if isinstance(critique, dict) else 0.0
        iteration = state.get("iteration", 0)

        # ``iteration`` counts critics already RUN (post-increment in
        # critic_node). ``max_iterations`` is the cap on REWRITES, so
        # we allow up to (max_iterations + 1) critic invocations
        # before forcing termination.
        if score >= pass_threshold:
            return "finalize"
        if iteration >= max_iterations + 1:
            return "finalize"
        return "write"

    return route


# ---------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------
def build_reflection_subgraph(
    *,
    model_router: ModelRouter,
    pass_threshold: float = 0.85,
    max_iterations: int = 2,
) -> CompiledStateGraph:
    """Compile the reflection critic + writer loop.

    Args:
        model_router: Shared router. The critic uses
            :attr:`ModelTier.LIGHT` (a grader is a classification
            task, not a creative writing task), the writer uses
            :attr:`ModelTier.HEAVY` (the rewrite IS creative
            synthesis under tight constraints).
        pass_threshold: Score at or above which the loop terminates
            on the current draft. Default 0.85 is calibrated for the
            critic prompt's "ship after a light rewrite" band; lower
            it if you find reflection rarely catches anything,
            raise it if you find it never stops.
        max_iterations: Maximum number of REWRITES (writer node
            invocations). At ``max_iterations=2`` the worst case is
            3 critics + 2 writers = 5 LLM calls. Set to 0 to make
            the subgraph a pure quality probe (one critic, never
            rewrites — useful for ablation studies).

    Returns:
        A compiled ``StateGraph`` consumable via ``ainvoke`` /
        ``astream``. The output state's ``messages`` will contain
        exactly the input messages plus ONE appended ``AIMessage``
        whose ``additional_kwargs['reflection']`` carries the audit
        trail.
    """
    graph: StateGraph = StateGraph(ReflectionState)

    graph.add_node("critic", _build_critic_node(model_router, pass_threshold=pass_threshold))
    graph.add_node("writer", _build_writer_node(model_router))
    graph.add_node("finalize", _build_finalize_node())

    graph.add_edge(START, "critic")
    graph.add_conditional_edges(
        "critic",
        _build_router(pass_threshold=pass_threshold, max_iterations=max_iterations),
        {"write": "writer", "finalize": "finalize"},
    )
    # After writer rewrites, always re-critique.
    graph.add_edge("writer", "critic")
    graph.add_edge("finalize", END)

    compiled = graph.compile()
    logger.info(
        "Reflection subgraph compiled: pass_threshold={:.2f} max_iterations={}",
        pass_threshold,
        max_iterations,
    )
    return compiled


__all__ = [
    "build_reflection_subgraph",
    "ReflectionState",
    "CRITIC_SYSTEM_PROMPT",
    "WRITER_SYSTEM_PROMPT",
]
