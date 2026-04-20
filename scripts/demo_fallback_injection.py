"""Demonstrate LangChain's ``with_fallbacks`` by injecting model failures.

Run:
    uv run python scripts/demo_fallback_injection.py

This script is NOT a unit test — it is a narrated demo. It proves that
``ModelRouter.get_model(tier)`` produces a runnable that **transparently
falls back to a backup model when the primary raises**.

We run four scenarios:
    1. Primary OK, backup OK           → primary answers
    2. Primary ALWAYS FAILS, backup OK → backup answers (the key guarantee)
    3. Primary OK, backup ALWAYS FAILS → primary answers (backup is never touched)
    4. Primary FAILS, backup FAILS     → error surfaces to the caller
    5. Streaming call under failure    → same semantics on the streaming path

Each scenario uses an injected *stub* LLM so the outcome is deterministic
and the demo runs offline without hitting any real API.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableWithFallbacks


CALL_LOG: list[str] = []
"""Module-level audit trail: appended whenever a stub model is invoked.

We keep it at module scope because BaseChatModel is a pydantic model —
declaring mutable class attributes on the subclass would be misinterpreted
as pydantic fields.
"""


def _reset_log() -> None:
    CALL_LOG.clear()


# ---------------------------------------------------------------------------
# Stub chat models (deterministic, offline)
# ---------------------------------------------------------------------------

class _StubChatModel(BaseChatModel):
    """Minimal in-process chat model for fault-injection demos.

    - If ``should_fail`` is True every ``_generate`` call raises.
    - Otherwise returns an AIMessage whose content encodes the instance
      ``label`` so the caller can tell WHICH model answered.
    """

    label: str
    should_fail: bool = False
    failure_message: str = "simulated upstream failure"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        CALL_LOG.append(self.label)
        if self.should_fail:
            raise RuntimeError(f"[{self.label}] {self.failure_message}")
        text = f"answer-from-{self.label}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # Simulate a streaming failure by raising before yielding the first chunk.
        CALL_LOG.append(f"{self.label}:stream")
        if self.should_fail:
            raise RuntimeError(f"[{self.label}] {self.failure_message} (stream)")
        yield ChatGenerationChunk(message=AIMessageChunk(content=f"answer-from-{self.label}"))

    @property
    def _llm_type(self) -> str:
        return "stub"


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _describe(runnable: Any) -> str:
    return type(runnable).__name__


def _invoke(runnable: Any, question: str) -> str:
    _reset_log()
    try:
        resp = runnable.invoke([HumanMessage(content=question)])
        return f"OK   -> {resp.content}   (path: {CALL_LOG})"
    except Exception as e:
        return f"FAIL -> {type(e).__name__}: {e}   (path: {CALL_LOG})"


def scenario_1_both_healthy() -> None:
    _banner("Scenario 1 — primary OK, backup OK")
    primary = _StubChatModel(label="primary")
    backup = _StubChatModel(label="backup")
    chain: RunnableWithFallbacks = primary.with_fallbacks([backup])
    print(f"  chain type        : {_describe(chain)}")
    print(f"  invocation result : {_invoke(chain, 'hello')}")
    print("  expected          : primary answers, backup never consulted.")


def scenario_2_primary_fails() -> None:
    _banner("Scenario 2 — primary ALWAYS FAILS, backup OK   [key guarantee]")
    primary = _StubChatModel(label="primary", should_fail=True)
    backup = _StubChatModel(label="backup")
    chain = primary.with_fallbacks([backup])
    print(f"  chain type        : {_describe(chain)}")
    print(f"  invocation result : {_invoke(chain, 'hello')}")
    print("  expected          : primary is tried, raises, backup takes over,")
    print("                      caller sees 'answer-from-backup' transparently.")


def scenario_3_backup_never_needed() -> None:
    _banner("Scenario 3 — primary OK, backup would fail")
    primary = _StubChatModel(label="primary")
    backup = _StubChatModel(label="backup", should_fail=True)
    chain = primary.with_fallbacks([backup])
    print(f"  invocation result : {_invoke(chain, 'hello')}")
    print("  expected          : primary succeeds; backup is never exercised,")
    print("                      so its latent breakage stays hidden until needed.")


def scenario_4_both_fail() -> None:
    _banner("Scenario 4 — primary FAILS, backup FAILS   [error surfaces]")
    primary = _StubChatModel(label="primary", should_fail=True)
    backup = _StubChatModel(label="backup", should_fail=True)
    chain = primary.with_fallbacks([backup])
    print(f"  invocation result : {_invoke(chain, 'hello')}")
    print("  expected          : both tried, final exception propagates to caller.")


def scenario_5_streaming() -> None:
    _banner("Scenario 5 — streaming path: primary FAILS mid-open, backup streams")
    primary = _StubChatModel(label="primary", should_fail=True)
    backup = _StubChatModel(label="backup")
    chain = primary.with_fallbacks([backup])

    _reset_log()
    try:
        chunks: list[str] = []
        for chunk in chain.stream([HumanMessage(content="hello")]):
            chunks.append(str(chunk.content))
        print(f"  streamed content  : {''.join(chunks)}")
        print(f"  path              : {CALL_LOG}")
        print("  expected          : primary stream raises, backup stream is opened,")
        print("                      caller still receives a coherent stream.")
    except Exception as e:
        print(f"  streaming FAILED  : {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Bonus — prove the real ModelRouter uses the same mechanism
# ---------------------------------------------------------------------------

def scenario_6_real_router_shape() -> None:
    _banner("Scenario 6 — real ModelRouter.get_model(HEAVY) wires the same chain")
    from research_agent.config import get_settings
    from research_agent.llm.provider import ModelRouter
    from research_agent.llm.tier import ModelTier

    settings = get_settings()
    router = ModelRouter(settings.llm)

    heavy = router.get_model(ModelTier.HEAVY)
    medium = router.get_model(ModelTier.MEDIUM)
    light = router.get_model(ModelTier.LIGHT)

    print(f"  HEAVY  -> {_describe(heavy)}   (expected RunnableWithFallbacks)")
    print(f"  MEDIUM -> {_describe(medium)}  (expected RunnableWithFallbacks)")
    print(f"  LIGHT  -> {_describe(light)}   (expected plain ChatOpenAI — no further degradation)")

    print(
        "\n  Observation: tiers with a defined FALLBACK_CHAIN entry get wrapped\n"
        "  by .with_fallbacks(...), so a production DashScope outage on\n"
        "  deepseek-v3.2 transparently degrades to qwen-turbo without any\n"
        "  change to caller code."
    )


# ---------------------------------------------------------------------------
# Takeaway
# ---------------------------------------------------------------------------

def takeaway() -> None:
    _banner("Takeaway")
    print(
        """
  * with_fallbacks([backup]) returns a Runnable that is API-compatible
    with the original LLM. Caller code stays the same.
  * Order matters: the list is tried left to right; the FIRST success wins.
  * Exceptions from the primary are caught INSIDE the chain, not re-raised,
    unless ALL fallbacks also fail.
  * It layers cleanly with ChatOpenAI(max_retries=2): retries first absorb
    transient 5xx / network jitter; fallbacks absorb sustained outages or
    quota exhaustion. Checkpointer (Phase 2) then absorbs whole-process
    crashes. Together: three-layered fault tolerance.
  * Design choice in our tier map: fallbacks DEGRADE quality for
    availability (HEAVY -> MEDIUM -> LIGHT) rather than retrying at the
    same tier. That matches real outage patterns better than "same-tier
    retry" which is what max_retries already handles.
        """.rstrip()
    )


def main() -> None:
    scenario_1_both_healthy()
    scenario_2_primary_fails()
    scenario_3_backup_never_needed()
    scenario_4_both_fail()
    scenario_5_streaming()
    scenario_6_real_router_shape()
    takeaway()


if __name__ == "__main__":
    main()
