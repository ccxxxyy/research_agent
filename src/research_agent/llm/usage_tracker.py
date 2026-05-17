"""Token usage tracking per agent and model tier.

Provides :class:`UsageTracker` for accumulating token counts and
estimated costs, plus :class:`UsageCallbackHandler` — a LangChain
callback that feeds every ``on_llm_end`` event into the tracker
automatically.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from loguru import logger


@dataclass
class UsageRecord:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
    total_cost_usd: float = 0.0


MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_1M_tokens, output_per_1M_tokens) in USD
    # --- OpenAI ---
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    # --- DeepSeek ---
    "deepseek-chat": (0.07, 0.27),
    "deepseek-reasoner": (0.55, 2.19),
    "deepseek-v4-pro": (0.55, 2.19),
    # --- Qwen (DashScope) ---
    "qwen3-max-2026-01-23": (0.16, 0.64),
    "qwen3-max": (0.16, 0.64),
    "qwen3.6-plus": (0.08, 0.32),
    "qwen-plus": (0.08, 0.32),
}


class UsageTracker:
    """Thread-safe token usage tracker with cost estimation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_agent: dict[str, UsageRecord] = {}
        self._by_model: dict[str, UsageRecord] = {}

    def record(
        self,
        agent_name: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        total = prompt_tokens + completion_tokens
        cost = self._estimate_cost(model_name, prompt_tokens, completion_tokens)

        with self._lock:
            for key, store in [
                (agent_name, self._by_agent),
                (model_name, self._by_model),
            ]:
                rec = store.setdefault(key, UsageRecord())
                rec.prompt_tokens += prompt_tokens
                rec.completion_tokens += completion_tokens
                rec.total_tokens += total
                rec.call_count += 1
                rec.total_cost_usd += cost

    @staticmethod
    def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = MODEL_PRICING.get(model)
        if not pricing:
            return 0.0
        input_price, output_price = pricing
        return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000

    def summary(self) -> dict:
        with self._lock:
            return {
                "by_agent": {k: _record_to_dict(v) for k, v in self._by_agent.items()},
                "by_model": {k: _record_to_dict(v) for k, v in self._by_model.items()},
                "total_cost_usd": sum(r.total_cost_usd for r in self._by_model.values()),
            }

    def reset(self) -> None:
        with self._lock:
            self._by_agent.clear()
            self._by_model.clear()


class UsageCallbackHandler(BaseCallbackHandler):
    """LangChain callback that pipes ``on_llm_end`` token usage into a :class:`UsageTracker`.

    Attach to a ``ChatOpenAI`` instance via its ``callbacks`` kwarg::

        handler = UsageCallbackHandler(tracker, tier_label="heavy")
        llm = ChatOpenAI(..., callbacks=[handler])

    The handler extracts ``token_usage`` from ``LLMResult.llm_output``
    (the dict OpenAI-compatible providers populate) and records it
    under ``(tier_label, model_name)``.

    Performance note
    ----------------
    The recording path is fast and non-blocking (a single
    ``threading.Lock`` acquire + a dict update). We set
    ``run_inline = True`` so LangChain executes us **directly on the
    event loop** when the wrapped LLM is invoked via ``ainvoke`` /
    ``astream`` — instead of paying the default ``run_in_executor``
    round-trip for every LLM call. For sync ``invoke`` callers nothing
    changes.
    """

    # Run synchronously in the async dispatch path. Safe because the
    # body is non-blocking (lock + dict update + structured log). See
    # ``langchain_core.callbacks.base.BaseCallbackHandler.run_inline``.
    run_inline: bool = True

    def __init__(self, tracker: UsageTracker, *, tier_label: str = "") -> None:
        super().__init__()
        self._tracker = tracker
        self._tier_label = tier_label

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        llm_output = response.llm_output or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)

        if prompt == 0 and completion == 0:
            return

        model_name = llm_output.get("model_name", "") or llm_output.get("model", "")
        agent_label = self._tier_label or model_name

        self._tracker.record(
            agent_name=agent_label,
            model_name=model_name or "unknown",
            prompt_tokens=prompt,
            completion_tokens=completion,
        )
        logger.trace(
            "LLM usage: tier={} model={} prompt={} completion={}",
            self._tier_label, model_name, prompt, completion,
        )


def _record_to_dict(rec: UsageRecord) -> dict:
    return {
        "prompt_tokens": rec.prompt_tokens,
        "completion_tokens": rec.completion_tokens,
        "total_tokens": rec.total_tokens,
        "call_count": rec.call_count,
        "total_cost_usd": round(rec.total_cost_usd, 6),
    }
