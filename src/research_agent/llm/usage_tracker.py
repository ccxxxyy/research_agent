"""Token usage tracking per agent and model tier."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class UsageRecord:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
    total_cost_usd: float = 0.0


# Approximate pricing per 1M tokens (input/output)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "deepseek-chat": (0.07, 0.27),
    "deepseek-reasoner": (0.55, 2.19),
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


@dataclass
class _CallbackState:
    agent_name: str = ""
    model_name: str = ""


def _record_to_dict(rec: UsageRecord) -> dict:
    return {
        "prompt_tokens": rec.prompt_tokens,
        "completion_tokens": rec.completion_tokens,
        "total_tokens": rec.total_tokens,
        "call_count": rec.call_count,
        "total_cost_usd": round(rec.total_cost_usd, 6),
    }
