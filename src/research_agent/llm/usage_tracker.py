"""按 Agent 和模型层级追踪 Token 用量。

提供 :class:`UsageTracker` 用于累计 Token 数与估算费用，
以及 :class:`UsageCallbackHandler` — 一个 LangChain 回调，自动将每个 ``on_llm_end`` 事件喂入追踪器。
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
    total_cost_cny: float = 0.0


MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (每百万输入 Token, 每百万输出 Token) 单位：人民币（元）
    # --- DeepSeek ---
    "deepseek-v4-pro": (12.00, 24.00),
    "deepseek-v4-flash": (1.00, 2.00),
    # --- Qwen (DashScope) ---
    "qwen3-max": (2.50, 10.00),
    "qwen3.6-plus": (2.00, 12.00),
    "qwen-plus": (0.80, 2.00),
}


class UsageTracker:
    """线程安全的 Token 用量追踪器，附带费用估算。"""

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
                rec.total_cost_cny += cost

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
                "total_cost_cny": sum(r.total_cost_cny for r in self._by_model.values()),
            }

    def reset(self) -> None:
        with self._lock:
            self._by_agent.clear()
            self._by_model.clear()


class UsageCallbackHandler(BaseCallbackHandler):
    """将 ``on_llm_end`` Token 用量导入 :class:`UsageTracker` 的 LangChain 回调。

    通过 ``callbacks`` 关键字参数附加到 ``ChatOpenAI`` 实例::

        handler = UsageCallbackHandler(tracker, tier_label="heavy")
        llm = ChatOpenAI(..., callbacks=[handler])

    该 handler 从 ``LLMResult.llm_output``（OpenAI 兼容提供商填充的字典）中提取 ``token_usage``，并记录在 ``(tier_label, model_name)`` 下。

    性能说明
    --------
    记录路径快速且无阻塞（单次 ``threading.Lock`` 获取 + 字典更新）。
    设置 ``run_inline = True``，这样当被包装的 LLM 通过 ``ainvoke`` /``astream`` 调用时，LangChain 会直接在事件循环上执行，
    而非为每次 LLM 调用支付默认的 ``run_in_executor`` 往返开销。对同步 ``invoke`` 调用者无影响。
    """

    # 在异步分发路径中同步执行。安全因为函数体是非阻塞的（锁 + 字典更新 +结构化日志）。见 ``langchain_core.callbacks.base.BaseCallbackHandler.run_inline``。
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
        "total_cost_cny": round(rec.total_cost_cny, 6),
    }
