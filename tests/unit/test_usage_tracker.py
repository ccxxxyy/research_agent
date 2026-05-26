"""token 用量追踪、费用估算及 LangChain 回调的测试。"""

from __future__ import annotations

from uuid import uuid4

from langchain_core.outputs import LLMResult

from research_agent.llm.usage_tracker import UsageCallbackHandler, UsageTracker


class TestUsageTracker:
    def test_record_and_summary(self):
        tracker = UsageTracker()
        tracker.record("retriever", "qwen-plus", prompt_tokens=100, completion_tokens=50)
        tracker.record("writer", "deepseek-v4-pro", prompt_tokens=500, completion_tokens=200)

        summary = tracker.summary()

        assert "retriever" in summary["by_agent"]
        assert "writer" in summary["by_agent"]
        assert summary["by_agent"]["retriever"]["call_count"] == 1
        assert summary["by_agent"]["writer"]["total_tokens"] == 700
        assert summary["total_cost_cny"] > 0

    def test_reset(self):
        tracker = UsageTracker()
        tracker.record("test", "deepseek-v4-pro", prompt_tokens=100, completion_tokens=50)
        tracker.reset()
        summary = tracker.summary()
        assert summary["total_cost_cny"] == 0

    def test_cost_estimation_deepseek_v4_pro(self):
        tracker = UsageTracker()
        tracker.record("test", "deepseek-v4-pro", prompt_tokens=1_000_000, completion_tokens=1_000_000)
        summary = tracker.summary()
        assert abs(summary["total_cost_cny"] - 36.00) < 0.01

    def test_cost_estimation_qwen(self):
        tracker = UsageTracker()
        tracker.record("test", "qwen3-max", prompt_tokens=1_000_000, completion_tokens=1_000_000)
        summary = tracker.summary()
        assert abs(summary["total_cost_cny"] - 12.50) < 0.01
        assert summary["by_model"]["qwen3-max"]["call_count"] == 1

    def test_cost_estimation_qwen36_plus(self):
        tracker = UsageTracker()
        tracker.record("test", "qwen3.6-plus", prompt_tokens=1_000_000, completion_tokens=1_000_000)
        summary = tracker.summary()
        assert abs(summary["total_cost_cny"] - 14.00) < 0.01

    def test_unknown_model_zero_cost(self):
        tracker = UsageTracker()
        tracker.record("test", "some-unknown-model", prompt_tokens=500, completion_tokens=500)
        summary = tracker.summary()
        assert summary["total_cost_cny"] == 0.0
        assert summary["by_model"]["some-unknown-model"]["call_count"] == 1


class TestUsageCallbackHandler:
    """测试将 on_llm_end 事件转发到 UsageTracker 的 LangChain 回调。"""

    @staticmethod
    def _make_llm_result(
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
        model_name: str = "deepseek-v4-pro",
    ) -> LLMResult:
        return LLMResult(
            generations=[[]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "model_name": model_name,
            },
        )

    def test_callback_records_usage(self):
        tracker = UsageTracker()
        handler = UsageCallbackHandler(tracker, tier_label="heavy")

        handler.on_llm_end(
            self._make_llm_result(200, 80, "deepseek-v4-pro"),
            run_id=uuid4(),
        )

        summary = tracker.summary()
        assert summary["by_agent"]["heavy"]["call_count"] == 1
        assert summary["by_agent"]["heavy"]["prompt_tokens"] == 200
        assert summary["by_agent"]["heavy"]["completion_tokens"] == 80
        assert "deepseek-v4-pro" in summary["by_model"]

    def test_callback_skips_zero_tokens(self):
        tracker = UsageTracker()
        handler = UsageCallbackHandler(tracker, tier_label="light")

        handler.on_llm_end(
            self._make_llm_result(0, 0, "qwen3-max"),
            run_id=uuid4(),
        )

        summary = tracker.summary()
        assert summary["by_agent"] == {}
        assert summary["by_model"] == {}

    def test_callback_handles_missing_llm_output(self):
        tracker = UsageTracker()
        handler = UsageCallbackHandler(tracker, tier_label="medium")

        result = LLMResult(generations=[[]], llm_output=None)
        handler.on_llm_end(result, run_id=uuid4())

        summary = tracker.summary()
        assert summary["by_agent"] == {}

    def test_callback_accumulates_multiple_calls(self):
        tracker = UsageTracker()
        handler = UsageCallbackHandler(tracker, tier_label="medium")

        for _ in range(3):
            handler.on_llm_end(
                self._make_llm_result(100, 50, "qwen3.6-plus"),
                run_id=uuid4(),
            )

        summary = tracker.summary()
        assert summary["by_agent"]["medium"]["call_count"] == 3
        assert summary["by_agent"]["medium"]["prompt_tokens"] == 300
        assert summary["by_model"]["qwen3.6-plus"]["total_tokens"] == 450

    def test_callback_uses_usage_key(self):
        """部分供应商将 token 计数放在 'usage' 而非 'token_usage' 键下。"""
        tracker = UsageTracker()
        handler = UsageCallbackHandler(tracker, tier_label="light")

        result = LLMResult(
            generations=[[]],
            llm_output={
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "model": "qwen3-max",
            },
        )
        handler.on_llm_end(result, run_id=uuid4())

        summary = tracker.summary()
        assert summary["by_agent"]["light"]["prompt_tokens"] == 10
