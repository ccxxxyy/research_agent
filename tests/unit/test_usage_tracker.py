"""Tests for token usage tracking and cost estimation."""

from research_agent.llm.usage_tracker import UsageTracker


class TestUsageTracker:
    def test_record_and_summary(self):
        tracker = UsageTracker()
        tracker.record("retriever", "deepseek-chat", prompt_tokens=100, completion_tokens=50)
        tracker.record("writer", "gpt-4o", prompt_tokens=500, completion_tokens=200)

        summary = tracker.summary()

        assert "retriever" in summary["by_agent"]
        assert "writer" in summary["by_agent"]
        assert summary["by_agent"]["retriever"]["call_count"] == 1
        assert summary["by_agent"]["writer"]["total_tokens"] == 700
        assert summary["total_cost_usd"] > 0

    def test_reset(self):
        tracker = UsageTracker()
        tracker.record("test", "gpt-4o", prompt_tokens=100, completion_tokens=50)
        tracker.reset()
        summary = tracker.summary()
        assert summary["total_cost_usd"] == 0

    def test_cost_estimation(self):
        tracker = UsageTracker()
        # gpt-4o: $2.50/1M input, $10.00/1M output
        tracker.record("test", "gpt-4o", prompt_tokens=1_000_000, completion_tokens=1_000_000)
        summary = tracker.summary()
        assert abs(summary["total_cost_usd"] - 12.50) < 0.01
