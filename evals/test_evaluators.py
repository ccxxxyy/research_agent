"""评估评分器的离线单元测试。

使用合成输入验证评估器逻辑，无 LLM 调用、无 LangSmith API、无网络。
在常规 ``pytest -m "not network"`` 门控下运行。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from evals.evaluators import (
    _build_reply_quality_evaluator,
    keyword_coverage,
    memory_persistence,
    routing_accuracy,
    tool_selection_precision,
)


def _make_run(outputs: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(outputs=outputs)


def _make_example(inputs: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(inputs=inputs)


# ---------------------------------------------------------------------------
# routing_accuracy（路由准确率）
# ---------------------------------------------------------------------------


class TestRoutingAccuracy:
    def test_exact_match(self) -> None:
        run = _make_run({"specialists_reached": ["data_expert", "report_expert"]})
        example = _make_example({"expected_specialists": ["data_expert", "report_expert"]})
        result = routing_accuracy(run, example)
        assert result["score"] == 1.0

    def test_exact_match_different_order(self) -> None:
        run = _make_run({"specialists_reached": ["report_expert", "data_expert"]})
        example = _make_example({"expected_specialists": ["data_expert", "report_expert"]})
        assert routing_accuracy(run, example)["score"] == 1.0

    def test_partial_overlap(self) -> None:
        run = _make_run({"specialists_reached": ["data_expert", "coder_expert"]})
        example = _make_example({"expected_specialists": ["data_expert", "report_expert"]})
        result = routing_accuracy(run, example)
        # Jaccard 相似度: {data_expert} / {data_expert, report_expert, coder_expert} = 1/3
        assert abs(result["score"] - 1 / 3) < 0.01

    def test_no_overlap(self) -> None:
        run = _make_run({"specialists_reached": ["coder_expert"]})
        example = _make_example({"expected_specialists": ["data_expert"]})
        assert routing_accuracy(run, example)["score"] == 0.0

    def test_both_empty(self) -> None:
        run = _make_run({"specialists_reached": []})
        example = _make_example({"expected_specialists": []})
        assert routing_accuracy(run, example)["score"] == 1.0

    def test_expected_empty_actual_nonempty(self) -> None:
        run = _make_run({"specialists_reached": ["data_expert"]})
        example = _make_example({"expected_specialists": []})
        assert routing_accuracy(run, example)["score"] == 0.0

    def test_expected_nonempty_actual_empty(self) -> None:
        run = _make_run({"specialists_reached": []})
        example = _make_example({"expected_specialists": ["data_expert"]})
        assert routing_accuracy(run, example)["score"] == 0.0

    def test_comment_contains_sets(self) -> None:
        run = _make_run({"specialists_reached": ["data_expert"]})
        example = _make_example({"expected_specialists": ["data_expert"]})
        result = routing_accuracy(run, example)
        assert "data_expert" in result["comment"]


# ---------------------------------------------------------------------------
# reply_quality（回复质量，使用模拟 LLM）
# ---------------------------------------------------------------------------


class TestReplyQuality:
    @pytest.mark.asyncio
    async def test_good_reply_scores_high(self) -> None:
        async def fake_llm(prompt: str) -> str:
            return '{"relevance": 5, "completeness": 5, "factuality": 5, "reasoning": "perfect"}'

        evaluator = _build_reply_quality_evaluator(fake_llm)
        run = _make_run({"reply": "宁德时代 2023 年营收 4009 亿元"})
        example = _make_example(
            {
                "query": "宁德时代营收",
                "expected_reply_keywords": ["营收"],
            }
        )
        result = await evaluator(run, example)
        assert result["score"] == 1.0
        assert result["comment"] == "perfect"

    @pytest.mark.asyncio
    async def test_mediocre_reply(self) -> None:
        async def fake_llm(prompt: str) -> str:
            return '{"relevance": 3, "completeness": 2, "factuality": 3, "reasoning": "mediocre"}'

        evaluator = _build_reply_quality_evaluator(fake_llm)
        run = _make_run({"reply": "some text"})
        example = _make_example({"query": "q", "expected_reply_keywords": []})
        result = await evaluator(run, example)
        # 均值 = (3+2+3)/3 = 2.667; 归一化 = (2.667-1)/4 = 0.417
        assert 0.4 <= result["score"] <= 0.45

    @pytest.mark.asyncio
    async def test_empty_reply_scores_zero(self) -> None:
        async def fake_llm(prompt: str) -> str:
            return "{}"

        evaluator = _build_reply_quality_evaluator(fake_llm)
        run = _make_run({"reply": ""})
        example = _make_example({"query": "q", "expected_reply_keywords": []})
        result = await evaluator(run, example)
        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_llm_error_returns_fallback(self) -> None:
        async def broken_llm(prompt: str) -> str:
            raise ConnectionError("timeout")

        evaluator = _build_reply_quality_evaluator(broken_llm)
        run = _make_run({"reply": "some answer"})
        example = _make_example({"query": "q", "expected_reply_keywords": []})
        result = await evaluator(run, example)
        assert result["score"] == 0.5
        assert "judge error" in result["comment"]

    @pytest.mark.asyncio
    async def test_unparseable_response(self) -> None:
        async def bad_llm(prompt: str) -> str:
            return "I cannot evaluate this."

        evaluator = _build_reply_quality_evaluator(bad_llm)
        run = _make_run({"reply": "some answer"})
        example = _make_example({"query": "q", "expected_reply_keywords": []})
        result = await evaluator(run, example)
        assert result["score"] == 0.5
        assert "unparseable" in result["comment"]


# ---------------------------------------------------------------------------
# memory_persistence（记忆持久化）
# ---------------------------------------------------------------------------


class TestMemoryPersistence:
    def test_normal_user_saved(self) -> None:
        run = _make_run({"reply": "answer text", "memory_saved": True})
        example = _make_example({"user_id": "alice"})
        assert memory_persistence(run, example)["score"] == 1.0

    def test_normal_user_not_saved(self) -> None:
        run = _make_run({"reply": "answer text", "memory_saved": False})
        example = _make_example({"user_id": "alice"})
        result = memory_persistence(run, example)
        assert result["score"] == 0.0
        assert "缺失" in result["comment"]

    def test_anonymous_not_saved_is_correct(self) -> None:
        run = _make_run({"reply": "answer", "memory_saved": False})
        example = _make_example({"user_id": "anonymous"})
        assert memory_persistence(run, example)["score"] == 1.0

    def test_anonymous_saved_is_wrong(self) -> None:
        run = _make_run({"reply": "answer", "memory_saved": True})
        example = _make_example({"user_id": "anonymous"})
        assert memory_persistence(run, example)["score"] == 0.0

    def test_empty_reply_not_saved_is_correct(self) -> None:
        run = _make_run({"reply": "", "memory_saved": False})
        example = _make_example({"user_id": "alice"})
        assert memory_persistence(run, example)["score"] == 1.0

    def test_empty_reply_saved_is_wrong(self) -> None:
        run = _make_run({"reply": "", "memory_saved": True})
        example = _make_example({"user_id": "alice"})
        assert memory_persistence(run, example)["score"] == 0.0


# ---------------------------------------------------------------------------
# keyword_coverage（关键词覆盖率）
# ---------------------------------------------------------------------------


class TestKeywordCoverage:
    def test_all_keywords_present(self) -> None:
        run = _make_run({"reply": "宁德时代 2023 年营收 4009 亿元，净利润 441 亿"})
        example = _make_example({"expected_reply_keywords": ["营收", "利润"]})
        assert keyword_coverage(run, example)["score"] == 1.0

    def test_partial_keywords(self) -> None:
        run = _make_run({"reply": "宁德时代的营收数据如下"})
        example = _make_example({"expected_reply_keywords": ["营收", "利润", "ROE"]})
        result = keyword_coverage(run, example)
        assert abs(result["score"] - 1 / 3) < 0.01

    def test_no_keywords_expected(self) -> None:
        run = _make_run({"reply": "你好，有什么可以帮你的？"})
        example = _make_example({"expected_reply_keywords": []})
        assert keyword_coverage(run, example)["score"] == 1.0

    def test_none_keywords_expected(self) -> None:
        run = _make_run({"reply": "some reply"})
        example = _make_example({})
        assert keyword_coverage(run, example)["score"] == 1.0

    def test_no_keywords_found(self) -> None:
        run = _make_run({"reply": "这是一段无关的回复"})
        example = _make_example({"expected_reply_keywords": ["ROE", "PE"]})
        assert keyword_coverage(run, example)["score"] == 0.0

    def test_case_insensitive(self) -> None:
        run = _make_run({"reply": "The ROE of BYD is 20%"})
        example = _make_example({"expected_reply_keywords": ["roe", "byd"]})
        assert keyword_coverage(run, example)["score"] == 1.0

    def test_empty_reply(self) -> None:
        run = _make_run({"reply": ""})
        example = _make_example({"expected_reply_keywords": ["营收"]})
        assert keyword_coverage(run, example)["score"] == 0.0


# ---------------------------------------------------------------------------
# tool_selection_precision（工具选择精确度）
# ---------------------------------------------------------------------------


class TestToolSelectionPrecision:
    def test_perfect_precision(self) -> None:
        run = _make_run({"specialists_reached": ["data_expert", "news_expert"]})
        example = _make_example({"expected_specialists": ["data_expert", "news_expert"]})
        assert tool_selection_precision(run, example)["score"] == 1.0

    def test_over_routing(self) -> None:
        run = _make_run({"specialists_reached": ["data_expert", "news_expert", "coder_expert"]})
        example = _make_example({"expected_specialists": ["data_expert"]})
        result = tool_selection_precision(run, example)
        assert abs(result["score"] - 1 / 3) < 0.01
        assert "多余" in result["comment"]

    def test_no_actual_no_expected(self) -> None:
        run = _make_run({"specialists_reached": []})
        example = _make_example({"expected_specialists": []})
        assert tool_selection_precision(run, example)["score"] == 1.0

    def test_no_actual_but_expected(self) -> None:
        run = _make_run({"specialists_reached": []})
        example = _make_example({"expected_specialists": ["data_expert"]})
        assert tool_selection_precision(run, example)["score"] == 0.0

    def test_actual_subset_of_expected(self) -> None:
        run = _make_run({"specialists_reached": ["data_expert"]})
        example = _make_example({"expected_specialists": ["data_expert", "news_expert"]})
        assert tool_selection_precision(run, example)["score"] == 1.0

    def test_completely_wrong_routing(self) -> None:
        run = _make_run({"specialists_reached": ["coder_expert"]})
        example = _make_example({"expected_specialists": ["data_expert"]})
        assert tool_selection_precision(run, example)["score"] == 0.0
