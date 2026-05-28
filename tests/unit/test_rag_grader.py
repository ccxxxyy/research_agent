"""rag.grader 单元测试 — RetrievalGrader。"""

from __future__ import annotations

from research_agent.rag.grader import RetrievalGrader


class TestRetrievalGrader:
    def setup_method(self):
        self.grader = RetrievalGrader(high_threshold=0.65, medium_threshold=0.40)

    def test_high_quality(self):
        assert self.grader.grade(top_score=0.85, mean_score=0.60, unique_sources=2) == "high"

    def test_high_quality_boundary(self):
        assert self.grader.grade(top_score=0.65, mean_score=0.50, unique_sources=1) == "high"

    def test_high_requires_at_least_one_source(self):
        assert self.grader.grade(top_score=0.90, mean_score=0.70, unique_sources=0) != "high"

    def test_medium_quality(self):
        assert self.grader.grade(top_score=0.50, mean_score=0.30, unique_sources=2) == "medium"

    def test_medium_boundary(self):
        assert self.grader.grade(top_score=0.40, mean_score=0.24, unique_sources=1) == "medium"

    def test_medium_needs_mean_above_threshold(self):
        result = self.grader.grade(top_score=0.45, mean_score=0.10, unique_sources=2)
        assert result == "low"

    def test_low_quality(self):
        assert self.grader.grade(top_score=0.20, mean_score=0.10, unique_sources=1) == "low"

    def test_zero_scores(self):
        assert self.grader.grade(top_score=0.0, mean_score=0.0, unique_sources=0) == "low"

    def test_custom_thresholds(self):
        strict = RetrievalGrader(high_threshold=0.90, medium_threshold=0.70)
        assert strict.grade(top_score=0.85, mean_score=0.60, unique_sources=2) == "medium"
        assert strict.grade(top_score=0.50, mean_score=0.40, unique_sources=2) == "low"

    def test_grade_consistency_with_original_function(self):
        """验证与 knowledge_server._classify_quality 相同的评分逻辑。"""
        g = RetrievalGrader(high_threshold=0.65, medium_threshold=0.40)
        assert g.grade(0.65, 0.5, 1) == "high"
        assert g.grade(0.64, 0.5, 1) == "medium"
        assert g.grade(0.40, 0.24, 0) == "medium"
        assert g.grade(0.39, 0.2, 1) == "low"
