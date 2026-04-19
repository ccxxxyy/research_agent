"""Tests for conditional edge routing logic."""

from research_agent.graph.edges import should_reflect, should_retry_retrieval
from research_agent.graph.state import ResearchState


class TestRetryRetrieval:
    def test_retry_when_irrelevant(self):
        state: ResearchState = {
            "query": "test",
            "messages": [],
            "retrieval_grade": "irrelevant",
            "retrieval_retry_count": 0,
            "max_retrieval_retries": 3,
        }
        assert should_retry_retrieval(state) == "rewrite_query"

    def test_proceed_when_relevant(self):
        state: ResearchState = {
            "query": "test",
            "messages": [],
            "retrieval_grade": "relevant",
        }
        assert should_retry_retrieval(state) == "analyze"

    def test_stop_retry_at_max(self):
        state: ResearchState = {
            "query": "test",
            "messages": [],
            "retrieval_grade": "irrelevant",
            "retrieval_retry_count": 3,
            "max_retrieval_retries": 3,
        }
        assert should_retry_retrieval(state) == "analyze"

    def test_defaults_to_analyze(self):
        state: ResearchState = {"query": "test", "messages": []}
        assert should_retry_retrieval(state) == "analyze"


class TestReflection:
    def test_reflect_when_low_score(self):
        state: ResearchState = {
            "query": "test",
            "messages": [],
            "quality_score": 0.5,
            "reflection_count": 0,
            "max_reflections": 3,
        }
        assert should_reflect(state) == "reflect"

    def test_finalize_when_high_score(self):
        state: ResearchState = {
            "query": "test",
            "messages": [],
            "quality_score": 0.9,
            "reflection_count": 1,
            "max_reflections": 3,
        }
        assert should_reflect(state) == "finalize"

    def test_finalize_at_max_reflections(self):
        state: ResearchState = {
            "query": "test",
            "messages": [],
            "quality_score": 0.3,
            "reflection_count": 3,
            "max_reflections": 3,
        }
        assert should_reflect(state) == "finalize"

    def test_defaults_to_reflect(self):
        state: ResearchState = {"query": "test", "messages": []}
        assert should_reflect(state) == "reflect"
