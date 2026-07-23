"""静态知识语义缓存单元测试（默认不加载真实 embedder）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from research_agent.cache.semantic_cache import (
    SemanticHit,
    SemanticKnowledgeCache,
    is_cacheable_query,
    normalize_query,
    reset_semantic_cache_for_tests,
)

SEED = Path(__file__).resolve().parents[2] / "src/research_agent/cache/seed/static_knowledge.json"


@pytest.fixture(autouse=True)
def _enable_semantic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "true")
    reset_semantic_cache_for_tests()
    yield
    reset_semantic_cache_for_tests()


def test_normalize_query_strips_punct_and_space() -> None:
    assert normalize_query(" 什么是 ROE？ ") == "什么是roe"
    assert normalize_query("ROE是什么意思!") == "roe是什么意思"


@pytest.mark.parametrize(
    ("query", "ok"),
    [
        ("什么是ROE", True),
        ("市盈率怎么算", True),
        ("A股交易时间是什么时候", True),
        ("300750最近怎么样", False),  # 含代码
        ("宁德时代最新行情", False),  # 时效+行情
        ("请帮我分析一下茅台", False),  # 动态研究
        ("近5日涨幅如何", False),
        ("x", False),  # 过短
    ],
)
def test_is_cacheable_query(query: str, ok: bool) -> None:
    assert is_cacheable_query(query) is ok


def test_l0_exact_hit_without_faiss(tmp_path: Path) -> None:
    """仅注入精确索引，不触达向量模型。"""
    cache = SemanticKnowledgeCache(
        db_dir=tmp_path,
        seed_path=SEED,
        enabled=True,
        similarity_threshold=0.99,
    )
    cache._seed_meta = {"version": "1", "locale": "zh-CN", "prompt_version": "v1"}
    cache._exact_index = {
        normalize_query("什么是ROE"): SemanticHit(
            answer="ROE 解释",
            cache_domain="glossary",
            score=1.0,
            matched_question="什么是ROE",
            version="1",
            locale="zh-CN",
            exact=True,
        )
    }
    cache._ready = True
    cache._store = None

    hit = cache.lookup("什么是 ROE？")
    assert hit is not None
    assert hit.exact is True
    assert hit.answer == "ROE 解释"
    assert cache.hits == 1


def test_skip_dynamic_query_increments_skips(tmp_path: Path) -> None:
    cache = SemanticKnowledgeCache(db_dir=tmp_path, seed_path=SEED, enabled=True)
    cache._ready = True
    assert cache.lookup("600519最新行情") is None
    assert cache.skips >= 1


def test_disabled_cache_always_none(tmp_path: Path) -> None:
    cache = SemanticKnowledgeCache(db_dir=tmp_path, seed_path=SEED, enabled=False)
    assert cache.lookup("什么是ROE") is None


def test_semantic_search_dimension_filter(tmp_path: Path) -> None:
    """版本/locale 不匹配的向量命中应被丢弃。"""

    class _FakeDoc:
        def __init__(self, meta: dict[str, Any], content: str = "q") -> None:
            self.metadata = meta
            self.page_content = content

    class _FakeStore:
        def similarity_search_with_relevance_scores(self, query: str, k: int = 4):
            return [
                (
                    _FakeDoc(
                        {
                            "cache_domain": "glossary",
                            "answer": "旧版答案",
                            "canonical_question": "什么是ROE",
                            "version": "0",  # 故意不匹配
                            "locale": "zh-CN",
                            "prompt_version": "v1",
                        }
                    ),
                    0.99,
                ),
                (
                    _FakeDoc(
                        {
                            "cache_domain": "glossary",
                            "answer": "正确答案",
                            "canonical_question": "什么是ROE",
                            "version": "1",
                            "locale": "zh-CN",
                            "prompt_version": "v1",
                        }
                    ),
                    0.90,
                ),
            ]

    cache = SemanticKnowledgeCache(
        db_dir=tmp_path,
        seed_path=SEED,
        enabled=True,
        similarity_threshold=0.80,
    )
    cache._seed_meta = {"version": "1", "locale": "zh-CN", "prompt_version": "v1"}
    cache._exact_index = {}
    cache._store = _FakeStore()
    cache._ready = True

    hit = cache.lookup("ROE 怎么理解")
    assert hit is not None
    assert hit.answer == "正确答案"
    assert hit.exact is False
    assert hit.score == 0.90


def test_below_threshold_is_miss(tmp_path: Path) -> None:
    class _FakeDoc:
        metadata = {
            "cache_domain": "faq",
            "answer": "x",
            "canonical_question": "q",
            "version": "1",
            "locale": "zh-CN",
            "prompt_version": "v1",
        }
        page_content = "q"

    class _FakeStore:
        def similarity_search_with_relevance_scores(self, query: str, k: int = 4):
            return [(_FakeDoc(), 0.50)]

    cache = SemanticKnowledgeCache(
        db_dir=tmp_path,
        seed_path=SEED,
        enabled=True,
        similarity_threshold=0.82,
    )
    cache._seed_meta = {"version": "1", "locale": "zh-CN", "prompt_version": "v1"}
    cache._exact_index = {}
    cache._store = _FakeStore()
    cache._ready = True

    assert cache.lookup("随机问题xyz") is None
    assert cache.misses == 1


@pytest.mark.slow
def test_seed_rebuild_and_l0_from_disk(tmp_path: Path) -> None:
    """加载真实 embedder 重建索引（slow）。"""
    cache = SemanticKnowledgeCache(
        db_dir=tmp_path,
        seed_path=SEED,
        enabled=True,
        similarity_threshold=0.75,
    )
    cache.ensure_ready()
    assert cache.stats()["exact_keys"] > 0
    hit = cache.lookup("ROE是什么意思")
    assert hit is not None
    assert hit.cache_domain == "glossary"
    assert "ROE" in hit.answer or "净资产" in hit.answer
