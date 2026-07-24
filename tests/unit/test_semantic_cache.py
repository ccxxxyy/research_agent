"""静态知识语义缓存单元测试（默认不加载真实 embedder）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from research_agent.cache.semantic_cache import (
    SemanticHit,
    SemanticKnowledgeCache,
    allowed_markets_for,
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
        ("美股交易时间是什么时候", True),
        ("什么是10-K", True),
        ("300750最近怎么样", False),  # 含代码
        ("宁德时代最新行情", False),  # 时效+行情
        ("请帮我分析一下茅台", False),  # 动态研究
        ("近5日涨幅如何", False),
        ("AAPL latest price", False),  # 美股 ticker + 英文时效
        ("What is the latest quote for MSFT", False),
        ("x", False),  # 过短
    ],
)
def test_is_cacheable_query(query: str, ok: bool) -> None:
    assert is_cacheable_query(query) is ok


def test_allowed_markets_for() -> None:
    assert allowed_markets_for("US") == frozenset({"US", "SHARED"})
    assert allowed_markets_for("CN_A") == frozenset({"CN_A", "SHARED"})
    assert allowed_markets_for("MIXED") == frozenset({"CN_A", "US", "SHARED"})
    assert allowed_markets_for(None) == frozenset({"CN_A", "US", "SHARED"})


def test_l0_exact_hit_without_faiss(tmp_path: Path) -> None:
    """仅注入精确索引，不触达向量模型。"""
    cache = SemanticKnowledgeCache(
        db_dir=tmp_path,
        seed_path=SEED,
        enabled=True,
        similarity_threshold=0.99,
    )
    cache._seed_meta = {"version": "2", "locale": "zh-CN", "prompt_version": "v1"}
    cache._exact_index = {
        normalize_query("什么是ROE"): SemanticHit(
            answer="ROE 解释",
            cache_domain="glossary",
            score=1.0,
            matched_question="什么是ROE",
            version="2",
            locale="zh-CN",
            exact=True,
            market="SHARED",
        )
    }
    cache._ready = True
    cache._store = None

    hit = cache.lookup("什么是 ROE？")
    assert hit is not None
    assert hit.exact is True
    assert hit.answer == "ROE 解释"
    assert cache.hits == 1


def test_l0_market_filter_blocks_cn_only(tmp_path: Path) -> None:
    cache = SemanticKnowledgeCache(
        db_dir=tmp_path,
        seed_path=SEED,
        enabled=True,
        similarity_threshold=0.99,
    )
    cache._seed_meta = {"version": "2", "locale": "zh-CN", "prompt_version": "v1"}
    cache._exact_index = {
        normalize_query("什么是北向资金"): SemanticHit(
            answer="北向解释",
            cache_domain="glossary",
            score=1.0,
            matched_question="什么是北向资金",
            version="2",
            locale="zh-CN",
            exact=True,
            market="CN_A",
        )
    }
    cache._ready = True
    cache._store = None

    assert cache.lookup("什么是北向资金", market="US") is None
    assert cache.misses == 1
    hit = cache.lookup("什么是北向资金", market="CN_A")
    assert hit is not None
    assert hit.market == "CN_A"


def test_l0_us_seed_hit_from_disk_exact(tmp_path: Path) -> None:
    """从真实种子构建 L0 索引（不强制 FAISS），验证 US 域条目。"""
    cache = SemanticKnowledgeCache(
        db_dir=tmp_path,
        seed_path=SEED,
        enabled=True,
        similarity_threshold=0.99,
    )
    entries = cache._load_seed()
    cache._build_exact_index(entries)
    cache._ready = True
    cache._store = None

    hit = cache.lookup("美股交易时间是什么时候", market="US")
    assert hit is not None
    assert hit.market == "US"
    assert "09:30" in hit.answer or "美东" in hit.answer

    # CN_A 偏好不应命中纯 US FAQ（L0）
    assert cache.lookup("美股交易时间是什么时候", market="CN_A") is None


def test_skip_dynamic_query_increments_skips(tmp_path: Path) -> None:
    cache = SemanticKnowledgeCache(db_dir=tmp_path, seed_path=SEED, enabled=True)
    cache._ready = True
    assert cache.lookup("600519最新行情") is None
    assert cache.skips >= 1


def test_disabled_cache_always_none(tmp_path: Path) -> None:
    cache = SemanticKnowledgeCache(db_dir=tmp_path, seed_path=SEED, enabled=False)
    assert cache.lookup("什么是ROE") is None


def test_semantic_search_dimension_filter(tmp_path: Path) -> None:
    """版本 / market 不匹配的向量命中应被丢弃。"""

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
                            "market": "SHARED",
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
                            "version": "2",
                            "locale": "zh-CN",
                            "prompt_version": "v1",
                            "market": "SHARED",
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
    cache._seed_meta = {"version": "2", "locale": "zh-CN", "prompt_version": "v1"}
    cache._exact_index = {}
    cache._store = _FakeStore()
    cache._ready = True

    hit = cache.lookup("ROE 怎么理解")
    assert hit is not None
    assert hit.answer == "正确答案"
    assert hit.exact is False
    assert hit.score == 0.90
    assert hit.market == "SHARED"


def test_semantic_search_filters_wrong_market(tmp_path: Path) -> None:
    class _FakeDoc:
        def __init__(self, meta: dict[str, Any]) -> None:
            self.metadata = meta
            self.page_content = "q"

    class _FakeStore:
        def similarity_search_with_relevance_scores(self, query: str, k: int = 4):
            return [
                (
                    _FakeDoc(
                        {
                            "cache_domain": "faq",
                            "answer": "A股时间",
                            "canonical_question": "A股交易时间",
                            "version": "2",
                            "locale": "zh-CN",
                            "prompt_version": "v1",
                            "market": "CN_A",
                        }
                    ),
                    0.95,
                )
            ]

    cache = SemanticKnowledgeCache(
        db_dir=tmp_path,
        seed_path=SEED,
        enabled=True,
        similarity_threshold=0.80,
    )
    cache._seed_meta = {"version": "2", "locale": "zh-CN", "prompt_version": "v1"}
    cache._exact_index = {}
    cache._store = _FakeStore()
    cache._ready = True

    assert cache.lookup("交易时间", market="US") is None
    hit = cache.lookup("交易时间", market="CN_A")
    assert hit is not None
    assert hit.answer == "A股时间"


def test_below_threshold_is_miss(tmp_path: Path) -> None:
    class _FakeDoc:
        metadata = {
            "cache_domain": "faq",
            "answer": "x",
            "canonical_question": "q",
            "version": "2",
            "locale": "zh-CN",
            "prompt_version": "v1",
            "market": "SHARED",
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
    cache._seed_meta = {"version": "2", "locale": "zh-CN", "prompt_version": "v1"}
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
    us_hit = cache.lookup("什么是10-K", market="US")
    assert us_hit is not None
    assert us_hit.market == "US"
