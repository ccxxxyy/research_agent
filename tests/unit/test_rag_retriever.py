"""rag.retriever 单元测试 — BM25Index + hybrid_rrf_fuse。"""

from __future__ import annotations

from research_agent.rag.retriever import BM25Index, hybrid_rrf_fuse


# ======================================================================
# BM25Index 测试
# ======================================================================
class TestBM25Index:
    def _sample_docs(self) -> list[dict]:
        return [
            {"content": "宁德时代 2023 年报 电池出货量创历史新高", "metadata": {"source": "ndt.pdf", "page": 1}},
            {"content": "比亚迪 新能源汽车 销量超预期 毛利率提升", "metadata": {"source": "byd.pdf", "page": 1}},
            {"content": "锂电池 上游材料 碳酸锂 价格波动 分析", "metadata": {"source": "lithium.pdf", "page": 3}},
        ]

    def test_basic_search_returns_ranked_results(self):
        docs = self._sample_docs()
        idx = BM25Index(docs)
        results = idx.search("宁德时代 电池", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        first_idx, first_score = results[0]
        assert first_idx == 0
        assert first_score > 0

    def test_empty_corpus_returns_empty(self):
        idx = BM25Index([])
        assert idx.search("anything", top_k=5) == []

    def test_empty_query_returns_empty(self):
        idx = BM25Index(self._sample_docs())
        assert idx.search("", top_k=5) == []
        assert idx.search("   ", top_k=5) == []

    def test_top_k_limits_output(self):
        docs = self._sample_docs()
        idx = BM25Index(docs)
        results = idx.search("电池", top_k=1)
        assert len(results) == 1

    def test_scores_sorted_descending(self):
        docs = self._sample_docs()
        idx = BM25Index(docs)
        results = idx.search("电池 材料", top_k=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


# ======================================================================
# hybrid_rrf_fuse 测试
# ======================================================================
class TestHybridRRFFuse:
    def test_vector_only(self):
        vec_hits = [
            ({"content": "A", "metadata": {"source": "a.pdf", "page": 1}}, 0.9),
            ({"content": "B", "metadata": {"source": "b.pdf", "page": 2}}, 0.7),
        ]
        result = hybrid_rrf_fuse(vec_hits, [])
        assert len(result) == 2
        assert result[0]["content"] == "A"
        assert result[0]["rrf_score"] > result[1]["rrf_score"]
        assert result[0]["bm25_rank"] is None

    def test_bm25_only(self):
        bm25_hits = [
            (0, 5.2, {"content": "X", "metadata": {"source": "x.pdf", "page": 1}}),
            (1, 3.1, {"content": "Y", "metadata": {"source": "y.pdf", "page": 1}}),
        ]
        result = hybrid_rrf_fuse([], bm25_hits)
        assert len(result) == 2
        assert result[0]["content"] == "X"
        assert result[0]["vector_rank"] is None

    def test_deduplication(self):
        doc = {"content": "Same content here", "metadata": {"source": "s.pdf", "page": 1}}
        vec_hits = [(doc, 0.8)]
        bm25_hits = [(0, 4.0, doc)]
        result = hybrid_rrf_fuse(vec_hits, bm25_hits)
        assert len(result) == 1
        assert result[0]["vector_score"] == 0.8
        assert result[0]["bm25_score"] == 4.0
        assert result[0]["rrf_score"] > 0

    def test_fusion_order_respects_weights(self):
        vec_hits = [
            ({"content": "Vec1", "metadata": {"source": "a.pdf", "page": 1}}, 0.95),
        ]
        bm25_hits = [
            (0, 8.0, {"content": "BM1", "metadata": {"source": "b.pdf", "page": 1}}),
        ]
        result = hybrid_rrf_fuse(
            vec_hits, bm25_hits, vector_weight=0.9, bm25_weight=0.1
        )
        assert result[0]["content"] == "Vec1"

    def test_empty_inputs(self):
        assert hybrid_rrf_fuse([], []) == []


# ======================================================================
# 集成风格测试：BM25Index 结果输入 hybrid_rrf_fuse
# ======================================================================
class TestBM25ToFusion:
    def test_end_to_end_pipeline(self):
        docs = [
            {"content": "Financial report Q3 revenue growth", "metadata": {"source": "q3.pdf", "page": 1}},
            {"content": "Annual ESG sustainability report", "metadata": {"source": "esg.pdf", "page": 1}},
            {"content": "Revenue and profit analysis for Q3", "metadata": {"source": "q3.pdf", "page": 5}},
        ]
        bm25 = BM25Index(docs)
        bm25_raw = bm25.search("Q3 revenue", top_k=3)
        bm25_hits = [(idx, score, docs[idx]) for idx, score in bm25_raw]

        vec_hits = [
            (docs[0], 0.85),
            (docs[2], 0.72),
        ]

        fused = hybrid_rrf_fuse(vec_hits, bm25_hits)
        assert len(fused) >= 2
        assert fused[0]["rrf_score"] >= fused[-1]["rrf_score"]
