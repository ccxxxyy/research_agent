"""cross-encoder reranker 的单元测试。

两个层级：

1. 快速层（默认）：mock 底层 CrossEncoder，以便在不下载 约 280 MB ``BAAI/bge-reranker-base`` 权重的情况下测试 reranker 的排序/降级逻辑。
2. 慢速层（``-m slow``）：针对真实模型的一次实际往返。默认跳过；选择性加入以进行完整验证。
"""

from __future__ import annotations

from typing import Any

import pytest

from research_agent.rag import reranker as reranker_module
from research_agent.rag.reranker import CrossEncoderReranker


class _FakeCrossEncoder:
    """``sentence_transformers.CrossEncoder`` 的可预测替身。

    每个 (query, document) 对返回一个分数。默认评分函数奖励与查询共享更多空格分词的文档内容 — 足够的确定性用于排序测试。
    """

    def __init__(self, score_fn=None) -> None:
        self._score_fn = score_fn or self._default_score

    @staticmethod
    def _default_score(query: str, content: str) -> float:
        q_tokens = set(query.lower().split())
        c_tokens = set(content.lower().split())
        return float(len(q_tokens & c_tokens))

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self._score_fn(q, d) for q, d in pairs]


@pytest.fixture(autouse=True)
def _reset_reranker_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """确保模块级单例不会在测试间泄漏。

    如果没有此 fixture，安装了伪造编码器的测试会污染后续想要安装不同编码器的测试。
    """
    monkeypatch.setattr(reranker_module, "_CROSS_ENCODER", None)


def _install_fake_encoder(
    monkeypatch: pytest.MonkeyPatch, encoder: _FakeCrossEncoder
) -> None:
    """在测试持续期间将伪造编码器固定为单例。"""
    monkeypatch.setattr(reranker_module, "_CROSS_ENCODER", encoder)


# ---------------------------------------------------------------------
# 简单情况 — 短路路径不得接触模型
# ---------------------------------------------------------------------


class TestTrivialCases:
    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self) -> None:
        out = await CrossEncoderReranker().rerank("anything", [])
        assert out == []

    @pytest.mark.asyncio
    async def test_single_doc_returns_single_doc_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 单文档路径必须在接触模型之前短路：安装一个如果 predict() 执行就会崩溃的伪造对象。
        class _Boom:
            def predict(self, _pairs: Any) -> list[float]:
                raise AssertionError("predict() should not be called for n=1")

        _install_fake_encoder(monkeypatch, _Boom())  # type: ignore[arg-type]
        docs = [{"content": "lonely doc"}]
        out = await CrossEncoderReranker().rerank("query", docs)
        assert len(out) == 1
        assert out[0]["content"] == "lonely doc"
        assert out[0]["rerank_score"] is None


# ---------------------------------------------------------------------
# 排序 — 核心保证
# ---------------------------------------------------------------------


class TestOrdering:
    @pytest.mark.asyncio
    async def test_reranks_docs_by_predicted_score(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_encoder(monkeypatch, _FakeCrossEncoder())
        docs = [
            {"content": "totally unrelated content"},  # 0 token overlap
            {"content": "carbon neutrality scope two emissions"},  # 3 overlap
            {"content": "carbon neutrality"},  # 2 overlap
        ]
        out = await CrossEncoderReranker().rerank(
            "carbon neutrality scope", docs
        )
        contents = [d["content"] for d in out]
        # 重叠度最高的文档必须排第一，然后是中等，最后是零。
        assert contents[0] == "carbon neutrality scope two emissions"
        assert contents[1] == "carbon neutrality"
        assert contents[-1] == "totally unrelated content"
        # rerank_score 已附加，单调非递增。
        scores = [d["rerank_score"] for d in out]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_metadata_is_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_encoder(monkeypatch, _FakeCrossEncoder())
        docs = [
            {"content": "alpha beta", "metadata": {"source": "a.pdf", "page": 7}},
            {"content": "alpha", "metadata": {"source": "b.pdf", "page": 3}},
        ]
        out = await CrossEncoderReranker().rerank("alpha beta", docs)
        # 调用者应能在重排序后查找元数据。
        assert out[0]["metadata"]["source"] == "a.pdf"
        assert out[0]["metadata"]["page"] == 7


# ---------------------------------------------------------------------
# 降级 — predict() 失败不得中断调用者
# ---------------------------------------------------------------------


class TestFallback:
    @pytest.mark.asyncio
    async def test_predict_exception_returns_input_order_with_none_scores(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Boom:
            def predict(self, _pairs: Any) -> list[float]:
                raise RuntimeError("simulated CUDA OOM / model load failure")

        _install_fake_encoder(monkeypatch, _Boom())  # type: ignore[arg-type]
        docs = [
            {"content": "doc one"},
            {"content": "doc two"},
            {"content": "doc three"},
        ]
        out = await CrossEncoderReranker().rerank("query", docs)
        # 顺序原样保留。
        assert [d["content"] for d in out] == ["doc one", "doc two", "doc three"]
        # 每个文档的 rerank_score 字段存在但为 None ——响应形状必须对下游代码保持稳定。
        assert all(d["rerank_score"] is None for d in out)


# ---------------------------------------------------------------------
# max_pairs 上限 — 超出上限的项保留（score=None）
# ---------------------------------------------------------------------


class TestMaxPairs:
    @pytest.mark.asyncio
    async def test_only_first_max_pairs_are_scored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_encoder(monkeypatch, _FakeCrossEncoder())
        docs = [{"content": f"doc {i}"} for i in range(10)]
        # 上限为 4 — 剩余 6 个不得丢失位置，但其 rerank_score应为 None 且排在最后。
        ranker = CrossEncoderReranker(max_pairs=4)
        out = await ranker.rerank("doc 0", docs)
        assert len(out) == 10
        # 前 ``max_pairs`` 个项（0..3）都与查询共享 "doc" 词元，因此获得正分数；doc 0 通过同时匹配 "0" 赢得平局。
        assert out[0]["content"] == "doc 0"
        # 未排序的项（5..9）以 None 分数落在尾部。
        unscored = [d for d in out if d["rerank_score"] is None]
        assert len(unscored) == 6
        unscored_contents = {d["content"] for d in unscored}
        assert unscored_contents == {f"doc {i}" for i in range(4, 10)}


# ---------------------------------------------------------------------
# knowledge_server._maybe_rerank — 环境变量门控 + 降级契约
# ---------------------------------------------------------------------


class TestMaybeRerankIntegration:
    """``knowledge_server`` 内的 ``_maybe_rerank`` 适配器是 FAISS / BM25
    检索器与 cross-encoder 之间的接缝。这些测试固定其三个不变量：

      1. ``KNOWLEDGE_RERANKER_ENABLED=0`` 以稳定的响应形状
         （``rerank_score: None``）短路。
      2. 启用 reranker 时，项获得正的 ``rerank_score``，
         且顺序反映 cross-encoder 的结果。
      3. reranker 内的任何异常都使候选列表保持可用
         （不崩溃，``rerank_score: None``）。
    """

    @pytest.mark.asyncio
    async def test_env_var_off_short_circuits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from research_agent.mcp_servers import knowledge_server as ks

        monkeypatch.setenv("KNOWLEDGE_RERANKER_ENABLED", "0")
        # 即使已安装了单例，环境变量分支也必须先运行且不接触它。
        ks._RERANKER = object()  # type: ignore[assignment]

        candidates = [
            {"content": "alpha", "rrf_score": 0.9},
            {"content": "beta", "rrf_score": 0.7},
        ]
        out = await ks._maybe_rerank("query", candidates)
        assert [d["content"] for d in out] == ["alpha", "beta"]
        assert all(d["rerank_score"] is None for d in out)

    @pytest.mark.asyncio
    async def test_env_var_on_reranks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from research_agent.mcp_servers import knowledge_server as ks

        monkeypatch.setenv("KNOWLEDGE_RERANKER_ENABLED", "1")
        # 在模块级槽位预装伪造 reranker，避免加载真实模型。
        fake_inner = _FakeCrossEncoder()
        monkeypatch.setattr(reranker_module, "_CROSS_ENCODER", fake_inner)
        # 强制包装器的新惰性初始化，以拾取我们注入的编码器 而非任何缓存实例。
        ks._RERANKER = None

        candidates = [
            {"content": "carbon neutrality scope two", "rrf_score": 0.6},
            {"content": "totally unrelated", "rrf_score": 0.9},
        ]
        out = await ks._maybe_rerank("carbon neutrality", candidates)
        # reranker 将碳中和文档提升到前面，尽管其 RRF 分数更低 — 这种提升正是其意义所在。
        assert "carbon" in out[0]["content"]
        assert all(d["rerank_score"] is not None for d in out)

    @pytest.mark.asyncio
    async def test_predict_exception_returns_candidates_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from research_agent.mcp_servers import knowledge_server as ks

        monkeypatch.setenv("KNOWLEDGE_RERANKER_ENABLED", "1")

        class _Boom:
            def predict(self, _pairs: Any) -> list[float]:
                raise RuntimeError("model went away")

        monkeypatch.setattr(reranker_module, "_CROSS_ENCODER", _Boom())
        ks._RERANKER = None

        candidates = [
            {"content": "first"},
            {"content": "second"},
        ]
        out = await ks._maybe_rerank("query", candidates)
        # 顺序保留，分数设为 None — 正是 ``_search``消费者所依赖的行为。
        assert [d["content"] for d in out] == ["first", "second"]
        assert all(d["rerank_score"] is None for d in out)


# ---------------------------------------------------------------------
# 慢速层 — 真实模型
# ---------------------------------------------------------------------


@pytest.mark.slow
class TestRealModel:
    """针对 ``BAAI/bge-reranker-base`` 的一次真实往返。

    首次运行下载约 280 MB 到 ``~/.cache/huggingface``；后续运行在预热缓存上不到一秒。使用 ``pytest -m slow`` 运行。
    """

    @pytest.mark.asyncio
    async def test_real_reranker_promotes_relevant_doc(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 强制重置所有模块级状态，以便此测试执行实际的 ``_get_cross_encoder`` 惰性加载路径。
        monkeypatch.setattr(reranker_module, "_CROSS_ENCODER", None)
        docs = [
            {"content": "Annual dividend payout is 0.12 USD per share."},
            {
                "content": (
                    "We commit to net-zero scope 1 and scope 2 emissions "
                    "by 2030, with a 50% reduction milestone in 2027."
                )
            },
            {"content": "Quarterly board meeting agenda placeholder."},
        ]
        out = await CrossEncoderReranker().rerank(
            "carbon neutrality 2030 target", docs
        )
        assert len(out) == 3
        # 碳中和文档必须排第 1 — 如果没有，说明集成 存在结构性问题（模型名称、max_pairs、评分方向）。
        assert "scope" in out[0]["content"].lower()
        assert all(d["rerank_score"] is not None for d in out)
