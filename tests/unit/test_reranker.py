"""Unit tests for the cross-encoder reranker.

Two tiers:

1. **Fast tier** (default): mock the underlying CrossEncoder so we
   exercise the reranker's ordering / fallback logic without
   downloading the ~280 MB ``BAAI/bge-reranker-base`` weights.
2. **Slow tier** (``-m slow``): one real round-trip against the
   actual model. Skipped by default; opt in for full validation.
"""

from __future__ import annotations

from typing import Any

import pytest

from research_agent.rag import reranker as reranker_module
from research_agent.rag.reranker import CrossEncoderReranker


class _FakeCrossEncoder:
    """Predictable stand-in for ``sentence_transformers.CrossEncoder``.

    Returns one score per (query, document) pair. The default scoring
    function rewards documents whose content shares more whitespace
    tokens with the query — enough determinism for ordering tests.
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
    """Make sure the module-level singleton never leaks across tests.

    Without this, a test that installed a fake encoder would poison
    later tests that wanted to install a different one.
    """
    monkeypatch.setattr(reranker_module, "_CROSS_ENCODER", None)


def _install_fake_encoder(
    monkeypatch: pytest.MonkeyPatch, encoder: _FakeCrossEncoder
) -> None:
    """Pin the fake encoder as the singleton for the duration of a test."""
    monkeypatch.setattr(reranker_module, "_CROSS_ENCODER", encoder)


# ---------------------------------------------------------------------
# Trivial cases — short-circuit paths must not touch the model
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
        # Single-doc path must short-circuit BEFORE touching the model:
        # install a fake that would crash if predict() ran.
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
# Ordering — the central guarantee
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
        # Highest-overlap doc must come first, then medium, then zero.
        assert contents[0] == "carbon neutrality scope two emissions"
        assert contents[1] == "carbon neutrality"
        assert contents[-1] == "totally unrelated content"
        # rerank_score is attached, monotonic non-increasing.
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
        # Caller should be able to look up metadata after rerank.
        assert out[0]["metadata"]["source"] == "a.pdf"
        assert out[0]["metadata"]["page"] == 7


# ---------------------------------------------------------------------
# Fallback — predict() failure must NOT break the caller
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
        # Order preserved verbatim.
        assert [d["content"] for d in out] == ["doc one", "doc two", "doc three"]
        # rerank_score field is present-but-None on every doc — the
        # response shape must stay stable for downstream code.
        assert all(d["rerank_score"] is None for d in out)


# ---------------------------------------------------------------------
# max_pairs ceiling — items past the cap stay (with score=None)
# ---------------------------------------------------------------------


class TestMaxPairs:
    @pytest.mark.asyncio
    async def test_only_first_max_pairs_are_scored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_encoder(monkeypatch, _FakeCrossEncoder())
        docs = [{"content": f"doc {i}"} for i in range(10)]
        # Cap at 4 — the remaining 6 must NOT lose their slot but
        # their rerank_score should be None and they should sort last.
        ranker = CrossEncoderReranker(max_pairs=4)
        out = await ranker.rerank("doc 0", docs)
        assert len(out) == 10
        # The first ``max_pairs`` items (0..3) all share the "doc"
        # token with the query so they get a positive score; doc 0
        # wins the tiebreak by also matching "0".
        assert out[0]["content"] == "doc 0"
        # The unranked items (5..9) settle at the tail with None.
        unscored = [d for d in out if d["rerank_score"] is None]
        assert len(unscored) == 6
        unscored_contents = {d["content"] for d in unscored}
        assert unscored_contents == {f"doc {i}" for i in range(4, 10)}


# ---------------------------------------------------------------------
# knowledge_server._maybe_rerank — env-var gate + fallback contract
# ---------------------------------------------------------------------


class TestMaybeRerankIntegration:
    """The ``_maybe_rerank`` adapter inside ``knowledge_server`` is the
    seam between the FAISS / BM25 retriever and the cross-encoder.
    These tests pin its three invariants:

      1. ``KNOWLEDGE_RERANKER_ENABLED=0`` short-circuits with stable
         response shape (``rerank_score: None``).
      2. With reranker on, items get positive ``rerank_score`` and
         the order reflects the cross-encoder.
      3. Any exception inside the reranker leaves the candidate
         list usable (no crash, ``rerank_score: None``).
    """

    @pytest.mark.asyncio
    async def test_env_var_off_short_circuits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from research_agent.mcp_servers import knowledge_server as ks

        monkeypatch.setenv("KNOWLEDGE_RERANKER_ENABLED", "0")
        # Even if a singleton happened to be installed, the env-var
        # branch must run first and never touch it.
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
        # Pre-install a fake reranker on the module-level slot so we
        # don't load the real model.
        fake_inner = _FakeCrossEncoder()
        monkeypatch.setattr(reranker_module, "_CROSS_ENCODER", fake_inner)
        # Force a fresh lazy-init of the wrapper so it picks up our
        # injected encoder rather than any cached instance.
        ks._RERANKER = None

        candidates = [
            {"content": "carbon neutrality scope two", "rrf_score": 0.6},
            {"content": "totally unrelated", "rrf_score": 0.9},
        ]
        out = await ks._maybe_rerank("carbon neutrality", candidates)
        # Reranker promotes the carbon-neutrality doc DESPITE its
        # lower RRF score — that promotion is the whole point.
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
        # Order preserved, scores set to None — exactly what the
        # ``_search`` consumer counts on.
        assert [d["content"] for d in out] == ["first", "second"]
        assert all(d["rerank_score"] is None for d in out)


# ---------------------------------------------------------------------
# Slow tier — real model
# ---------------------------------------------------------------------


@pytest.mark.slow
class TestRealModel:
    """One real round-trip against ``BAAI/bge-reranker-base``.

    First run downloads ~280 MB into ``~/.cache/huggingface``;
    subsequent runs are sub-second on warm cache. Run with
    ``pytest -m slow``.
    """

    @pytest.mark.asyncio
    async def test_real_reranker_promotes_relevant_doc(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force-reset any module-level state so this test exercises
        # the actual ``_get_cross_encoder`` lazy-load path.
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
        # The carbon-neutrality doc must rank #1 — if it doesn't,
        # something is structurally wrong with our integration
        # (model name, max_pairs, scoring direction).
        assert "scope" in out[0]["content"].lower()
        assert all(d["rerank_score"] is not None for d in out)
