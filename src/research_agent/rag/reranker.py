"""Cross-encoder reranker — local, fast, no API calls.

Why a cross-encoder, not an LLM?
--------------------------------
The first iteration of this module shipped an ``LLMReranker`` that
asked an LLM to score every (query, document) pair via JSON. That
worked, but had three problems:

* **Latency** — one LLM round-trip per document. With ``top_k=5`` and
  even a fast LIGHT-tier model this is 3-10 s of wall-clock per
  search call, which compounds with the corrective-RAG retry loop.
* **Cost** — every search burns 5+ LIGHT-tier prompts, paid per
  request.
* **Drift** — the score JSON is parsed from a chat completion; the
  model sometimes returns prose, ``score: "high"``, or a numeric
  string with units, all of which the parser had to babysit.

A small **cross-encoder** (here ``BAAI/bge-reranker-base``, ~280 MB)
solves all three: it runs locally on CPU at ~50 ms / pair, costs
nothing per request, and returns a real-valued logit. The bi-encoder
embeddings used for the initial FAISS retrieval and a cross-encoder
used for reranking are complementary by design — the bi-encoder is
fast enough to score thousands of candidates, the cross-encoder is
accurate enough to re-order the top few.

Where this is wired
-------------------
The single production caller is
``research_agent.mcp_servers.knowledge_server._search()``. That
function over-fetches from RRF (typically ``top_k * 3``), feeds the
candidates here, and trims the reranked output to ``top_k``. The
``rerank_score`` is attached to each result so downstream agents
(the ``knowledge_expert`` corrective-RAG loop in particular) can
inspect the cross-encoder's verdict alongside the bi-encoder vector
score.

Optional / failure-tolerant
---------------------------
Reranking is gated by the ``KNOWLEDGE_RERANKER_ENABLED`` env var
(default ``"1"``). When disabled — or when the model fails to load
on a given host — the consumer falls back to RRF order. ``search()``
never raises because reranking went wrong; the worst case is "you
get RRF order plus a one-line warning in the log".
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from loguru import logger

_MODELSCOPE_CACHE_PATH = os.path.expanduser(
    "~/.cache/modelscope/hub/models/BAAI/bge-reranker-base"
)
"""Pre-computed path to the ModelScope cache directory.

On Chinese networks HuggingFace Hub's Xet S3 backend is often
unreachable. If the weights were pre-downloaded (e.g. via
``modelscope.snapshot_download('BAAI/bge-reranker-base')`` after
``uv sync --extra modelscope``) they land here. We prefer this path
when it exists and contains the safetensors file, avoiding any
network call at model-load time.
"""


def _resolve_default_model() -> str:
    """Pick the best available model path at import time.

    Priority:
      1. Explicit ``KNOWLEDGE_RERANKER_MODEL`` env var (user override).
      2. ModelScope local cache (fast, no network).
      3. HuggingFace model id (needs network or HF cache hit).
    """
    explicit = os.environ.get("KNOWLEDGE_RERANKER_MODEL", "").strip()
    if explicit:
        return explicit
    if os.path.isfile(os.path.join(_MODELSCOPE_CACHE_PATH, "model.safetensors")):
        return _MODELSCOPE_CACHE_PATH
    return "BAAI/bge-reranker-base"


DEFAULT_RERANKER_MODEL = _resolve_default_model()
"""Default cross-encoder model path.

Resolved at import time via :func:`_resolve_default_model`:

* If ``KNOWLEDGE_RERANKER_MODEL`` is set, that wins.
* Otherwise, if ``~/.cache/modelscope/hub/models/BAAI/
  bge-reranker-base/model.safetensors`` exists (pre-downloaded with
  the optional ``modelscope`` extra), we use the local path — zero
  network IO.
* Otherwise falls back to the HuggingFace model id
  ``BAAI/bge-reranker-base`` (needs HF Hub access or a warm
  ``~/.cache/huggingface`` cache).

``BAAI/bge-reranker-base`` is bilingual (zh + en), ~1 GB on disk,
and runs at ~50 ms per pair on a modern CPU.
"""

# Module-level cache: loading a CrossEncoder spins up a tokenizer and
# pulls a few hundred MB of weights into RAM. We keep one instance
# alive for the lifetime of the process and reuse it across all
# search calls.
_CROSS_ENCODER: Any | None = None


def _get_cross_encoder(model_name: str | None = None) -> Any:
    """Return the singleton CrossEncoder, building it on first call.

    The ``sentence_transformers`` import is deferred so a process
    that never asks for reranking (e.g. a unit test of unrelated
    helpers) doesn't pay the ~1 s import cost. First call additionally
    pays the model-weight load — ~3 s warm cache, longer on cold.
    """
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        from sentence_transformers import CrossEncoder

        name = model_name or DEFAULT_RERANKER_MODEL
        logger.info("Loading cross-encoder reranker: {}", name)
        _CROSS_ENCODER = CrossEncoder(name, device="cpu")
    return _CROSS_ENCODER


class CrossEncoderReranker:
    """Local cross-encoder reranker.

    Operates on the dict shape used by
    :mod:`research_agent.mcp_servers.knowledge_server` so the
    integration is drop-in: each input dict needs a ``"content"``
    key (the chunk text); everything else is preserved verbatim and
    a new ``"rerank_score"`` field is added. The output is the same
    list, sorted descending by ``rerank_score``.

    Args:
        model_name: HuggingFace model id. ``None`` (default) uses
            :data:`DEFAULT_RERANKER_MODEL`, which itself respects
            the ``KNOWLEDGE_RERANKER_MODEL`` env var.
        max_pairs: Hard ceiling on how many (query, document) pairs
            we send through the cross-encoder per call. Protects
            against accidentally feeding 1000 candidates and pegging
            the CPU. Anything beyond ``max_pairs`` is left unranked
            at the tail of the output.

    Example::

        reranker = CrossEncoderReranker()
        ranked = await reranker.rerank(
            query="2030 carbon neutrality",
            documents=fused_rrf_hits,  # list[dict] with "content"
        )
        top_5 = ranked[:5]
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        max_pairs: int = 64,
    ) -> None:
        self._model_name = model_name
        # Bound the per-call cost. 64 pairs ≈ 3 s on CPU with the
        # base reranker — already well above any sensible top_k.
        self._max_pairs = max(1, int(max_pairs))

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Re-score ``documents`` against ``query`` and sort descending.

        Trivial cases (0 or 1 document) short-circuit without loading
        the model — useful in offline test paths and on cold starts
        where the candidate list is empty.

        The function never raises: any exception from the model
        layer is caught and surfaces as the input order with a
        warning log, so the caller can keep its happy path.
        """
        if not documents:
            return []
        if len(documents) == 1:
            documents[0].setdefault("rerank_score", None)
            return documents

        head = documents[: self._max_pairs]
        tail = documents[self._max_pairs :]

        try:
            scores = await asyncio.to_thread(self._predict, query, head)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CrossEncoderReranker failed ({}); returning input order",
                exc,
            )
            for d in documents:
                d.setdefault("rerank_score", None)
            return documents

        for doc, score in zip(head, scores):
            doc["rerank_score"] = round(float(score), 4)

        # Items past the max_pairs ceiling never went through the
        # model. We keep them, but they sort below any reranked item
        # because their score is None. Use a sentinel float for the
        # sort so None doesn't crash the comparison.
        for d in tail:
            d.setdefault("rerank_score", None)

        def _sort_key(d: dict[str, Any]) -> float:
            score = d.get("rerank_score")
            return float(score) if score is not None else float("-inf")

        return sorted(documents, key=_sort_key, reverse=True)

    def _predict(self, query: str, documents: list[dict[str, Any]]) -> list[float]:
        """Run the cross-encoder on (query, content) pairs.

        Pulled out as a sync method so the async ``rerank`` can ship
        it through ``asyncio.to_thread`` without leaking torch import
        details into the public surface.
        """
        model = _get_cross_encoder(self._model_name)
        pairs = [(query, doc.get("content", "") or "") for doc in documents]
        raw = model.predict(pairs)
        # ``CrossEncoder.predict`` returns a numpy ndarray for batch
        # input. Convert to a plain list of floats so downstream
        # callers aren't forced to import numpy.
        return [float(s) for s in raw]


__all__ = [
    "CrossEncoderReranker",
    "DEFAULT_RERANKER_MODEL",
]
