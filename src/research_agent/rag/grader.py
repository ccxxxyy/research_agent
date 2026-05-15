"""Retrieval quality grader — three-bucket classifier for corrective RAG.

Extracted from ``knowledge_server._classify_quality`` so the grading
logic is independently importable and testable.

The agent uses the ``quality`` label returned alongside search hits
to decide whether to re-issue the search with a rewritten query:

* ``"high"``   — top hit is clearly on-topic; answer directly.
* ``"medium"`` — mixed signal; answer but warn it may be partial.
* ``"low"``    — top hit is weak; rewrite and retry.
"""

from __future__ import annotations


class RetrievalGrader:
    """Classify hybrid-retrieval quality into high / medium / low.

    Thresholds are calibrated for **normalised cosine similarity**
    from ``BAAI/bge-small-zh-v1.5`` (with ``normalize_embeddings=True``).
    Swap the embedding model → recalibrate.

    Parameters
    ----------
    high_threshold:
        Minimum ``top_score`` for ``"high"`` quality.
    medium_threshold:
        Minimum ``top_score`` for ``"medium"`` quality.  The
        ``mean_score`` must additionally exceed
        ``medium_threshold * 0.6`` to prevent a single lucky hit
        from masking an otherwise weak result set.
    """

    def __init__(
        self,
        *,
        high_threshold: float = 0.65,
        medium_threshold: float = 0.40,
    ) -> None:
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    def grade(
        self,
        top_score: float,
        mean_score: float,
        unique_sources: int,
    ) -> str:
        """Return ``"high"``, ``"medium"``, or ``"low"``."""
        if top_score >= self.high_threshold and unique_sources >= 1:
            return "high"
        if (
            top_score >= self.medium_threshold
            and mean_score >= self.medium_threshold * 0.6
        ):
            return "medium"
        return "low"


__all__ = ["RetrievalGrader"]
