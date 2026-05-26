"""检索质量评分器 — 用于 Corrective RAG 的三档分类器。

从 ``knowledge_server._classify_quality`` 中提取，使评分逻辑可独立导入和测试。

Agent 使用与搜索结果一同返回的 ``quality`` 标签来决定是否使用重写后的查询重新发起搜索：

* ``"high"``   — 首条命中明确相关，直接回答。
* ``"medium"`` — 信号混杂，回答但提示结果可能不完整。
* ``"low"``    — 首条命中质量较弱，重写查询并重试。
"""

from __future__ import annotations


class RetrievalGrader:
    """将混合检索质量分为 high / medium / low 三档。

    阈值基于 ``BAAI/bge-small-zh-v1.5``（启用 ``normalize_embeddings=True``）的归一化余弦相似度进行校准。由嵌入模型决定。

    Parameters
    ----------
    high_threshold:
        达到 ``"high"`` 质量所需的最低 ``top_score``。
    medium_threshold:
        达到 ``"medium"`` 质量所需的最低 ``top_score``。此外，``mean_score`` 还必须超过 ``medium_threshold * 0.6``，以防止单个偶然命中掩盖整体较弱的结果集。
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
        """返回 ``"high"``、``"medium"`` 或 ``"low"``。"""
        if top_score >= self.high_threshold and unique_sources >= 1:
            return "high"
        if (
            top_score >= self.medium_threshold
            and mean_score >= self.medium_threshold * 0.6
        ):
            return "medium"
        return "low"


__all__ = ["RetrievalGrader"]
