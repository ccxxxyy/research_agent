"""模型层级定义与 Agent 到模型的映射。"""

from __future__ import annotations

from enum import Enum


class ModelTier(str, Enum):
    """按任务复杂度划分的模型路由层级。"""

    LIGHT = "light"    # 分类、提取、格式化、评分
    MEDIUM = "medium"  # 摘要、分析、评估
    HEAVY = "heavy"    # 深度推理、报告撰写、规划


class AgentName(str, Enum):
    """已注册的 Agent 标识符，用于模型路由。"""

    SUPERVISOR = "supervisor"
    RETRIEVER = "retriever"
    ANALYST = "analyst"
    REASONER = "reasoner"
    WRITER = "writer"
    RAG_GRADER = "rag_grader"
    QUERY_REWRITER = "query_rewriter"


AGENT_TIER_MAP: dict[AgentName, ModelTier] = {
    AgentName.SUPERVISOR: ModelTier.HEAVY,
    AgentName.RETRIEVER: ModelTier.LIGHT,
    AgentName.ANALYST: ModelTier.MEDIUM,
    AgentName.REASONER: ModelTier.HEAVY,
    AgentName.WRITER: ModelTier.HEAVY,
    AgentName.RAG_GRADER: ModelTier.LIGHT,
    AgentName.QUERY_REWRITER: ModelTier.LIGHT,
}

FALLBACK_CHAIN: dict[ModelTier, ModelTier] = {
    ModelTier.HEAVY: ModelTier.MEDIUM,
    ModelTier.MEDIUM: ModelTier.LIGHT,
}
