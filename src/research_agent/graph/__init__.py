"""LangGraph 编排层 — 为 API 提供 supervisor 应用。

- :func:`build_minimal_supervisor` — 简单专家组 + 可选 MCP 编程专家（``/api/supervisor`` 最小化路径）。
- :func:`build_research_supervisor` — 金融研究团队（``data_expert``、``us_data_expert``、``report_expert``、``coder_expert``、``news_expert``、 ``knowledge_expert``），支持可选工具子集。

已移除：``build_research_graph``（Chroma + 节点级检索 /评分 / 重写）。改用 ``research_supervisor`` + ``knowledge_server``。
"""

from research_agent.graph.minimal_supervisor import build_minimal_supervisor
from research_agent.graph.research_supervisor import build_research_supervisor

__all__ = [
    "build_minimal_supervisor",
    "build_research_supervisor",
]
