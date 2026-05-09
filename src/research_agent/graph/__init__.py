"""LangGraph orchestration — supervisor apps for the API.

- :func:`build_minimal_supervisor` — toy specialists + optional MCP coder
  (``/api/supervisor`` minimal path).
- :func:`build_research_supervisor` — financial research team
  (``data_expert``, ``report_expert``, ``coder_expert``, ``news_expert``,
  ``knowledge_expert``) with optional tool subsets.

**Removed:** Phase-3 ``build_research_graph`` (Chroma + node-level retrieve /
grade / rewrite). Use ``research_supervisor`` + ``knowledge_server`` instead.
"""

from research_agent.graph.minimal_supervisor import build_minimal_supervisor
from research_agent.graph.research_supervisor import build_research_supervisor

__all__ = [
    "build_minimal_supervisor",
    "build_research_supervisor",
]
