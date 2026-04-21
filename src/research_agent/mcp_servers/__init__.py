"""MCP (Model Context Protocol) tool servers.

Each server exposes tools via the MCP protocol, enabling protocol-
enforced tool calling instead of prompt-driven tool calling. Agents
discover and invoke these tools through
``langchain_mcp_adapters.client.MultiServerMCPClient``.

Public API of this package
==========================
Only the *active* servers are imported at package level. To use
an active server, do::

    from research_agent.mcp_servers import code_server  # or echo_server

Deprecated servers (see below) are deliberately NOT imported here.
If code ever needs to touch one (it shouldn't), it must spell out the
full module path, e.g.
``import research_agent.mcp_servers.search_server``. The missing
package-level alias is the *signal* that those servers are off-limits
for new code.

Active servers (wired into Agents as of Phase 3+)
-------------------------------------------------
``echo_server``
    Deterministic upper-case / length tools. Used as a smoke-test
    server for the MCP plumbing itself; not wired into the research
    pipeline.

``code_server``
    Sandboxed Python execution. Wired into the ``coder_expert``
    specialist under the minimal supervisor. In Phase 4 it will be the
    workhorse that runs LLM-generated pandas / numpy snippets over
    akshare dataframes for financial analysis.

Deprecated servers (NOT re-exported)
------------------------------------
``search_server``
    Generic DuckDuckGo/Tavily wrapper → replaced in Phase 4 by a
    finance-specific ``news_server`` (东方财富 RSS + 雪球讨论).

``document_server``
    Generic PDF / Markdown parser → replaced in Phase 4 by
    ``pdf_report_server`` (巨潮资讯 research-report PDFs with page-
    level citation metadata).

``vectordb_server``
    Generic Chroma wrapper that returns synthetic data → removed in
    favor of directly using
    ``research_agent.rag.retriever.HybridRetriever`` inside the
    LangGraph retriever node.

The files are retained only so git history references still resolve;
they will be deleted in Phase 4.
"""

from __future__ import annotations

from research_agent.mcp_servers import code_server, echo_server

ACTIVE_SERVERS: tuple[str, ...] = ("echo_server", "code_server")
"""Submodule names that are part of this package's public API."""

DEPRECATED_SERVERS: tuple[str, ...] = (
    "search_server",
    "document_server",
    "vectordb_server",
)
"""Phase-0 placeholder submodules. Not re-exported from the package;
do NOT import these into new code. Will be removed in Phase 4."""

__all__ = [
    "ACTIVE_SERVERS",
    "DEPRECATED_SERVERS",
    "code_server",
    "echo_server",
]
