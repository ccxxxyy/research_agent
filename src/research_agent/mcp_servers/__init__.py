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

``fin_data_server`` (Phase 4.1)
    Real A-share market and fundamentals data via ``akshare``. Five
    tools: ``get_stock_basic_info``, ``get_stock_price_history``,
    ``get_financial_abstract``, ``get_financial_indicators``,
    ``search_stock_by_name``. The two price/quote tools cascade 东方
    财富 → 雪球 / 新浪 to survive upstream outages. Will be wired into
    the ``data_expert`` specialist in Phase 4.4.

``pdf_report_server`` (Phase 4.2)
    Disclosure / research-report PDFs from 巨潮资讯. Four tools:
    ``search_announcements``, ``download_pdf`` (hashed on-disk cache
    under ``./data/pdf_cache/``), ``parse_pdf_pages`` (bounded 20-page
    window per call), ``extract_pdf_metadata``. Feeds the
    ``report_expert`` specialist in Phase 4.4.

``knowledge_server`` (Phase 4.6)
    User-uploaded PDF library with hybrid retrieval (vector + BM25)
    and corrective-RAG quality signals. Four tools:
    ``ingest_pdf``, ``search`` (returns ``quality`` ∈
    {high,medium,low} so the agent can decide to rewrite + retry),
    ``list_collections``, ``delete_collection``. Persists to
    ``./data/knowledge_db/`` via Chroma. Wired into the
    ``knowledge_expert`` specialist of the research supervisor.

Deprecated servers (NOT re-exported)
------------------------------------
``search_server``
    Generic DuckDuckGo/Tavily wrapper → replaced in Phase 4 by a
    finance-specific ``news_server`` (东方财富 RSS + 雪球讨论).

``document_server``
    Generic PDF / Markdown parser → superseded by
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

import importlib
from typing import TYPE_CHECKING

ACTIVE_SERVERS: tuple[str, ...] = (
    "echo_server",
    "code_server",
    "fin_data_server",
    "pdf_report_server",
    "knowledge_server",
)
"""Submodule names that are part of this package's public API."""

DEPRECATED_SERVERS: tuple[str, ...] = (
    "search_server",
    "document_server",
    "vectordb_server",
)
"""Phase-0 placeholder submodules. Not re-exported from the package;
do NOT import these into new code. Will be removed in Phase 4."""


def __getattr__(name: str):
    """Lazy-import active submodules on first attribute access.

    Why lazy instead of ``from research_agent.mcp_servers import
    code_server``? When a subprocess is spawned via ``python -m
    research_agent.mcp_servers.code_server`` (the MCP stdio launch
    path), an eager top-level import would cause Python to import the
    module twice — once implicitly via the package ``__init__``, once
    explicitly via ``runpy`` — triggering a ``RuntimeWarning: 'X'
    found in sys.modules after import of package``. Deferring via
    PEP 562 ``__getattr__`` avoids that and keeps ``from
    research_agent.mcp_servers import code_server`` working for
    callers that want the package-level alias.
    """
    if name in ACTIVE_SERVERS:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover - helps type checkers & IDE autocomplete
    from research_agent.mcp_servers import (
        code_server,
        echo_server,
        fin_data_server,
        knowledge_server,
        pdf_report_server,
    )


__all__ = [
    "ACTIVE_SERVERS",
    "DEPRECATED_SERVERS",
    "code_server",
    "echo_server",
    "fin_data_server",
    "knowledge_server",
    "pdf_report_server",
]
