"""MCP (Model Context Protocol) tool servers.

Each server exposes tools via the MCP protocol, enabling protocol-
enforced tool calling instead of prompt-driven tool calling. Agents
discover and invoke these tools through
``langchain_mcp_adapters.client.MultiServerMCPClient``.

Public API of this package
==========================
The active servers are lazy-imported on first attribute access. To use
an active server, do::

    from research_agent.mcp_servers import code_server  # or echo_server

Active servers
--------------
``echo_server``
    Deterministic upper-case / length tools. Used as a smoke-test
    server for the MCP plumbing itself; not wired into the research
    pipeline.

``code_server``
    Sandboxed Python execution. Wired into the ``coder_expert``
    specialist of the minimal and research supervisors. The workhorse
    that runs LLM-generated pandas / numpy snippets over akshare
    DataFrames for derived-metric computation.

``fin_data_server``
    Real A-share market and fundamentals data via ``akshare``. Five
    tools: ``get_stock_basic_info``, ``get_stock_price_history``,
    ``get_financial_abstract``, ``get_financial_indicators``,
    ``search_stock_by_name``. The two price/quote tools cascade 东方
    财富 → 雪球 / 新浪 to survive upstream outages. Wired into the
    ``data_expert`` specialist of the research supervisor.

``pdf_report_server``
    Disclosure / research-report PDFs from 巨潮资讯. Four tools:
    ``search_announcements``, ``download_pdf`` (hashed on-disk cache
    under ``./data/pdf_cache/``), ``parse_pdf_pages`` (bounded 20-page
    window per call), ``extract_pdf_metadata``. Feeds the
    ``report_expert`` specialist.

``news_server``
    A-share news / sentiment via 东方财富 / 财联社 / 百度财经 / 雪球.
    Five tools: ``get_stock_news``, ``get_market_telegraph``,
    ``get_hot_keywords``, ``get_economic_news``,
    ``get_xueqiu_discussion_hot_rank`` (雪球讨论热度个股榜 via
    ``stock_hot_tweet_xq``). Feeds the ``news_expert`` specialist.

``knowledge_server``
    User-uploaded PDF library with hybrid retrieval (FAISS + BM25 +
    cross-encoder rerank) and corrective-RAG quality signals. Four
    tools: ``ingest_pdf``, ``search`` (returns ``quality`` ∈
    ``{high, medium, low}`` + per-hit ``rerank_score`` so the agent
    can decide to rewrite + retry), ``list_collections``,
    ``delete_collection``. Persists to ``./data/knowledge_db/`` via
    FAISS. Wired into the ``knowledge_expert`` specialist.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

ACTIVE_SERVERS: tuple[str, ...] = (
    "echo_server",
    "code_server",
    "fin_data_server",
    "pdf_report_server",
    "news_server",
    "knowledge_server",
)
"""Submodule names that are part of this package's public API."""


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
        news_server,
        pdf_report_server,
    )


__all__ = [
    "ACTIVE_SERVERS",
    "code_server",
    "echo_server",
    "fin_data_server",
    "knowledge_server",
    "news_server",
    "pdf_report_server",
]
