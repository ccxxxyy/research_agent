"""In-process LangChain tools for the knowledge-base RAG plane.

Why this module exists
----------------------
The four knowledge-base tools (``ingest_pdf``, ``search``,
``list_collections``, ``delete_collection``) are *defined* in
:mod:`research_agent.mcp_servers.knowledge_server` because that module
also acts as the canonical MCP contract — its docstrings, validation
rules, and JSON return shapes are authoritative.

At runtime, however, we deliver them as **in-process** LangChain tools
rather than through an MCP-stdio subprocess. The reason is documented
at length in ``knowledge_server.py`` (TL;DR: on Windows + Python 3.13,
fastmcp's stdio JSON-RPC writer interacts badly with the heavy import
chain pulled in by ``sentence-transformers`` / ``torch`` / ``faiss``,
and after ``ingest_pdf`` finishes its work the JSON-RPC response
silently never reaches the parent).

This module is the bridge: it takes the four ``@mcp.tool``-decorated
coroutines from ``knowledge_server`` and re-exposes them as
``langchain_core.tools.BaseTool`` instances with the **same prefixed
names** the MCP path used (``knowledge_*``). That means the
``KNOWLEDGE_EXPERT_PROMPT`` keeps working unchanged — the agent sees
the same toolbelt regardless of whether tools are delivered in-process
or out-of-process.

Why this is safe
----------------
The ``@mcp.tool()`` decorator from fastmcp 3.x registers the function
with the FastMCP instance but otherwise leaves it unchanged
(``type(ingest_pdf)`` is ``types.FunctionType``, no ``.fn`` shim).
Calling them directly therefore runs exactly the same code path the
MCP transport would have invoked, including the ``asyncio.to_thread``
hop that protects the event loop from blocking work.

Returned shapes
---------------
Each underlying coroutine returns a ``dict`` (success or
``{error, context}`` on validation failure). The LangChain tool
runtime serialises that dict to JSON for the LLM. Both the prompt and
the corrective-RAG loop already read the JSON shape, so no adapter
work is required.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from research_agent.mcp_servers.knowledge_server import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    delete_collection as _delete_collection_impl,
    ingest_pdf as _ingest_pdf_impl,
    list_collections as _list_collections_impl,
    search as _search_impl,
)


# ---------------------------------------------------------------------
# Thin async wrappers
#
# StructuredTool requires an awaitable that maps the LLM-supplied
# arguments to the underlying coroutine. We could pass the imported
# coroutines directly, but going through a wrapper buys us:
#   1. A stable place to re-document each tool with prompt-friendly
#      language (the MCP docstrings are authoritative for *us*, but
#      a couple of phrasings are tuned for an LLM consumer).
#   2. A choke-point for any future cross-cutting concern
#      (rate-limit, audit log, caching) without touching either the
#      MCP module or the agent prompt.
# ---------------------------------------------------------------------
async def _ingest_pdf(
    local_path: str,
    collection: str = "default",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    """Ingest a single local PDF into a persistent FAISS collection.

    Args:
        local_path: Filesystem path to a ``.pdf`` (typically what
            ``pdf_download_pdf`` returned). The path must already
            exist; this tool never invents files.
        collection: Target collection name (created on first use).
            Names must match ``[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]``
            and be 3–63 chars.
        chunk_size: Characters per chunk. Default 800 is tuned for
            the bge-small-zh embedder's 512-token window.
        chunk_overlap: Characters of slide between adjacent chunks.

    Returns:
        ``{collection, source, num_pages, num_chunks_added,
        total_chunks_in_collection}`` on success, or
        ``{error, context}`` on validation failure.
    """
    return await _ingest_pdf_impl(
        local_path=local_path,
        collection=collection,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


async def _search(
    query: str,
    collection: str = "default",
    top_k: int = 5,
) -> dict[str, Any]:
    """Hybrid (vector + BM25) search over an ingested collection.

    The response carries top-level ``quality`` (``high``/``medium``/
    ``low``) plus per-hit ``vector_score`` / ``bm25_score`` /
    ``rrf_score`` so the agent can drive the corrective-RAG loop:
    on ``quality == "low"`` the agent should rewrite the query
    (split a compound question, add domain keywords, replace
    pronouns) and call this tool again, up to three attempts per
    user turn.

    Args:
        query: Free-form natural-language question (Chinese or
            English; the bge-small-zh embedder is bilingual).
        collection: Collection to search. Empty / missing collections
            return ``quality='low'`` with a ``warning`` field rather
            than raising — safe to probe.
        top_k: Maximum hits to return after fusion (capped at 20).

    Returns:
        ``{collection, query, top_k_returned, quality, top_score,
        mean_score, unique_sources, results: [...]}``.
    """
    return await _search_impl(
        query=query,
        collection=collection,
        top_k=top_k,
    )


async def _list_collections() -> dict[str, Any]:
    """List all FAISS collections currently persisted on disk.

    Useful as the agent's first call when the user implies a library
    but does not name a collection (e.g. "what's in my ESG library?").
    Each entry carries ``name`` and ``chunk_count``; chunk_count
    is ``-1`` if the FAISS pair is unreadable.

    Returns:
        ``{db_dir, collections: [{name, chunk_count}]}``.
    """
    return await _list_collections_impl()


async def _delete_collection(collection: str) -> dict[str, Any]:
    """Delete a collection and its in-memory caches. Idempotent.

    Use ONLY when the user explicitly asks to wipe a collection.
    Missing collections do not raise — the response just reports
    ``existed=False``.

    Returns:
        ``{collection, existed, deleted}``.
    """
    return await _delete_collection_impl(collection=collection)


# ---------------------------------------------------------------------
# StructuredTool exports.
#
# The ``knowledge_`` prefix matches the prefix the MCP-stdio loader
# (``MultiServerMCPClient``) would have produced via
# ``tool_name_prefix=True`` with the ``"knowledge"`` server key. That
# way the agent's KNOWLEDGE_EXPERT_PROMPT — which already names the
# tools as ``knowledge_ingest_pdf`` etc. — works unchanged.
# ---------------------------------------------------------------------
knowledge_ingest_pdf: BaseTool = StructuredTool.from_function(
    coroutine=_ingest_pdf,
    name="knowledge_ingest_pdf",
    description=(
        "Ingest a single local PDF into a persistent FAISS knowledge-"
        "base collection. Args: local_path (str), collection "
        '(str, default "default"), chunk_size (int, default 800), '
        "chunk_overlap (int, default 120). Returns a dict with "
        "num_chunks_added or an error."
    ),
)

knowledge_search: BaseTool = StructuredTool.from_function(
    coroutine=_search,
    name="knowledge_search",
    description=(
        "Hybrid (vector + BM25) search over an ingested knowledge-"
        "base collection. Args: query (str), collection (str, default "
        '"default"), top_k (int, default 5). Returns a dict with '
        "top-level quality ('high'/'medium'/'low'), top_score, "
        "mean_score, and a results list. Use the quality signal to "
        "drive the corrective-RAG retry loop (rewrite + retry on "
        "'low', up to 3 attempts)."
    ),
)

knowledge_list_collections: BaseTool = StructuredTool.from_function(
    coroutine=_list_collections,
    name="knowledge_list_collections",
    description=(
        "List all FAISS collections currently persisted on disk. "
        "Returns {db_dir, collections: [{name, chunk_count}]}. Call "
        "first when the user implies a library but does not name a "
        "collection."
    ),
)

knowledge_delete_collection: BaseTool = StructuredTool.from_function(
    coroutine=_delete_collection,
    name="knowledge_delete_collection",
    description=(
        "Delete a knowledge-base collection (housekeeping). Idempotent: "
        "missing collections return existed=False rather than raising. "
        "Args: collection (str). Use ONLY when the user explicitly asks "
        "to wipe a collection."
    ),
)


KNOWLEDGE_TOOLS: list[BaseTool] = [
    knowledge_ingest_pdf,
    knowledge_search,
    knowledge_list_collections,
    knowledge_delete_collection,
]
"""Canonical roster of the in-process knowledge-base tools.

Hand this list to ``build_knowledge_expert(model_router, mcp_tools)``
(the parameter name is historical — it accepts any ``Sequence[BaseTool]``
and does not care about delivery mechanism)."""


__all__ = [
    "KNOWLEDGE_TOOLS",
    "knowledge_delete_collection",
    "knowledge_ingest_pdf",
    "knowledge_list_collections",
    "knowledge_search",
]
