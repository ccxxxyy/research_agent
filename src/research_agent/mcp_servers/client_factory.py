"""Factories for loading MCP tools into the LangGraph/LangChain runtime.

Why a factory module?
---------------------
``MultiServerMCPClient`` from ``langchain_mcp_adapters`` spawns a fresh
stdio subprocess each time a tool is invoked. We want one single place
that owns the subprocess-launch parameters (Python executable, module
path, transport) so every Agent builder / test / demo talks to the
exact same server surface. Cherry-picking ``sys.executable`` and ``-m``
paths at four different call sites would be a maintenance trap.

Usage
-----
Async (production / scripts / tests)::

    tools = await load_code_server_tools()
    supervisor = build_minimal_supervisor(
        model_router=router,
        coder_tools=tools,
    )

The returned tools are ``langchain_core.tools.BaseTool`` instances that
work with ``create_react_agent`` and ``langgraph_supervisor`` exactly
like locally-defined ``@tool`` functions would.
"""

from __future__ import annotations

import sys
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


def _stdio_server_spec(module: str) -> dict[str, Any]:
    """Build a stdio launch spec for an in-repo MCP server module.

    Using ``sys.executable`` guarantees the subprocess inherits the same
    virtualenv (and thus the same ``research_agent`` install) as the
    parent process. Using ``-m`` avoids hard-coded file paths that would
    break on CI / other checkouts.
    """
    return {
        "command": sys.executable,
        "args": ["-m", module],
        "transport": "stdio",
    }


CODE_SERVER_MODULE = "research_agent.mcp_servers.code_server"
ECHO_SERVER_MODULE = "research_agent.mcp_servers.echo_server"
FIN_DATA_SERVER_MODULE = "research_agent.mcp_servers.fin_data_server"
PDF_REPORT_SERVER_MODULE = "research_agent.mcp_servers.pdf_report_server"
KNOWLEDGE_SERVER_MODULE = "research_agent.mcp_servers.knowledge_server"
NEWS_SERVER_MODULE = "research_agent.mcp_servers.news_server"
NEWS_SENTIMENT_SERVER_MODULE = "research_agent.mcp_servers.news_sentiment_server"


async def load_code_server_tools() -> list[BaseTool]:
    """Spawn the ``code_server`` over stdio and return its tool list.

    Currently exposes one tool: ``code_execute_python`` (the
    ``tool_name_prefix=True`` flag prepends the server key ``code``).
    Callers should be prepared for the tool name to be prefixed.
    """
    client = MultiServerMCPClient(
        {"code": _stdio_server_spec(CODE_SERVER_MODULE)},
        tool_name_prefix=True,
    )
    return await client.get_tools()


async def load_echo_server_tools() -> list[BaseTool]:
    """Spawn the ``echo_server`` over stdio and return its tool list.

    Primarily used by the MCP smoke tests; production agents do not
    consume the echo tools.
    """
    client = MultiServerMCPClient(
        {"echo": _stdio_server_spec(ECHO_SERVER_MODULE)},
        tool_name_prefix=True,
    )
    return await client.get_tools()


async def load_fin_data_server_tools() -> list[BaseTool]:
    """Spawn the ``fin_data_server`` over stdio and return its tool list.

    Exposes five A-share data tools, each prefixed with ``fin_`` by
    ``tool_name_prefix=True``:

    - ``fin_get_stock_basic_info``
    - ``fin_get_stock_price_history``
    - ``fin_get_financial_abstract``
    - ``fin_get_financial_indicators``
    - ``fin_search_stock_by_name``

    The first call to the MCP subprocess will load ``akshare`` and may
    take ~1 second for its internal deferred imports; subsequent tool
    invocations reuse the already-warm process.
    """
    client = MultiServerMCPClient(
        {"fin": _stdio_server_spec(FIN_DATA_SERVER_MODULE)},
        tool_name_prefix=True,
    )
    return await client.get_tools()


async def load_news_server_tools() -> list[BaseTool]:
    """Spawn the ``news_server`` over stdio and return its tool list.

    Exposes five A-share news tools, each prefixed with ``news_`` by
    ``tool_name_prefix=True``:

    - ``news_get_stock_news``        — 东方财富 individual-stock news.
    - ``news_get_market_telegraph``  — 财联社 real-time flashes.
    - ``news_get_hot_keywords``      — 东方财富 trending keywords.
    - ``news_get_economic_news``     — 百度财经 早晚报 digest.
    - ``news_get_xueqiu_discussion_hot_rank`` — 雪球讨论热度个股榜
      (``akshare.stock_hot_tweet_xq``).

    Like ``fin_data_server``, the subprocess imports ``akshare``
    lazily on first tool call (~1 second) and reuses the warm
    interpreter for subsequent calls.
    """
    client = MultiServerMCPClient(
        {"news": _stdio_server_spec(NEWS_SERVER_MODULE)},
        tool_name_prefix=True,
    )
    return await client.get_tools()


async def load_news_sentiment_server_tools() -> list[BaseTool]:
    """Spawn ``news_sentiment_server`` over stdio and return its tool list.

    Exposes two sentiment-analysis tools, each prefixed with
    ``sentiment_`` by ``tool_name_prefix=True``:

    - ``sentiment_analyze_text_sentiment`` — pure text scoring: pass
      in a list of strings, get back per-item sentiment scores +
      aggregate statistics. No external data source dependency.
    - ``sentiment_get_stock_sentiment_report`` — one-stop: fetch
      Eastmoney news for a ticker → score each item → return a
      structured report with per-item scores + aggregate + audit
      metadata (model version, text fingerprint, timestamp).

    The subprocess imports ``snownlp`` + ``akshare`` lazily on first
    tool call; subsequent calls reuse the warm interpreter.
    """
    client = MultiServerMCPClient(
        {"sentiment": _stdio_server_spec(NEWS_SENTIMENT_SERVER_MODULE)},
        tool_name_prefix=True,
    )
    return await client.get_tools()


async def load_pdf_report_server_tools() -> list[BaseTool]:
    """Spawn the ``pdf_report_server`` over stdio and return its tool list.

    Exposes four disclosure-PDF tools, each prefixed with ``pdf_`` by
    ``tool_name_prefix=True``:

    - ``pdf_search_announcements`` — list cninfo announcements with
      pre-derived ``pdf_url`` fields.
    - ``pdf_download_pdf`` — cache-aware download into
      ``./data/pdf_cache/``.
    - ``pdf_parse_pdf_pages`` — bounded page-range text extraction
      (max 20 pages per call).
    - ``pdf_extract_pdf_metadata`` — page count / title / author /
      size.

    The subprocess imports ``pypdf`` and ``httpx`` lazily at tool-call
    time, so launch is fast even on cold starts.
    """
    client = MultiServerMCPClient(
        {"pdf": _stdio_server_spec(PDF_REPORT_SERVER_MODULE)},
        tool_name_prefix=True,
    )
    return await client.get_tools()


async def load_knowledge_server_tools() -> list[BaseTool]:
    """**Deprecated** — kept only for backwards compatibility.

    The MCP-stdio delivery path for ``knowledge_server`` is unstable
    on Windows + Python 3.13: after a successful ``ingest_pdf`` the
    fastmcp stdout writer never flushes the JSON-RPC response back to
    the parent. See ``knowledge_server.py`` module docstring for the
    full forensic trail.

    Production code should call
    :func:`load_knowledge_tools_inproc` instead, which returns the
    same four tools with the same ``knowledge_*`` names but invokes
    them in-process (no subprocess, no JSON-RPC framing, no stdio
    pipes).

    This function is preserved so older scripts / tests don't break,
    but it raises immediately on first use to make the deprecation
    impossible to miss.
    """
    raise RuntimeError(
        "load_knowledge_server_tools (MCP-stdio) is deprecated; "
        "use load_knowledge_tools_inproc() instead. The stdio path "
        "is known to deadlock on Windows + Python 3.13 with the "
        "FAISS/sentence-transformers import chain — see "
        "knowledge_server.py for the diagnosis."
    )


async def load_knowledge_tools_inproc() -> list[BaseTool]:
    """Return the four knowledge-base tools as in-process LangChain tools.

    Same toolbelt the (legacy) MCP-stdio loader produced — same names,
    same arg shapes, same return dicts — but delivered through
    ``research_agent.tools.knowledge_tools.KNOWLEDGE_TOOLS`` instead
    of an MCP subprocess. The signature is kept ``async`` so callers
    can swap loaders without changing call sites.

    Tools returned (each is a ``StructuredTool``):

    - ``knowledge_ingest_pdf``      — load → chunk → embed → write to
      a persistent FAISS collection under ``./data/knowledge_db/``.
    - ``knowledge_search``          — hybrid retrieval (vector + BM25)
      with corrective-RAG quality signals (``quality`` ∈
      ``{high, medium, low}``, plus per-hit scores).
    - ``knowledge_list_collections`` — enumerate persisted collections.
    - ``knowledge_delete_collection`` — idempotent housekeeping.

    Cost profile: importing this loader pulls in
    ``langchain_text_splitters`` + ``faiss-cpu`` (~9 s the first
    time, cached thereafter). The bge-small embedding model is
    *not* loaded until the first ``ingest_pdf`` / ``search`` call
    (~3–17 s cold, sub-second warm).
    """
    from research_agent.tools.knowledge_tools import KNOWLEDGE_TOOLS

    return list(KNOWLEDGE_TOOLS)


def extract_text_content(value: object) -> str:
    """Flatten the content-block list returned by langchain-mcp-adapters.

    Newer versions of ``langchain_mcp_adapters`` (>=0.1) wrap every tool
    response in a list of content blocks shaped like
    ``[{'type': 'text', 'text': '...', 'id': '...'}]``. Older versions
    returned scalars directly. This helper normalizes both shapes to a
    plain string so downstream assertions don't need to know which
    version is installed.
    """
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(value)


__all__ = [
    "CODE_SERVER_MODULE",
    "ECHO_SERVER_MODULE",
    "FIN_DATA_SERVER_MODULE",
    "KNOWLEDGE_SERVER_MODULE",
    "NEWS_SENTIMENT_SERVER_MODULE",
    "NEWS_SERVER_MODULE",
    "PDF_REPORT_SERVER_MODULE",
    "extract_text_content",
    "load_code_server_tools",
    "load_echo_server_tools",
    "load_fin_data_server_tools",
    "load_knowledge_server_tools",
    "load_knowledge_tools_inproc",
    "load_news_sentiment_server_tools",
    "load_news_server_tools",
    "load_pdf_report_server_tools",
]
