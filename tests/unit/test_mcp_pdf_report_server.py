"""Phase-4.2: MCP ``pdf_report_server`` round-trip tests.

Why this file is shaped the way it is
-------------------------------------
``pdf_report_server`` is the document plane of the Phase-4 financial
agent. Every tool on it ultimately does I/O: ``search_announcements``
hits the 巨潮资讯 listing endpoint via ``akshare``; ``download_pdf``
hits ``static.cninfo.com.cn``; ``parse_pdf_pages`` and
``extract_pdf_metadata`` are local-only but still need a PDF on disk
that one of the first two calls produced.

Each test therefore opens **one** MCP session and chains as many tool
calls as it needs to verify a contract, reusing the same subprocess
across them — same pattern as ``test_mcp_fin_data_server.py``.

All tests hit live HTTP endpoints, so they are tagged ``network``;
offline CI should run with ``pytest -m 'not network'``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from research_agent.mcp_servers.client_factory import (
    PDF_REPORT_SERVER_MODULE,
    extract_text_content,
)

pytestmark = pytest.mark.network

# Tools as returned by ``load_mcp_tools(session)`` — RAW names, without
# the ``pdf_`` prefix that ``MultiServerMCPClient.get_tools()`` adds
# via ``tool_name_prefix=True``. Prefixing is a client-layer concern,
# not a server-layer one; the Agent-facing ``load_pdf_report_server_tools()``
# helper keeps the ``pdf_`` prefix for supervisor disambiguation, and
# that path is exercised separately by ``scripts/smoke_test_pdf_report_mcp.py``.
EXPECTED_TOOL_NAMES: set[str] = {
    "search_announcements",
    "download_pdf",
    "parse_pdf_pages",
    "extract_pdf_metadata",
}

# 300750 宁德时代 reliably publishes an annual report each March. The
# 2024-published disclosures (covering fiscal year 2023) are stable
# historical data that akshare serves without session cookies — the
# tests remain deterministic even a year after the publish date.
SAMPLE_SYMBOL = "300750"
SAMPLE_START = "20240101"
SAMPLE_END = "20241231"


@asynccontextmanager
async def _open_session() -> AsyncIterator[dict[str, BaseTool]]:
    """Launch one ``pdf_report_server`` subprocess and yield its tools.

    Tools returned here are bound to the open session, so any number
    of ``ainvoke(...)`` calls inside the ``async with`` block reuse
    the same subprocess. This is the fast path.
    """
    client = MultiServerMCPClient(
        {
            "pdf": {
                "command": sys.executable,
                "args": ["-m", PDF_REPORT_SERVER_MODULE],
                "transport": "stdio",
            }
        },
        tool_name_prefix=True,
    )
    async with client.session("pdf") as session:
        from langchain_mcp_adapters.tools import load_mcp_tools

        tools = await load_mcp_tools(session)
        yield {t.name: t for t in tools}


def _parse(raw: object) -> dict[str, Any]:
    """Decode the JSON content block an MCP tool returns."""
    return json.loads(extract_text_content(raw))


async def _first_pdf_url(tools: dict[str, BaseTool]) -> str:
    """Run a known-good search and return the first derivable pdf_url.

    Kept as a helper rather than a fixture because pytest-asyncio
    fixtures across session-scoped contextmanagers would force us to
    share one subprocess across every test in the module, and we'd
    rather fail fast per test than entangle their lifecycles.
    """
    hits = _parse(
        await tools["search_announcements"].ainvoke(
            {
                "symbol": SAMPLE_SYMBOL,
                "start_date": SAMPLE_START,
                "end_date": SAMPLE_END,
                "category": "年报",
            }
        )
    )
    assert "error" not in hits, hits
    for row in hits["announcements"]:
        if row.get("pdf_url"):
            return row["pdf_url"]
    pytest.fail(f"no derivable pdf_url in {hits['announcements']!r}")


# ---------------------------------------------------------------------
# Test 1: tool discovery + search contract + input validation
# ---------------------------------------------------------------------
async def test_discovery_and_search() -> None:
    """All four tools advertised; ``search_announcements`` works end-to-end.

    Consolidates:
      - MCP handshake + tool-schema round-trip
      - Happy-path search (2023 年报 for 宁德时代, 2 stable records)
      - Each record exposes a derivable ``pdf_url``
      - Bad category rejected at the tool boundary (before hitting
        cninfo) with a structured error.
    """
    async with _open_session() as tools:
        assert EXPECTED_TOOL_NAMES.issubset(tools.keys()), (
            f"missing tools: {EXPECTED_TOOL_NAMES - tools.keys()}"
        )

        hits = _parse(
            await tools["search_announcements"].ainvoke(
                {
                    "symbol": SAMPLE_SYMBOL,
                    "start_date": SAMPLE_START,
                    "end_date": SAMPLE_END,
                    "category": "年报",
                }
            )
        )
        assert "error" not in hits, hits
        assert hits["symbol"] == SAMPLE_SYMBOL
        assert hits["count"] >= 1
        announcements = hits["announcements"]
        assert len(announcements) == hits["count"]

        row = announcements[0]
        # Contract: every row has the six fields agent prompts rely on.
        for key in ("code", "name", "title", "publish_date", "detail_url", "pdf_url"):
            assert key in row, f"missing {key!r} in {row!r}"
        assert row["code"] == SAMPLE_SYMBOL
        assert row["pdf_url"] is not None and row["pdf_url"].endswith(".PDF"), (
            f"expected a cninfo finalpage .PDF URL, got {row['pdf_url']!r}"
        )

        bad = _parse(
            await tools["search_announcements"].ainvoke(
                {
                    "symbol": SAMPLE_SYMBOL,
                    "start_date": SAMPLE_START,
                    "end_date": SAMPLE_END,
                    "category": "NOT_A_REAL_CATEGORY",
                }
            )
        )
        assert "error" in bad
        assert "category must be one of" in bad["error"], bad


# ---------------------------------------------------------------------
# Test 2: download + on-disk cache + URL validation
# ---------------------------------------------------------------------
async def test_download_and_cache() -> None:
    """``download_pdf`` writes a valid PDF and is idempotent on re-call.

    Verifies:
      - First call actually downloads and writes a %PDF-magic file,
        returning ``from_cache=False`` and a positive ``size_bytes``.
      - Second call with the same URL hits the cache
        (``from_cache=True``) and returns the identical path.
      - An absolute non-http URL fails at the tool boundary with a
        structured error, never hitting the wire.
    """
    async with _open_session() as tools:
        pdf_url = await _first_pdf_url(tools)

        first = _parse(await tools["download_pdf"].ainvoke({"pdf_url": pdf_url}))
        assert "error" not in first, first
        assert first["pdf_url"] == pdf_url
        assert first["size_bytes"] > 10_000, first  # even the shortest summary is >10 KB
        assert isinstance(first["from_cache"], bool)
        local_path_first = first["local_path"]

        second = _parse(await tools["download_pdf"].ainvoke({"pdf_url": pdf_url}))
        assert "error" not in second, second
        assert second["from_cache"] is True, (
            f"second call should hit cache, got from_cache={second['from_cache']!r}"
        )
        assert second["local_path"] == local_path_first
        assert second["size_bytes"] == first["size_bytes"]

        bad = _parse(await tools["download_pdf"].ainvoke({"pdf_url": "not-a-url"}))
        assert "error" in bad
        assert "absolute http" in bad["error"], bad


# ---------------------------------------------------------------------
# Test 3: parse + metadata + page-window guard
# ---------------------------------------------------------------------
async def test_parse_and_metadata() -> None:
    """Page-range extraction and metadata are consistent for one PDF.

    Chain: search → download → parse_pdf_pages → extract_pdf_metadata.

    Verifies:
      - ``extract_pdf_metadata`` reports a positive ``num_pages`` and
        a ``metadata`` dict with lowercase keys.
      - ``parse_pdf_pages`` returns text for pages ``[1, N]`` where
        ``N <= total_pages``, with correct 1-indexed page numbers and
        non-trivial ``char_count`` for at least one page (cninfo
        annual reports are text-layered, not scanned images).
      - Requesting a window larger than ``MAX_PAGE_WINDOW`` is
        rejected at the tool boundary.
    """
    async with _open_session() as tools:
        pdf_url = await _first_pdf_url(tools)

        dl = _parse(await tools["download_pdf"].ainvoke({"pdf_url": pdf_url}))
        assert "error" not in dl, dl
        local_path = dl["local_path"]

        meta = _parse(
            await tools["extract_pdf_metadata"].ainvoke({"local_path": local_path})
        )
        assert "error" not in meta, meta
        assert meta["local_path"] == local_path
        assert meta["num_pages"] >= 1
        assert meta["size_bytes"] == dl["size_bytes"]
        assert isinstance(meta["metadata"], dict)
        # PDF metadata keys, when present, should be the lowercase form
        # produced by our ``key_map`` normalization — not raw ``/Title``.
        for k in meta["metadata"]:
            assert not k.startswith("/"), (
                f"metadata key {k!r} should have been stripped of its leading slash"
            )

        pages_payload = _parse(
            await tools["parse_pdf_pages"].ainvoke(
                {"local_path": local_path, "start_page": 1, "end_page": 3}
            )
        )
        assert "error" not in pages_payload, pages_payload
        assert pages_payload["total_pages"] == meta["num_pages"]
        assert pages_payload["requested_range"] == {"start": 1, "end": 3}
        pages = pages_payload["pages"]
        expected_returned = min(3, meta["num_pages"])
        assert len(pages) == expected_returned
        assert [p["page_number"] for p in pages] == list(range(1, expected_returned + 1))
        assert any(p["char_count"] > 50 for p in pages), (
            "expected at least one page with >50 chars of extracted text"
        )

        too_wide = _parse(
            await tools["parse_pdf_pages"].ainvoke(
                {"local_path": local_path, "start_page": 1, "end_page": 100}
            )
        )
        assert "error" in too_wide
        assert "page window" in too_wide["error"], too_wide
