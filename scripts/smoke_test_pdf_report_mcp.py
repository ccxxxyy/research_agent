"""Phase 4.2 smoke test — pdf_report_server MCP end-to-end via production path.

Unlike ``tests/unit/test_mcp_pdf_report_server.py`` which opens a single
session and bypasses the client-level prefixing for speed, this script
walks the **exact same code path** the Agent will use at runtime:

    load_pdf_report_server_tools()       # via MultiServerMCPClient
        -> spawns subprocess
        -> discovers tools
        -> applies ``pdf_`` prefix
    tool.ainvoke(...)                    # re-entry, normal Agent flow
        -> spawns subprocess again
        -> runs HTTP / pypdf work
        -> returns MCP content block

Running this script before wiring the ``report_expert`` specialist
catches schema / prefix / JSON-serialization issues that unit tests
(which skip the client-level prefix) would miss.

Exit code:
    0 → all 4 tools executed successfully
    1 → any tool crashed with a non-structured error

Usage::

    uv run python scripts/smoke_test_pdf_report_mcp.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from loguru import logger

from research_agent.mcp_servers.client_factory import (
    extract_text_content,
    load_pdf_report_server_tools,
)

SAMPLE_SYMBOL = "300750"  # 宁德时代
SAMPLE_START = "20240101"
SAMPLE_END = "20241231"

EXPECTED_PREFIXED_NAMES: set[str] = {
    "pdf_search_announcements",
    "pdf_download_pdf",
    "pdf_parse_pdf_pages",
    "pdf_extract_pdf_metadata",
}


def _parse(raw: object) -> dict[str, Any]:
    return json.loads(extract_text_content(raw))


def _is_structured_error(payload: dict[str, Any]) -> bool:
    """A 'graceful' failure that the Agent can recover from."""
    return "error" in payload and "context" in payload


async def main() -> int:
    logger.info("=" * 60)
    logger.info("Phase 4.2 smoke test — pdf_report_server via production path")
    logger.info("=" * 60)

    tools = await load_pdf_report_server_tools()
    tool_map = {t.name: t for t in tools}

    logger.info("Discovered {} tools:", len(tools))
    for name in sorted(tool_map):
        logger.info("  - {}", name)

    missing = EXPECTED_PREFIXED_NAMES - tool_map.keys()
    if missing:
        logger.error("Missing expected tools: {}", missing)
        return 1
    logger.success("All 4 prefixed tools discovered.")

    all_ok = True

    # ---- Tool 1: search_announcements ----
    logger.info(
        "\n[1/4] pdf_search_announcements(symbol='{}', category='年报') ...",
        SAMPLE_SYMBOL,
    )
    search_payload = _parse(
        await tool_map["pdf_search_announcements"].ainvoke(
            {
                "symbol": SAMPLE_SYMBOL,
                "start_date": SAMPLE_START,
                "end_date": SAMPLE_END,
                "category": "年报",
            }
        )
    )
    if "error" in search_payload:
        if _is_structured_error(search_payload):
            logger.warning("  STRUCTURED FAILURE: {}", search_payload["error"])
        else:
            logger.error("  FAIL (unstructured): {}", search_payload)
        all_ok = False
        return 1 if all_ok is False else 0
    logger.success(
        "  OK ({} announcements; first title: {!r})",
        search_payload["count"],
        search_payload["announcements"][0]["title"] if search_payload["announcements"] else None,
    )

    pdf_url = next(
        (r["pdf_url"] for r in search_payload["announcements"] if r.get("pdf_url")),
        None,
    )
    if pdf_url is None:
        logger.error("  no derivable pdf_url in search results; cannot continue.")
        return 1
    logger.info("  using pdf_url = {}", pdf_url)

    # ---- Tool 2: download_pdf (exercises cache on 2nd call) ----
    logger.info("\n[2/4] pdf_download_pdf(pdf_url=...) ...")
    dl_first = _parse(
        await tool_map["pdf_download_pdf"].ainvoke({"pdf_url": pdf_url})
    )
    if "error" in dl_first:
        if _is_structured_error(dl_first):
            logger.warning("  STRUCTURED FAILURE: {}", dl_first["error"])
        else:
            logger.error("  FAIL (unstructured): {}", dl_first)
        all_ok = False
        return 1
    logger.success(
        "  OK (size={} bytes, from_cache={}, path={})",
        dl_first["size_bytes"],
        dl_first["from_cache"],
        dl_first["local_path"],
    )

    dl_second = _parse(
        await tool_map["pdf_download_pdf"].ainvoke({"pdf_url": pdf_url})
    )
    if dl_second.get("from_cache") is not True:
        logger.error("  FAIL: 2nd call should have hit cache, got: {}", dl_second)
        all_ok = False
    else:
        logger.success("  OK (2nd call hit cache, same path)")

    local_path = dl_first["local_path"]

    # ---- Tool 3: extract_pdf_metadata ----
    logger.info("\n[3/4] pdf_extract_pdf_metadata(local_path=...) ...")
    meta = _parse(
        await tool_map["pdf_extract_pdf_metadata"].ainvoke({"local_path": local_path})
    )
    if "error" in meta:
        logger.error("  FAIL: {}", meta)
        all_ok = False
    else:
        logger.success(
            "  OK (num_pages={}, metadata keys: {})",
            meta["num_pages"],
            list(meta["metadata"].keys()),
        )

    # ---- Tool 4: parse_pdf_pages ----
    logger.info("\n[4/4] pdf_parse_pdf_pages(local_path=..., pages 1-3) ...")
    pages = _parse(
        await tool_map["pdf_parse_pdf_pages"].ainvoke(
            {"local_path": local_path, "start_page": 1, "end_page": 3}
        )
    )
    if "error" in pages:
        logger.error("  FAIL: {}", pages)
        all_ok = False
    else:
        chars = [p["char_count"] for p in pages["pages"]]
        logger.success(
            "  OK (total_pages={}, returned={} pages, chars per page: {})",
            pages["total_pages"],
            len(pages["pages"]),
            chars,
        )

    logger.info("\n" + "=" * 60)
    if all_ok:
        logger.success("Phase 4.2 smoke test: ALL 4 TOOLS OK via production path.")
        return 0
    logger.error("Phase 4.2 smoke test: one or more tools failed.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
