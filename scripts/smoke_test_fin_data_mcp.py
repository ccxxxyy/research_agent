"""Phase 4.1 smoke test — fin_data_server MCP end-to-end via production path.

Unlike ``tests/unit/test_mcp_fin_data_server.py`` which opens a single
session and bypasses the client-level prefixing for speed, this
script walks the **exact same code path** the Agent will use at
runtime:

    load_fin_data_server_tools()        # via MultiServerMCPClient
        -> spawns subprocess
        -> discovers tools
        -> applies ``fin_`` prefix
    tool.ainvoke(...)                   # re-entry, normal Agent flow
        -> spawns subprocess again
        -> runs akshare call
        -> returns MCP content block

Running this script before wiring the ``data_expert`` specialist
catches schema / prefix / JSON-serialization issues that unit tests
(which skip the client-level prefix) would miss.

Exit code:
    0 → all 5 tools executed successfully (or gracefully fell back)
    1 → any tool crashed with a non-structured error

Usage::

    uv run python scripts/smoke_test_fin_data_mcp.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from loguru import logger

from research_agent.mcp_servers.client_factory import (
    extract_text_content,
    load_fin_data_server_tools,
)

SAMPLE_SYMBOL = "300750"  # 宁德时代
SAMPLE_KEYWORD = "平安"

EXPECTED_PREFIXED_NAMES: set[str] = {
    "fin_get_stock_basic_info",
    "fin_get_stock_price_history",
    "fin_get_financial_abstract",
    "fin_get_financial_indicators",
    "fin_search_stock_by_name",
}


def _parse(raw: object) -> dict[str, Any]:
    return json.loads(extract_text_content(raw))


def _is_structured_error(payload: dict[str, Any]) -> bool:
    """A 'graceful' failure that the Agent can recover from."""
    return "error" in payload and "context" in payload


async def main() -> int:
    logger.info("=" * 60)
    logger.info("Phase 4.1 smoke test — fin_data_server via production path")
    logger.info("=" * 60)

    tools = await load_fin_data_server_tools()
    tool_map = {t.name: t for t in tools}

    logger.info("Discovered {} tools:", len(tools))
    for name in sorted(tool_map):
        logger.info("  - {}", name)

    missing = EXPECTED_PREFIXED_NAMES - tool_map.keys()
    if missing:
        logger.error("Missing expected tools: {}", missing)
        return 1
    logger.success("All 5 prefixed tools discovered.")

    all_ok = True

    # ---- Tool 1: search_stock_by_name (deterministic filter) ----
    logger.info("\n[1/5] fin_search_stock_by_name(keyword='{}') ...", SAMPLE_KEYWORD)
    payload = _parse(
        await tool_map["fin_search_stock_by_name"].ainvoke(
            {"keyword": SAMPLE_KEYWORD, "limit": 5}
        )
    )
    if "error" in payload:
        logger.error("  FAIL: {}", payload)
        all_ok = False
    else:
        logger.success(
            "  OK ({} matches, first: {})",
            len(payload["matches"]),
            payload["matches"][0] if payload["matches"] else None,
        )

    # ---- Tool 2: get_stock_basic_info (multi-source fallback) ----
    logger.info("\n[2/5] fin_get_stock_basic_info(symbol='{}') ...", SAMPLE_SYMBOL)
    payload = _parse(
        await tool_map["fin_get_stock_basic_info"].ainvoke({"symbol": SAMPLE_SYMBOL})
    )
    if "error" in payload:
        if _is_structured_error(payload):
            logger.warning(
                "  STRUCTURED FAILURE (both sources down): {}",
                payload.get("attempts", payload["error"]),
            )
        else:
            logger.error("  FAIL (unstructured): {}", payload)
            all_ok = False
    else:
        logger.success(
            "  OK (source={}, {} info fields)",
            payload["source"],
            len(payload["info"]),
        )

    # ---- Tool 3: get_stock_price_history (multi-source fallback) ----
    logger.info("\n[3/5] fin_get_stock_price_history(symbol='{}', days=15) ...", SAMPLE_SYMBOL)
    payload = _parse(
        await tool_map["fin_get_stock_price_history"].ainvoke(
            {"symbol": SAMPLE_SYMBOL, "days": 15}
        )
    )
    if "error" in payload:
        if _is_structured_error(payload):
            logger.warning(
                "  STRUCTURED FAILURE: {}",
                payload.get("attempts", payload["error"]),
            )
        else:
            logger.error("  FAIL (unstructured): {}", payload)
            all_ok = False
    else:
        logger.success(
            "  OK (source={}, sessions={}, pct_change={}%)",
            payload["source"],
            payload["summary"]["sessions"],
            payload["summary"]["pct_change"],
        )

    # ---- Tool 4: get_financial_abstract ----
    logger.info("\n[4/5] fin_get_financial_abstract(symbol='{}', last_n_periods=2) ...", SAMPLE_SYMBOL)
    payload = _parse(
        await tool_map["fin_get_financial_abstract"].ainvoke(
            {"symbol": SAMPLE_SYMBOL, "last_n_periods": 2}
        )
    )
    if "error" in payload:
        logger.error("  FAIL: {}", payload)
        all_ok = False
    else:
        logger.success(
            "  OK (periods={}, metrics={})",
            payload["periods"],
            list(payload["metrics"].keys())[:3],
        )

    # ---- Tool 5: get_financial_indicators ----
    logger.info("\n[5/5] fin_get_financial_indicators(symbol='{}', start_year='2024') ...", SAMPLE_SYMBOL)
    payload = _parse(
        await tool_map["fin_get_financial_indicators"].ainvoke(
            {"symbol": SAMPLE_SYMBOL, "start_year": "2024"}
        )
    )
    if "error" in payload:
        logger.error("  FAIL: {}", payload)
        all_ok = False
    else:
        logger.success(
            "  OK ({} periods, first 3 ratios: {})",
            len(payload["periods"]),
            list(payload["ratios"].keys())[:3],
        )

    logger.info("\n" + "=" * 60)
    if all_ok:
        logger.success("Phase 4.1 smoke test: ALL 5 TOOLS OK via production path.")
        return 0
    logger.error("Phase 4.1 smoke test: one or more tools failed.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
