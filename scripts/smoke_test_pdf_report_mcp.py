"""冒烟测试 — pdf_report_server MCP 端到端生产路径验证。

与 ``tests/unit/test_mcp_pdf_report_server.py`` 不同（后者打开单个 session 并跳过客户端级别的前缀以提高速度），本脚本走的是与 Agent运行时完全相同的代码路径：

    load_pdf_report_server_tools()       # 通过 MultiServerMCPClient
        -> 启动子进程
        -> 发现工具
        -> 添加 ``pdf_`` 前缀
    tool.ainvoke(...)                    # 再次进入，正常 Agent 流程
        -> 启动子进程
        -> 执行 HTTP / pypdf 操作
        -> 返回 MCP 内容块

在接入 ``report_expert`` 专家之前运行本脚本，可捕获 schema / 前缀 / JSON 序列化问题 — 这些是跳过客户端前缀的单元测试无法发现的。

退出码:
    0 → 全部 4 个工具执行成功
    1 → 任何工具以非结构化错误崩溃

用法::

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
    """Agent 可以从中恢复的"优雅"失败。"""
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

    # ---- 工具 1: search_announcements ----
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

    # ---- 工具 2: download_pdf（第二次调用测试缓存） ----
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

    # ---- 工具 3: extract_pdf_metadata ----
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

    # ---- 工具 4: parse_pdf_pages ----
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
