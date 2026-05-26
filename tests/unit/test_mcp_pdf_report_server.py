"""Phase-4.2：MCP ``pdf_report_server`` 往返测试。

为什么此文件如此组织
-------------------------------------
``pdf_report_server`` 是 Phase-4 金融 Agent 的文档平面。其上的每个工具最终都涉及 I/O：``search_announcements`` 通过``akshare`` 访问巨潮资讯的列表端点；
``download_pdf`` 访问``static.cninfo.com.cn``；
``parse_pdf_pages`` 和``extract_pdf_metadata`` 仅在本地执行，但仍需要前两个调用生成的磁盘上的 PDF 文件。

因此每个测试打开一个MCP 会话，并链式调用所有需要验证契约的工具，跨调用复用同一子进程——与 ``test_mcp_fin_data_server.py``模式相同。

所有测试访问实时 HTTP 端点，因此标记为 ``network``；离线 CI 应使用 ``pytest -m 'not network'`` 运行。
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

# 工具名称由 ``load_mcp_tools(session)`` 返回——原始名称，不带 ``MultiServerMCPClient.get_tools()`` 通过 ``tool_name_prefix=True``添加的 ``pdf_`` 前缀。
# 前缀是客户端层的关注点，而非服务器层的；
# 面向 Agent 的 ``load_pdf_report_server_tools()`` 辅助函数保留``pdf_`` 前缀用于 supervisor 消歧，该路径由``scripts/smoke_test_pdf_report_mcp.py`` 单独验证。
EXPECTED_TOOL_NAMES: set[str] = {
    "search_announcements",
    "download_pdf",
    "parse_pdf_pages",
    "extract_pdf_metadata",
}

# 300750 宁德时代每年 3 月可靠地发布年报。2024 年发布的披露（涵盖 2023 财年）是稳定的历史数据，akshare 无需会话 cookie即可提供——即使在发布日期一年后，测试仍保持确定性。
SAMPLE_SYMBOL = "300750"
SAMPLE_START = "20240101"
SAMPLE_END = "20241231"


@asynccontextmanager
async def _open_session() -> AsyncIterator[dict[str, BaseTool]]:
    """启动一个 ``pdf_report_server`` 子进程并 yield 其工具。

    此处返回的工具绑定到已打开的会话，因此 ``async with`` 块内 任意数量的 ``ainvoke(...)`` 调用都复用同一个子进程。这是快速路径。
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
    """解码 MCP 工具返回的 JSON 内容块。"""
    return json.loads(extract_text_content(raw))


async def _first_pdf_url(tools: dict[str, BaseTool]) -> str:
    """执行一次已知可用的搜索并返回第一个可派生的 pdf_url。

    保持为辅助函数而非 fixture，因为跨会话作用域上下文管理器的 pytest-asyncio fixture 会迫使模块内所有测试共享同一子进程，
    更希望每个测试独立快速失败，而非耦合它们的生命周期。
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
# 测试 1：工具发现 + 搜索契约 + 输入校验
# ---------------------------------------------------------------------
async def test_discovery_and_search() -> None:
    """所有四个工具均已发布；``search_announcements`` 端到端正常工作。

    整合验证：
      - MCP 握手 + 工具 schema 往返
      - 正常路径搜索（宁德时代 2023 年报，2 条稳定记录）
      - 每条记录暴露一个可派生的 ``pdf_url``
      - 无效分类在工具边界（访问 cninfo 之前）被拒绝并返回结构化错误。
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
        # 契约：每行包含 Agent 提示词依赖的六个字段。
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
# 测试 2：下载 + 磁盘缓存 + URL 校验
# ---------------------------------------------------------------------
async def test_download_and_cache() -> None:
    """``download_pdf`` 写入有效 PDF 且重复调用具有幂等性。

    验证：
      - 首次调用实际下载并写入 %PDF-magic 文件， 返回 ``from_cache=False`` 和正数 ``size_bytes``。
      - 相同 URL 的第二次调用命中缓存 （``from_cache=True``）并返回相同路径。
      - 非 http 的绝对 URL 在工具边界以结构化错误失败，不会触达网络。
    """
    async with _open_session() as tools:
        pdf_url = await _first_pdf_url(tools)

        first = _parse(await tools["download_pdf"].ainvoke({"pdf_url": pdf_url}))
        assert "error" not in first, first
        assert first["pdf_url"] == pdf_url
        assert first["size_bytes"] > 10_000, first  # 即使最短的摘要也 >10 KB
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
# 测试 3：解析 + 元数据 + 页面窗口限制
# ---------------------------------------------------------------------
async def test_parse_and_metadata() -> None:
    """单个 PDF 的页面范围提取与元数据一致性验证。

    调用链：search → download → parse_pdf_pages → extract_pdf_metadata。

    验证：
      - ``extract_pdf_metadata`` 报告正数 ``num_pages``  含小写键的 ``metadata`` 字典。
      - ``parse_pdf_pages`` 返回页面 ``[1, N]``（``N <= total_pages``）的文本，使用正确的 1 索引页码，且至少一页有非零的 ``char_count``（巨潮年报是文本层而非扫描图像）。
      - 请求超过 ``MAX_PAGE_WINDOW`` 的窗口在工具边界被拒绝。
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
        # PDF 元数据键（如果存在）应为我们 ``key_map`` 规范化后的 小写形式 — 而非原始的 ``/Title``。
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
