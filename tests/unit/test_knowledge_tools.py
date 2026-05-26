"""单元测试 — 进程内知识库工具包装器。

验证 ``research_agent.tools.knowledge_tools`` 正确地将四个知识库工具暴露为 ``StructuredTool`` 实例，且具有正确的名称、
参数 schema 和透传语义。不加载 embedding 模型或 FAISS 索引 — 底层``knowledge_server`` 函数通过 monkeypatch 替换。
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from research_agent.tools import knowledge_tools as kt
from research_agent.tools.knowledge_tools import (
    KNOWLEDGE_TOOLS,
    knowledge_delete_collection,
    knowledge_ingest_pdf,
    knowledge_list_collections,
    knowledge_search,
)


class TestKnowledgeToolsRoster:
    """守护导出的工具清单结构 — Agent 和提示词依赖于此。"""

    def test_roster_has_four_tools(self) -> None:
        assert len(KNOWLEDGE_TOOLS) == 4

    def test_roster_names(self) -> None:
        names = {t.name for t in KNOWLEDGE_TOOLS}
        assert names == {
            "knowledge_ingest_pdf",
            "knowledge_search",
            "knowledge_list_collections",
            "knowledge_delete_collection",
        }

    def test_all_tools_are_base_tool_instances(self) -> None:
        from langchain_core.tools import BaseTool

        for t in KNOWLEDGE_TOOLS:
            assert isinstance(t, BaseTool), f"{t.name} is not a BaseTool"

    def test_module_all_matches_roster(self) -> None:
        assert "KNOWLEDGE_TOOLS" in kt.__all__
        for t in KNOWLEDGE_TOOLS:
            assert t.name in kt.__all__


class TestIngestPdfTool:
    def test_name_and_required_args(self) -> None:
        assert knowledge_ingest_pdf.name == "knowledge_ingest_pdf"
        args = knowledge_ingest_pdf.args
        assert "local_path" in args
        assert args["local_path"]["type"] == "string"

    def test_has_collection_arg_with_default(self) -> None:
        args = knowledge_ingest_pdf.args
        assert "collection" in args
        assert args["collection"].get("default") == "default"

    @pytest.mark.asyncio
    async def test_delegates_to_impl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_result: dict[str, Any] = {
            "collection": "test",
            "source": "/tmp/x.pdf",
            "num_pages": 2,
            "num_chunks_added": 10,
            "total_chunks_in_collection": 10,
        }
        mock = AsyncMock(return_value=fake_result)
        monkeypatch.setattr(kt, "_ingest_pdf_impl", mock)

        result = await kt._ingest_pdf(local_path="/tmp/x.pdf", collection="test")
        assert result == fake_result
        mock.assert_awaited_once_with(
            local_path="/tmp/x.pdf",
            collection="test",
            chunk_size=kt.DEFAULT_CHUNK_SIZE,
            chunk_overlap=kt.DEFAULT_CHUNK_OVERLAP,
        )


class TestSearchTool:
    def test_name_and_required_args(self) -> None:
        assert knowledge_search.name == "knowledge_search"
        args = knowledge_search.args
        assert "query" in args
        assert args["query"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_delegates_to_impl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_result: dict[str, Any] = {
            "collection": "default",
            "query": "test",
            "top_k_returned": 1,
            "quality": "high",
            "results": [],
        }
        mock = AsyncMock(return_value=fake_result)
        monkeypatch.setattr(kt, "_search_impl", mock)

        result = await kt._search(query="test", collection="default", top_k=3)
        assert result == fake_result
        mock.assert_awaited_once_with(query="test", collection="default", top_k=3)


class TestListCollectionsTool:
    def test_name(self) -> None:
        assert knowledge_list_collections.name == "knowledge_list_collections"

    @pytest.mark.asyncio
    async def test_delegates_to_impl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_result: dict[str, Any] = {
            "db_dir": "/tmp/kb",
            "collections": [{"name": "c1", "chunk_count": 42}],
        }
        mock = AsyncMock(return_value=fake_result)
        monkeypatch.setattr(kt, "_list_collections_impl", mock)

        result = await kt._list_collections()
        assert result == fake_result
        mock.assert_awaited_once()


class TestDeleteCollectionTool:
    def test_name_and_required_args(self) -> None:
        assert knowledge_delete_collection.name == "knowledge_delete_collection"
        args = knowledge_delete_collection.args
        assert "collection" in args

    @pytest.mark.asyncio
    async def test_delegates_to_impl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_result: dict[str, Any] = {
            "collection": "old",
            "existed": True,
            "deleted": True,
        }
        mock = AsyncMock(return_value=fake_result)
        monkeypatch.setattr(kt, "_delete_collection_impl", mock)

        result = await kt._delete_collection(collection="old")
        assert result == fake_result
        mock.assert_awaited_once_with(collection="old")
