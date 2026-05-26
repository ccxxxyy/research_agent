"""rag.query_rewriter 单元测试 — QueryRewriter。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from research_agent.rag.query_rewriter import QueryRewriter


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.ainvoke = AsyncMock()
    return model


class TestQueryRewriter:
    @pytest.mark.asyncio
    async def test_basic_rewrite(self, mock_model):
        mock_model.ainvoke.return_value = MagicMock(content="宁德时代 2023年报 电池出货量 ROE")
        rewriter = QueryRewriter(model=mock_model)
        result = await rewriter.rewrite("宁德时代 电池")
        assert result == "宁德时代 2023年报 电池出货量 ROE"
        mock_model.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_rewrite_with_context(self, mock_model):
        mock_model.ainvoke.return_value = MagicMock(content="improved query")
        rewriter = QueryRewriter(model=mock_model)
        result = await rewriter.rewrite(
            "ESG",
            context="Search returned only general ESG overviews, not company-specific data.",
        )
        assert result == "improved query"
        call_args = mock_model.ainvoke.call_args[0][0]
        assert "ESG" in call_args[1].content
        assert "Search returned only general" in call_args[1].content

    @pytest.mark.asyncio
    async def test_fallback_on_empty_response(self, mock_model):
        mock_model.ainvoke.return_value = MagicMock(content="")
        rewriter = QueryRewriter(model=mock_model)
        result = await rewriter.rewrite("original query")
        assert result == "original query"

    @pytest.mark.asyncio
    async def test_fallback_on_exception(self, mock_model):
        mock_model.ainvoke.side_effect = RuntimeError("LLM timeout")
        rewriter = QueryRewriter(model=mock_model)
        result = await rewriter.rewrite("original query")
        assert result == "original query"

    @pytest.mark.asyncio
    async def test_strips_whitespace(self, mock_model):
        mock_model.ainvoke.return_value = MagicMock(content="  trimmed query  \n")
        rewriter = QueryRewriter(model=mock_model)
        result = await rewriter.rewrite("old")
        assert result == "trimmed query"
