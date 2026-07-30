"""``client_factory`` MCP 硬化：Connection closed 识别与重试包装。"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.tools import StructuredTool

from research_agent.mcp_servers import client_factory as cf


def test_is_mcp_connection_error_nested_group() -> None:
    inner = RuntimeError("McpError: Connection closed")
    # 模拟 anyio/langgraph 嵌套 ExceptionGroup
    group = ExceptionGroup("unhandled errors in a TaskGroup", [inner])
    outer = ExceptionGroup("unhandled errors in a TaskGroup", [group])
    assert cf._is_mcp_connection_error(outer) is True
    assert cf._is_mcp_connection_error(ValueError("other")) is False


@pytest.mark.asyncio
async def test_harden_mcp_tool_retries_then_returns_error() -> None:
    calls = {"n": 0}

    async def _flaky(**_kwargs):
        calls["n"] += 1
        raise ExceptionGroup(
            "unhandled errors in a TaskGroup",
            [RuntimeError("Connection closed")],
        )

    raw = StructuredTool.from_function(
        coroutine=_flaky,
        name="us_filing_search_filings",
        description="test",
    )
    hardened = cf._harden_mcp_tool(raw)
    out = await hardened.ainvoke({})
    assert calls["n"] == cf._MCP_RETRY_ATTEMPTS
    assert isinstance(out, dict)
    assert "Connection closed" in out["error"]
    assert out["context"] == "us_filing_search_filings"


@pytest.mark.asyncio
async def test_harden_mcp_tool_succeeds_on_retry() -> None:
    calls = {"n": 0}

    async def _sometimes(**_kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("Connection closed")
        return {"ok": True, "n": calls["n"]}

    raw = StructuredTool.from_function(
        coroutine=_sometimes,
        name="us_filing_search_filings",
        description="test",
    )
    hardened = cf._harden_mcp_tool(raw)
    out = await hardened.ainvoke({})
    assert out == {"ok": True, "n": 2}


@pytest.mark.asyncio
async def test_harden_mcp_tool_parent_hard_timeout() -> None:
    async def _hang(**_kwargs):
        await asyncio.sleep(60)

    raw = StructuredTool.from_function(
        coroutine=_hang,
        name="sentiment_get_stock_sentiment_report",
        description="test",
    )
    hardened = cf._harden_mcp_tool(raw)
    # 临时压低超时便于单测
    old = cf._MCP_CALL_TIMEOUT_BY_PREFIX.get("sentiment_")
    cf._MCP_CALL_TIMEOUT_BY_PREFIX["sentiment_"] = 0.2
    try:
        out = await hardened.ainvoke({})
    finally:
        if old is None:
            cf._MCP_CALL_TIMEOUT_BY_PREFIX.pop("sentiment_", None)
        else:
            cf._MCP_CALL_TIMEOUT_BY_PREFIX["sentiment_"] = old
    assert isinstance(out, dict)
    assert "超时" in out["error"] or "timeout" in out["error"].lower()


@pytest.mark.asyncio
async def test_harden_error_matches_content_and_artifact() -> None:
    """MCP 工具默认 content_and_artifact；错误路径必须返回二元组，否则 ToolNode 崩图。"""

    async def _flaky(**_kwargs):
        raise RuntimeError("Connection closed")

    raw = StructuredTool.from_function(
        coroutine=_flaky,
        name="fin_get_index_quotes",
        description="test",
        response_format="content_and_artifact",
    )
    hardened = cf._harden_mcp_tool(raw)
    # 直接调 coroutine，验证原始返回形
    content, artifact = await hardened.coroutine()
    assert isinstance(content, str)
    assert "Connection closed" in content
    assert artifact["structured_content"]["context"] == "fin_get_index_quotes"

    # 带 tool_call_id 的 ainvoke 不得再抛 two-tuple ValueError
    msg = await hardened.ainvoke(
        {"type": "tool_call", "id": "call-1", "name": "fin_get_index_quotes", "args": {}}
    )
    assert msg.status == "success" or "Connection closed" in str(msg.content)


def test_stdio_server_spec_disables_fastmcp_banner() -> None:
    spec = cf._stdio_server_spec(cf.US_FILING_SERVER_MODULE)
    env = spec["env"]
    assert env["FASTMCP_SHOW_SERVER_BANNER"] == "false"
    assert env["FASTMCP_LOG_ENABLED"] == "false"
    assert env["FASTMCP_CHECK_FOR_UPDATES"] == "off"
    assert env["NO_PROXY"] == "*"
