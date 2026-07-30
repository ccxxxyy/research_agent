"""``client_factory`` MCP 硬化：Connection closed 识别与重试包装。"""

from __future__ import annotations

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


def test_stdio_server_spec_disables_fastmcp_banner() -> None:
    spec = cf._stdio_server_spec(cf.US_FILING_SERVER_MODULE)
    env = spec["env"]
    assert env["FASTMCP_SHOW_SERVER_BANNER"] == "false"
    assert env["FASTMCP_LOG_ENABLED"] == "false"
    assert env["FASTMCP_CHECK_FOR_UPDATES"] == "off"
    assert env["NO_PROXY"] == "*"
