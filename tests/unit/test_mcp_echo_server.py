"""Phase-3: MCP echo server tools load and execute via stdio."""

from __future__ import annotations

import sys

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient


def _extract_text(value: object) -> str:
    """Normalize a tool return into a plain string.

    ``langchain_mcp_adapters`` may return either a scalar (older versions) or
    a list of content blocks such as ``[{'type': 'text', 'text': 'ABC', ...}]``
    (newer versions). We flatten both shapes for assertions.
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


@pytest.mark.asyncio
async def test_mcp_echo_tools_round_trip() -> None:
    client = MultiServerMCPClient(
        {
            "echo": {
                "command": sys.executable,
                "args": ["-m", "research_agent.mcp_servers.echo_server"],
                "transport": "stdio",
            }
        },
        tool_name_prefix=True,
    )

    tools = await client.get_tools()
    assert len(tools) >= 2

    upper = next(t for t in tools if "echo_upper" in t.name)
    length = next(t for t in tools if "echo_length" in t.name)

    out_u = await upper.ainvoke({"text": "abc"})
    out_n = await length.ainvoke({"text": "abc"})

    assert _extract_text(out_u) == "ABC"
    assert int(_extract_text(out_n)) == 3


def test_echo_upper_tool_logic_sync() -> None:
    """Direct FastMCP-decorated call path is not exposed; smoke import only."""
    from research_agent.mcp_servers import echo_server as echo_mod

    assert echo_mod.mcp.name == "Echo"
