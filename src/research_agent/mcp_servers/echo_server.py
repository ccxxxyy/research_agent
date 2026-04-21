"""Minimal MCP server (stdio) for Phase-3 MCP integration demos.

Run directly:
    uv run python -m research_agent.mcp_servers.echo_server

Or as a subprocess target for :class:`langchain_mcp_adapters.client.MultiServerMCPClient`.

This server intentionally avoids network calls so demos work offline.
"""

from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("Echo")


@mcp.tool()
def echo_upper(text: str) -> str:
    """Return ``text`` converted to upper case.

    Args:
        text: Arbitrary string to echo.
    """
    return text.upper()


@mcp.tool()
def echo_length(text: str) -> int:
    """Return the character length of ``text`` (not word count).

    Args:
        text: Input string.
    """
    return len(text)


if __name__ == "__main__":
    mcp.run(transport="stdio")
