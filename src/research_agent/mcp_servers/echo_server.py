"""最小化 MCP Server（stdio），用于 MCP 集成演示。

直接运行：
    uv run python -m research_agent.mcp_servers.echo_server

    或作为 :class:`langchain_mcp_adapters.client.MultiServerMCPClient` 的子进程目标。

本服务器刻意不进行网络调用，以确保演示可离线运行。
"""

from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("Echo")


@mcp.tool()
def echo_upper(text: str) -> str:
    """返回将 ``text`` 转换为大写后的结果。

    Args:
        text: 任意待回显的字符串。
    """
    return text.upper()


@mcp.tool()
def echo_length(text: str) -> int:
    """返回 ``text`` 的字符长度（非单词数）。

    Args:
        text: 输入字符串。
    """
    return len(text)


if __name__ == "__main__":
    mcp.run(transport="stdio")
