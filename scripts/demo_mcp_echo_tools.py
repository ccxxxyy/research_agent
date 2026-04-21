"""Load tools from the in-repo MCP echo server via langchain-mcp-adapters.

Run from the project root:
    uv run python scripts/demo_mcp_echo_tools.py

This demonstrates the Phase-3 MCP integration path:

    FastMCP server (stdio)  →  MultiServerMCPClient  →  LangChain BaseTool list

No network calls — the echo server is fully local.
"""

from __future__ import annotations

import asyncio
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient


def _extract_text(value: object) -> str:
    """Flatten the content-block list returned by new langchain-mcp-adapters."""
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(value)


async def main() -> None:
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
    print(f"Loaded {len(tools)} tools from MCP echo server:")
    for t in tools:
        print(f"  - {t.name}")

    upper_tool = next(t for t in tools if "echo_upper" in t.name)
    len_tool = next(t for t in tools if "echo_length" in t.name)

    u_raw = await upper_tool.ainvoke({"text": "phase three"})
    n_raw = await len_tool.ainvoke({"text": "hello"})

    u = _extract_text(u_raw)
    n = _extract_text(n_raw)

    print("\nResults:")
    print(f"  echo_upper('phase three') -> {u!r}  (raw={u_raw!r})")
    print(f"  echo_length('hello')      -> {n!r}  (raw={n_raw!r})")

    assert u == "PHASE THREE"
    assert int(n) == 5
    print("\n[PASS] MCP tool round-trip succeeded.")


if __name__ == "__main__":
    asyncio.run(main())
