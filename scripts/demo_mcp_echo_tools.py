"""通过 langchain-mcp-adapters 加载仓库内 MCP echo 服务器的工具。

从项目根目录运行::

    uv run python scripts/demo_mcp_echo_tools.py

本演示展示 MCP 集成路径：

    FastMCP server (stdio)  →  MultiServerMCPClient  →  LangChain BaseTool 列表

无网络调用 — echo 服务器完全本地运行。
"""

from __future__ import annotations

import asyncio
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient


def _extract_text(value: object) -> str:
    """展平新版 langchain-mcp-adapters 返回的内容块列表。"""
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
    print(f"从 MCP echo 服务器加载了 {len(tools)} 个工具:")
    for t in tools:
        print(f"  - {t.name}")

    upper_tool = next(t for t in tools if "echo_upper" in t.name)
    len_tool = next(t for t in tools if "echo_length" in t.name)

    u_raw = await upper_tool.ainvoke({"text": "phase three"})
    n_raw = await len_tool.ainvoke({"text": "hello"})

    u = _extract_text(u_raw)
    n = _extract_text(n_raw)

    print("\n结果:")
    print(f"  echo_upper('phase three') -> {u!r}  (raw={u_raw!r})")
    print(f"  echo_length('hello')      -> {n!r}  (raw={n_raw!r})")

    assert u == "PHASE THREE"
    assert int(n) == 5
    print("\n[PASS] MCP 工具往返调用成功。")


if __name__ == "__main__":
    asyncio.run(main())
