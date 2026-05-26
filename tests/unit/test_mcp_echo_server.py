"""MCP echo server 工具通过 stdio 加载并执行。"""

from __future__ import annotations

import sys

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient


def _extract_text(value: object) -> str:
    """将工具返回值规范化为纯字符串。

    ``langchain_mcp_adapters`` 可能返回标量（旧版本）或内容块列表，
    如 ``[{'type': 'text', 'text': 'ABC', ...}]``（新版本）。将两种形式展平以便用于断言。
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
    """FastMCP 装饰器的直接调用路径未暴露；仅做冒烟导入测试。"""
    from research_agent.mcp_servers import echo_server as echo_mod

    assert echo_mod.mcp.name == "Echo"
