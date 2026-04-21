"""Factories for loading MCP tools into the LangGraph/LangChain runtime.

Why a factory module?
---------------------
``MultiServerMCPClient`` from ``langchain_mcp_adapters`` spawns a fresh
stdio subprocess each time a tool is invoked. We want one single place
that owns the subprocess-launch parameters (Python executable, module
path, transport) so every Agent builder / test / demo talks to the
exact same server surface. Cherry-picking ``sys.executable`` and ``-m``
paths at four different call sites would be a maintenance trap.

Usage
-----
Async (production / scripts / tests)::

    tools = await load_code_server_tools()
    supervisor = build_minimal_supervisor(
        model_router=router,
        coder_tools=tools,
    )

The returned tools are ``langchain_core.tools.BaseTool`` instances that
work with ``create_react_agent`` and ``langgraph_supervisor`` exactly
like locally-defined ``@tool`` functions would.
"""

from __future__ import annotations

import sys
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


def _stdio_server_spec(module: str) -> dict[str, Any]:
    """Build a stdio launch spec for an in-repo MCP server module.

    Using ``sys.executable`` guarantees the subprocess inherits the same
    virtualenv (and thus the same ``research_agent`` install) as the
    parent process. Using ``-m`` avoids hard-coded file paths that would
    break on CI / other checkouts.
    """
    return {
        "command": sys.executable,
        "args": ["-m", module],
        "transport": "stdio",
    }


CODE_SERVER_MODULE = "research_agent.mcp_servers.code_server"
ECHO_SERVER_MODULE = "research_agent.mcp_servers.echo_server"


async def load_code_server_tools() -> list[BaseTool]:
    """Spawn the ``code_server`` over stdio and return its tool list.

    Currently exposes one tool: ``code_execute_python`` (the
    ``tool_name_prefix=True`` flag prepends the server key ``code``).
    Callers should be prepared for the tool name to be prefixed.
    """
    client = MultiServerMCPClient(
        {"code": _stdio_server_spec(CODE_SERVER_MODULE)},
        tool_name_prefix=True,
    )
    return await client.get_tools()


async def load_echo_server_tools() -> list[BaseTool]:
    """Spawn the ``echo_server`` over stdio and return its tool list.

    Primarily used by the MCP smoke tests; production agents do not
    consume the echo tools.
    """
    client = MultiServerMCPClient(
        {"echo": _stdio_server_spec(ECHO_SERVER_MODULE)},
        tool_name_prefix=True,
    )
    return await client.get_tools()


def extract_text_content(value: object) -> str:
    """Flatten the content-block list returned by langchain-mcp-adapters.

    Newer versions of ``langchain_mcp_adapters`` (>=0.1) wrap every tool
    response in a list of content blocks shaped like
    ``[{'type': 'text', 'text': '...', 'id': '...'}]``. Older versions
    returned scalars directly. This helper normalizes both shapes to a
    plain string so downstream assertions don't need to know which
    version is installed.
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


__all__ = [
    "CODE_SERVER_MODULE",
    "ECHO_SERVER_MODULE",
    "extract_text_content",
    "load_code_server_tools",
    "load_echo_server_tools",
]
