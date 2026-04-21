"""Phase-3: MCP ``code_server`` independent round-trip tests.

Why these tests matter
----------------------
Unlike ``echo_server`` (which has no real logic to validate), the
``code_server`` carries the actual sandboxing contract that
``coder_expert`` — and in Phase 4 the financial analyst — will rely
on. If the sandbox silently lets through ``open(...)``, or if an
exception does NOT surface as a structured ``error`` field, every
downstream agent will misbehave.

These tests therefore cover four independent guarantees:

1. Tool discovery: the MCP client sees ``code_execute_python``.
2. Happy path: stdout capture + ``result`` binding both work.
3. Sandbox denial: names outside the whitelist raise ``NameError``
   (checked for ``open`` and ``__import__``).
4. Runtime error capture: raising inside user code produces
   ``error`` rather than crashing the subprocess.

All tests spawn a real MCP subprocess via stdio — the same code path
used by the ``coder_expert`` specialist at runtime — so a pass here
means the full pipeline is wired correctly.
"""

from __future__ import annotations

import json

import pytest

from research_agent.mcp_servers.client_factory import (
    extract_text_content,
    load_code_server_tools,
)

CODE_TOOL_NAME = "code_execute_python"


def _parse_execute_result(raw: object) -> dict:
    """Decode the JSON payload wrapped in the MCP content block."""
    text = extract_text_content(raw)
    return json.loads(text)


@pytest.mark.asyncio
async def test_tool_is_discoverable() -> None:
    """MCP handshake succeeds and the expected tool is advertised."""
    tools = await load_code_server_tools()
    names = {t.name for t in tools}
    assert CODE_TOOL_NAME in names, (
        f"expected {CODE_TOOL_NAME!r} in discovered tool names, got {names!r}"
    )


@pytest.mark.asyncio
async def test_execute_python_stdout_and_return_value() -> None:
    """A trivial script roundtrips both ``stdout`` and ``result``."""
    tools = await load_code_server_tools()
    tool = next(t for t in tools if t.name == CODE_TOOL_NAME)

    code = "print(sum(range(10)))\nresult = 42"
    out = await tool.ainvoke({"code": code})
    payload = _parse_execute_result(out)

    assert payload["error"] is None, f"unexpected error field: {payload!r}"
    assert payload["stdout"].strip() == "45"
    assert payload["return_value"] == 42


@pytest.mark.asyncio
async def test_execute_python_captures_runtime_error() -> None:
    """An exception inside user code must be returned as ``error``,
    not raised to the caller."""
    tools = await load_code_server_tools()
    tool = next(t for t in tools if t.name == CODE_TOOL_NAME)

    code = "raise ValueError('boom')"
    out = await tool.ainvoke({"code": code})
    payload = _parse_execute_result(out)

    assert payload["error"] is not None
    assert "ValueError" in payload["error"]
    assert "boom" in payload["error"]


@pytest.mark.asyncio
async def test_sandbox_blocks_open_builtin() -> None:
    """``open`` is not in the safe-globals whitelist → NameError."""
    tools = await load_code_server_tools()
    tool = next(t for t in tools if t.name == CODE_TOOL_NAME)

    code = "open('/etc/passwd').read()"
    out = await tool.ainvoke({"code": code})
    payload = _parse_execute_result(out)

    assert payload["error"] is not None
    assert "NameError" in payload["error"]
    assert "open" in payload["error"]


@pytest.mark.asyncio
async def test_sandbox_blocks_dunder_import() -> None:
    """``__import__`` is also denied by the minimal safe-globals."""
    tools = await load_code_server_tools()
    tool = next(t for t in tools if t.name == CODE_TOOL_NAME)

    code = "os_mod = __import__('os')\nresult = os_mod.name"
    out = await tool.ainvoke({"code": code})
    payload = _parse_execute_result(out)

    assert payload["error"] is not None
    assert "NameError" in payload["error"]


@pytest.mark.asyncio
async def test_preimported_math_module_available() -> None:
    """``math`` is pre-injected into safe globals for numeric work."""
    tools = await load_code_server_tools()
    tool = next(t for t in tools if t.name == CODE_TOOL_NAME)

    code = "result = round(math.sqrt(2), 4)"
    out = await tool.ainvoke({"code": code})
    payload = _parse_execute_result(out)

    assert payload["error"] is None, payload
    assert payload["return_value"] == 1.4142
