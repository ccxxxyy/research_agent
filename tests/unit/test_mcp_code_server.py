"""Phase-3：MCP ``code_server`` 独立往返测试。

为什么这些测试很重要
----------------------
与 ``echo_server``（没有实际逻辑需要验证）不同，``code_server``承载着 ``coder_expert``（以及金融分析师）所依赖的实际沙箱契约。
如果沙箱静默放行 ``open(...)``，或者异常未能以结构化的``error`` 字段呈现，所有下游 Agent 都会出现异常行为。

因此这些测试覆盖四项独立保证：

1. 工具发现：MCP 客户端能看到 ``code_execute_python``。
2. 正常路径：stdout 捕获和 ``result`` 绑定均正常工作。
3. 沙箱拒绝：不在白名单中的名称抛出 ``NameError``（针对 ``open`` 和 ``__import__`` 进行检查）。
4. 运行时错误捕获：用户代码中的异常以 ``error`` 形式返回， 而非导致子进程崩溃。

所有测试通过 stdio 启动真实的 MCP 子进程——与 ``coder_expert``专家在运行时使用的代码路径相同——因此测试通过意味着整个流水线连接正确。
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
    """解码 MCP 内容块中包装的 JSON 载荷。"""
    text = extract_text_content(raw)
    return json.loads(text)


@pytest.mark.asyncio
async def test_tool_is_discoverable() -> None:
    """MCP 握手成功且预期工具已被发布。"""
    tools = await load_code_server_tools()
    names = {t.name for t in tools}
    assert CODE_TOOL_NAME in names, (
        f"expected {CODE_TOOL_NAME!r} in discovered tool names, got {names!r}"
    )


@pytest.mark.asyncio
async def test_execute_python_stdout_and_return_value() -> None:
    """一个简单脚本同时完成 ``stdout`` 和 ``result`` 的往返验证。"""
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
    """用户代码中的异常必须以 ``error`` 形式返回，而非向调用者抛出异常。"""
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
    """``open`` 不在安全全局变量白名单中 → NameError。"""
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
    """``__import__`` 同样被最小安全全局变量所拒绝。"""
    tools = await load_code_server_tools()
    tool = next(t for t in tools if t.name == CODE_TOOL_NAME)

    code = "os_mod = __import__('os')\nresult = os_mod.name"
    out = await tool.ainvoke({"code": code})
    payload = _parse_execute_result(out)

    assert payload["error"] is not None
    assert "NameError" in payload["error"]


@pytest.mark.asyncio
async def test_preimported_math_module_available() -> None:
    """``math`` 已预注入到安全全局变量中以支持数值计算。"""
    tools = await load_code_server_tools()
    tool = next(t for t in tools if t.name == CODE_TOOL_NAME)

    code = "result = round(math.sqrt(2), 4)"
    out = await tool.ainvoke({"code": code})
    payload = _parse_execute_result(out)

    assert payload["error"] is None, payload
    assert payload["return_value"] == 1.4142
