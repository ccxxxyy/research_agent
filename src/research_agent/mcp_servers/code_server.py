"""MCP Server — 用于数据分析的沙箱 Python 代码执行。"""

from __future__ import annotations

import asyncio
import io
import contextlib
from typing import Any

from fastmcp import FastMCP

mcp = FastMCP("CodeExecutor")


@mcp.tool()
async def execute_python(code: str, timeout_seconds: int = 30) -> dict:
    """在沙箱环境中执行 Python 代码。

    适用于数值分析、数据处理和计算。
    可用常用库：math、statistics、json、collections、itertools。

    Args:
        code: 要执行的 Python 代码。
        timeout_seconds: 最大执行时间（秒）。

    Returns:
        包含标准输出、返回值和错误信息的字典。
    """
    stdout_capture = io.StringIO()
    result: dict[str, Any] = {
        "stdout": "",
        "return_value": None,
        "error": None,
    }

    safe_globals: dict[str, Any] = {
        "__builtins__": {
            # --- 纯函数/数据构造函数 ---
            "print": print,
            "range": range,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "sorted": sorted,
            "reversed": reversed,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "any": any,
            "all": all,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "frozenset": frozenset,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "type": type,
            "isinstance": isinstance,
            "repr": repr,
            "hash": hash,
            # --- 异常层次结构 ---
            # 没有这些，用户代码中任何 ``raise ValueError(...)`` 都会在真正的错误浮出之前抛出 NameError，
            # 沙箱环境中 __builtins__ 是自定义的白名单。如果白名单里不包含 ValueError 这个名字，那么当 LLM 生成的代码写了 raise ValueError("no data") 时，Python 不会报 ValueError: no data，而是会报 NameError: name 'ValueError' is not defined——因为沙箱里根本没定义 ValueError 这个词。这对经常写``if df.empty: raise ValueError('no data')`` 的分析脚本是个陷阱。
            "Exception": Exception,
            "BaseException": BaseException,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "AttributeError": AttributeError,
            "ZeroDivisionError": ZeroDivisionError,
            "ArithmeticError": ArithmeticError,
            "RuntimeError": RuntimeError,
            "StopIteration": StopIteration,
            # --- 常数 ---
            "True": True,
            "False": False,
            "None": None,
        }
    }

    import math
    import statistics
    import json
    import collections

    safe_globals.update({
        "math": math,
        "statistics": statistics,
        "json": json,
        "collections": collections,
    })

    def _run_code() -> None:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, safe_globals)  # noqa: S102

    try:
        await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _run_code),
            timeout=timeout_seconds,
        )
        result["stdout"] = stdout_capture.getvalue()
        result["return_value"] = safe_globals.get("result")
    except asyncio.TimeoutError:
        result["error"] = f"TimeoutError: execution exceeded {timeout_seconds}s limit"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
