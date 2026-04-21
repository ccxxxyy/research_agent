"""MCP Server — Sandboxed Python code execution for data analysis."""

from __future__ import annotations

import asyncio
import io
import contextlib
from typing import Any

from fastmcp import FastMCP

mcp = FastMCP("CodeExecutor")


@mcp.tool()
async def execute_python(code: str, timeout_seconds: int = 30) -> dict:
    """Execute Python code in a sandboxed environment.

    Useful for numerical analysis, data processing, and calculations.
    Common libraries available: math, statistics, json, collections, itertools.

    Args:
        code: Python code to execute.
        timeout_seconds: Maximum execution time in seconds.

    Returns:
        Dictionary with stdout output, return value, and any errors.
    """
    stdout_capture = io.StringIO()
    result: dict[str, Any] = {
        "stdout": "",
        "return_value": None,
        "error": None,
    }

    safe_globals: dict[str, Any] = {
        "__builtins__": {
            # --- Pure functions / data constructors ---
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
            # --- Exception hierarchy ---
            # Without these, *any* ``raise ValueError(...)`` in user code
            # fails with NameError before the real error even surfaces,
            # which is a trap for analyst scripts that routinely do
            # ``if df.empty: raise ValueError('no data')``.
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
            # --- Constants ---
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
