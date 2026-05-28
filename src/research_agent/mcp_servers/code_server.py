"""MCP Server — 用于数据分析的沙箱 Python 代码执行。

安全模型（双层隔离）
--------------------
1. 进程隔离（外层）：代码在独立子进程中执行，通过 ``subprocess.run``与主进程完全隔离。子进程拥有：
   - 独立的临时工作目录（执行完成后删除）
   - 严格的超时限制（默认 30 秒，超时则 SIGKILL）
   - 精简的环境变量（不继承主进程的密钥/路径）
   - 标准输出/错误流捕获，防止侧信道泄漏

2. API 白名单（内层）：子进程内执行的代码只能访问预先定义的安全
   全局变量和模块（math、statistics、json、collections、itertools）。
   ``__builtins__`` 被替换为白名单字典，阻止 ``open``、``import``、
   ``eval``、``__import__`` 等危险操作。

攻击面分析
----------
- 文件系统读写：``open`` 不在白名单中，且工作目录为空的临时目录。
- 网络访问：``socket``/``urllib``/``requests`` 无法导入（__import__ 不在白名单）。
- 进程逃逸：子进程使用 ``env={}`` 启动，没有 PATH，无法调用系统命令。
- 资源耗尽：timeout 强制终止；内层 exec 无法绕过外层的进程级超时。
- 内存炸弹：由操作系统 OOM killer 兜底（Windows 下无 rlimit，但子进程超时后会被 kill）。
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import subprocess
import sys
import tempfile
import textwrap
from typing import Any

from fastmcp import FastMCP

mcp = FastMCP("CodeExecutor")

_SANDBOX_RUNNER = textwrap.dedent("""\
    import sys, io, contextlib, json

    safe_globals = {
        "__builtins__": {
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
            "True": True,
            "False": False,
            "None": None,
        }
    }

    import math, statistics, json as json_mod, collections, itertools
    safe_globals.update({
        "math": math,
        "statistics": statistics,
        "json": json_mod,
        "collections": collections,
        "itertools": itertools,
    })

    code = sys.stdin.read()
    stdout_capture = io.StringIO()
    result = {"stdout": "", "return_value": None, "error": None}

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, safe_globals)
        result["stdout"] = stdout_capture.getvalue()
        result["return_value"] = safe_globals.get("result")
    except Exception as e:
        result["stdout"] = stdout_capture.getvalue()
        result["error"] = f"{type(e).__name__}: {e}"

    sys.stdout = sys.__stdout__
    print(json.dumps(result, ensure_ascii=False, default=str))
""")


@mcp.tool()
async def execute_python(code: str, timeout_seconds: int = 30) -> dict:
    """在沙箱子进程中执行 Python 代码。主工具，代码运行在一个完全独立的子进程中。

    双层隔离：进程级隔离（subprocess + 超时 + 环境清洗）+ API 白名单（受限 builtins，禁止 import/open/eval）。
    适用于数值分析、数据处理和计算。

    可用库：math、statistics、json、collections、itertools。
    将计算结果赋值给变量 ``result`` 即可在返回值中获取。

    subprocess.run([sys.executable, "-c", _SANDBOX_RUNNER], ...) — 新开独立 Python 进程
    cwd=tmpdir — 空临时目录
    env=sanitized_env — 不继承主进程的 API Key、PATH 等
    timeout=30 — 超时 kill 子进程
    用户代码通过 stdin 传给子进程内的 _SANDBOX_RUNNER

    Args:
        code: 要执行的 Python 代码。
        timeout_seconds: 最大执行时间（秒），超时强制终止子进程。

    Returns:
        包含 stdout、return_value 和 error 的字典。
    """
    timeout_seconds = max(1, min(timeout_seconds, 120))

    def _run_in_subprocess() -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="sandbox_") as tmpdir:
            sanitized_env = {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONIOENCODING": "utf-8",
            }
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", _SANDBOX_RUNNER],
                    input=code,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    cwd=tmpdir,
                    env=sanitized_env,
                )
            except subprocess.TimeoutExpired:
                return {
                    "stdout": "",
                    "return_value": None,
                    "error": f"TimeoutError: execution exceeded {timeout_seconds}s limit",
                }

            if proc.returncode != 0 and not proc.stdout.strip():
                error_msg = proc.stderr.strip() or f"Process exited with code {proc.returncode}"
                return {
                    "stdout": "",
                    "return_value": None,
                    "error": error_msg,
                }

            import json
            try:
                return json.loads(proc.stdout)
            except (json.JSONDecodeError, ValueError):
                return {
                    "stdout": proc.stdout,
                    "return_value": None,
                    "error": proc.stderr.strip() or None,
                }

    return await asyncio.get_event_loop().run_in_executor(None, _run_in_subprocess)


@mcp.tool()
async def execute_python_inproc(code: str, timeout_seconds: int = 30) -> dict:
    """进程内沙箱执行（仅白名单 builtins，无进程隔离）。

    备用工具，代码直接在主进程内部执行，当子进程执行不可用时的降级方案。安全性低于 ``execute_python``，但避免了子进程启动开销，适合可信来源的轻量计算。

    Args:
        code: 要执行的 Python 代码。
        timeout_seconds: 最大执行时间（秒）。

    Returns:
        包含 stdout、return_value 和 error 的字典。
    """
    stdout_capture = io.StringIO()
    result: dict[str, Any] = {
        "stdout": "",
        "return_value": None,
        "error": None,
    }

    safe_globals: dict[str, Any] = {
        "__builtins__": {
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
            "True": True,
            "False": False,
            "None": None,
        }
    }

    import collections
    import itertools
    import json
    import math
    import statistics

    safe_globals.update({
        "math": math,
        "statistics": statistics,
        "json": json,
        "collections": collections,
        "itertools": itertools,
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
    except TimeoutError:
        result["error"] = f"TimeoutError: execution exceeded {timeout_seconds}s limit"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
