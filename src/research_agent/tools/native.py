"""内置原生工具，演示 LangChain 的 Function Calling。

每个工具使用 ``@tool`` 装饰器，并通过类型提示 + 文档字符串自动生成 Schema。LLM 读取这些 Schema 来决定何时以及如何调用每个工具。

3 个简单内置工具：calculate、get_current_time、get_word_count
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from langchain_core.tools import tool
from loguru import logger


@tool
def get_current_time(timezone_name: str = "Asia/Shanghai") -> str:
    """返回指定 IANA 时区的当前日期和时间。

    当用户询问当前时间、今天的日期或需要时间感知推理时使用此工具。

    Args:
        timezone_name: IANA 时区名称，例如 "Asia/Shanghai"、"UTC"、 "America/New_York"。默认为 "Asia/Shanghai"。

    Returns:
        ISO-8601 格式的时间戳字符串。
    """
    logger.debug("Tool call: get_current_time(timezone={})", timezone_name)
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = UTC
    now = datetime.now(tz)
    return now.isoformat(timespec="seconds")


@tool
def calculate(expression: str) -> str:
    """计算简单算术表达式并返回数值结果。

    支持 +、-、*、/、//、%、**、括号以及标准数学函数。请勿传入任意 Python 代码——仅接受数学表达式。

    Args:
        expression: 数学表达式，例如 "2 + 3 * 4" 或 "(100 - 5) / 19"。

    Returns:
        数值结果的字符串表示，或错误消息。
    """
    logger.debug("Tool call: calculate(expression={!r})", expression)

    allowed_names: dict[str, object] = {
        "abs": abs, "round": round, "min": min, "max": max,
        "pow": pow, "sum": sum,
    }

    import math
    for name in ("sqrt", "log", "log2", "log10", "exp", "sin", "cos", "tan", "pi", "e"):
        allowed_names[name] = getattr(math, name)

    try:
        code = compile(expression, "<calc>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                return f"Error: use of '{name}' is not allowed"
        result = eval(code, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return str(result)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError) as e:
        return f"Error: {type(e).__name__}: {e}"


@tool
def get_word_count(text: str) -> int:
    """统计给定文本中以空白字符分隔的单词数量。

    Args:
        text: 待分析的输入文本。

    Returns:
        整数形式的单词数。
    """
    logger.debug("Tool call: get_word_count(text_len={})", len(text))
    return len(text.split())


DEFAULT_TOOLS = [get_current_time, calculate, get_word_count]
"""暴露给单 Agent 冒烟测试的默认工具集。"""
