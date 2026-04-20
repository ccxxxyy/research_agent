"""Built-in native tools demonstrating LangChain's Function Calling.

Each tool is decorated with ``@tool`` and uses type hints + docstrings
for automatic schema generation. The LLM reads these schemas to decide
when and how to invoke each tool.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from langchain_core.tools import tool
from loguru import logger


@tool
def get_current_time(timezone_name: str = "Asia/Shanghai") -> str:
    """Return the current date and time for a given IANA timezone.

    Use this tool when the user asks about the current time, today's date,
    or needs time-aware reasoning.

    Args:
        timezone_name: IANA timezone name, e.g. "Asia/Shanghai", "UTC",
            "America/New_York". Defaults to "Asia/Shanghai".

    Returns:
        ISO-8601 formatted timestamp string.
    """
    logger.debug("Tool call: get_current_time(timezone={})", timezone_name)
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = timezone.utc
    now = datetime.now(tz)
    return now.isoformat(timespec="seconds")


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression and return the numeric result.

    Supports +, -, *, /, //, %, **, parentheses, and standard math functions.
    Do NOT pass arbitrary Python code — only mathematical expressions.

    Args:
        expression: Mathematical expression, e.g. "2 + 3 * 4" or "(100 - 5) / 19".

    Returns:
        String representation of the numerical result, or an error message.
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
    """Count the number of whitespace-separated words in the given text.

    Args:
        text: The input text to analyze.

    Returns:
        Integer word count.
    """
    logger.debug("Tool call: get_word_count(text_len={})", len(text))
    return len(text.split())


DEFAULT_TOOLS = [get_current_time, calculate, get_word_count]
"""The default toolset exposed to single-agent smoke tests."""
