"""Unit tests for the native LangChain tools exposed to ReAct agents.

These tests validate three independent concerns:

1. **Schema correctness** — the ``@tool`` decorator must produce the
   expected name, description, and argument schema. The LLM relies on
   these fields to decide when and how to invoke each tool, so schema
   drift is a common source of silent Function Calling bugs.
2. **Behavioural correctness** — each tool must produce the right output
   for a representative set of inputs and edge cases.
3. **Safety** — ``calculate`` evaluates user-provided expressions; it
   must sandbox builtins and reject attempts to import modules, access
   attributes, or call arbitrary functions.
"""

from __future__ import annotations

import re

import pytest

from research_agent.tools.native import (
    DEFAULT_TOOLS,
    calculate,
    get_current_time,
    get_word_count,
)


class TestToolSchemas:
    """Verify LangChain produces correct schemas for Function Calling."""

    def test_default_toolset_contains_three_tools(self) -> None:
        names = {t.name for t in DEFAULT_TOOLS}
        assert names == {"get_current_time", "calculate", "get_word_count"}

    def test_get_current_time_schema(self) -> None:
        assert get_current_time.name == "get_current_time"
        assert "timezone" in get_current_time.description.lower()
        args = get_current_time.args
        assert "timezone_name" in args
        assert args["timezone_name"]["type"] == "string"
        assert args["timezone_name"].get("default") == "Asia/Shanghai"

    def test_calculate_schema(self) -> None:
        assert calculate.name == "calculate"
        assert "expression" in calculate.args
        assert calculate.args["expression"]["type"] == "string"
        assert "default" not in calculate.args["expression"]

    def test_get_word_count_schema(self) -> None:
        assert get_word_count.name == "get_word_count"
        assert "text" in get_word_count.args
        assert get_word_count.args["text"]["type"] == "string"


class TestGetCurrentTime:
    ISO_8601 = re.compile(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$"
    )

    def test_default_timezone_returns_iso8601(self) -> None:
        result = get_current_time.invoke({})
        assert self.ISO_8601.match(result), f"not ISO-8601: {result!r}"
        assert result.endswith("+08:00")

    def test_explicit_utc(self) -> None:
        result = get_current_time.invoke({"timezone_name": "UTC"})
        assert self.ISO_8601.match(result)
        assert result.endswith("+00:00")

    def test_explicit_ny(self) -> None:
        result = get_current_time.invoke({"timezone_name": "America/New_York"})
        assert self.ISO_8601.match(result)
        assert result.endswith(("-04:00", "-05:00"))

    def test_invalid_timezone_falls_back_to_utc(self) -> None:
        result = get_current_time.invoke({"timezone_name": "Not/AReal_Zone"})
        assert self.ISO_8601.match(result)
        assert result.endswith("+00:00")


class TestCalculate:
    @pytest.mark.parametrize(
        "expression,expected",
        [
            ("2 + 3", "5"),
            ("2 + 3 * 4", "14"),
            ("(100 - 5) / 19", "5.0"),
            ("2 ** 10", "1024"),
            ("abs(-42)", "42"),
            ("round(3.14159, 2)", "3.14"),
            ("min(1, 2, 3)", "1"),
            ("max(1, 2, 3)", "3"),
            ("pow(2, 8)", "256"),
        ],
    )
    def test_valid_arithmetic(self, expression: str, expected: str) -> None:
        assert calculate.invoke({"expression": expression}) == expected

    def test_math_functions(self) -> None:
        assert calculate.invoke({"expression": "sqrt(16)"}) == "4.0"
        result = calculate.invoke({"expression": "pi"})
        assert result.startswith("3.1415")

    def test_division_by_zero_is_captured(self) -> None:
        result = calculate.invoke({"expression": "1 / 0"})
        assert result.startswith("Error:")
        assert "ZeroDivisionError" in result

    def test_syntax_error_is_captured(self) -> None:
        result = calculate.invoke({"expression": "2 ++ ** 3"})
        assert result.startswith("Error:")
        assert "SyntaxError" in result

    @pytest.mark.parametrize(
        "evil_expression",
        [
            "__import__('os').system('ls')",
            "open('secrets.txt').read()",
            "globals()",
            "(1).__class__.__bases__",
            "exec('print(1)')",
            "eval('1+1')",
        ],
    )
    def test_security_blocks_forbidden_names(self, evil_expression: str) -> None:
        result = calculate.invoke({"expression": evil_expression})
        assert result.startswith("Error:"), (
            f"expected sandbox to reject {evil_expression!r}, got {result!r}"
        )


class TestGetWordCount:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("hello world", 2),
            ("one two three four five", 5),
            ("  spaced   out   text  ", 3),
            ("single", 1),
            ("", 0),
            ("\t\n  \t", 0),
            ("tab\tseparated\twords", 3),
            ("multi\nline\ntext here", 4),
        ],
    )
    def test_various_texts(self, text: str, expected: int) -> None:
        assert get_word_count.invoke({"text": text}) == expected

    def test_unicode_is_counted_by_whitespace(self) -> None:
        assert get_word_count.invoke({"text": "你好 世界 foo"}) == 3

    def test_returns_int_not_str(self) -> None:
        result = get_word_count.invoke({"text": "a b c"})
        assert isinstance(result, int)
