"""Native LangChain tools for in-process Function Calling.

These tools use the `@tool` decorator and run in the same process as the agent.
For out-of-process tools served over MCP protocol, see ``mcp_servers/``.
"""

from research_agent.tools.native import (
    calculate,
    get_current_time,
    get_word_count,
    DEFAULT_TOOLS,
)

__all__ = [
    "calculate",
    "get_current_time",
    "get_word_count",
    "DEFAULT_TOOLS",
]
