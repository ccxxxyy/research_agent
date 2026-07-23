"""工具结果缓存（缓存 MCP 原始返回值，而非 LLM 回答）。"""

from research_agent.cache.tool_cache import (
    TTL_DAILY,
    TTL_LONG,
    TTL_MEDIUM,
    TTL_REALTIME,
    TTL_SHORT,
    ToolResultCache,
    cached_tool,
    get_tool_cache,
    reset_tool_cache_for_tests,
)

__all__ = [
    "TTL_DAILY",
    "TTL_LONG",
    "TTL_MEDIUM",
    "TTL_REALTIME",
    "TTL_SHORT",
    "ToolResultCache",
    "cached_tool",
    "get_tool_cache",
    "reset_tool_cache_for_tests",
]
