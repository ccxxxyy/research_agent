"""进程内 LangChain 工具。

这些工具使用 ``@tool`` / ``StructuredTool`` 装饰器，与 Agent 运行在同一进程中。
跨进程通过 MCP-stdio 提供的工具见 ``mcp_servers/``（目前包括：``code_server``、``fin_data_server``、``pdf_report_server``、``news_server``、``news_sentiment_server``、``echo_server``）。

知识库工具定义在 ``mcp_servers/knowledge_server.py``（持有 MCP 契约），但通过 :mod:`research_agent.tools.knowledge_tools` 以进程内方式交付，
原因是其导入链（sentence-transformers / faiss-cpu / torch）与 fastmcp 在 Windows + Python 3.13 上的 stdio 传输不兼容。见 ``knowledge_server.py`` 中的完整诊断。
"""

from research_agent.tools.knowledge_tools import (
    KNOWLEDGE_TOOLS,
    knowledge_delete_collection,
    knowledge_ingest_pdf,
    knowledge_list_collections,
    knowledge_search,
)
from research_agent.tools.native import (
    DEFAULT_TOOLS,
    calculate,
    get_current_time,
    get_word_count,
)

__all__ = [
    "DEFAULT_TOOLS",
    "KNOWLEDGE_TOOLS",
    "calculate",
    "get_current_time",
    "get_word_count",
    "knowledge_delete_collection",
    "knowledge_ingest_pdf",
    "knowledge_list_collections",
    "knowledge_search",
]
