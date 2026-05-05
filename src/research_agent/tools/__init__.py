"""In-process LangChain tools.

These tools use ``@tool`` / ``StructuredTool`` and run in the same
process as the agent. For out-of-process tools served over MCP-stdio,
see ``mcp_servers/`` (currently: ``code_server``, ``fin_data_server``,
``pdf_report_server``, ``echo_server``).

The knowledge-base tools are *defined* in
``mcp_servers/knowledge_server.py`` (which holds the MCP contract) but
*delivered* in-process via :mod:`research_agent.tools.knowledge_tools`
because their import chain (sentence-transformers / faiss-cpu / torch)
is incompatible with fastmcp's stdio transport on Windows + Python
3.13. See ``knowledge_server.py`` for the full diagnosis.
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
