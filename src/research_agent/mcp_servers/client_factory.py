"""将 MCP 工具加载到 LangGraph/LangChain 运行时的工厂模块。

是"工具加载器"——它帮助启动 MCP 子进程，拿到工具列表，然后就把工具传给专家去用。
流程就三步：
调 await load_fin_data_server_tools() → 启动 fin_data_server 子进程 → 拿到 5 个工具
把工具传给 build_data_expert(router, tools) → 创建专家
专家就能用这些工具了

为什么需要工厂模块？
--------------------
``langchain_mcp_adapters`` 的 ``MultiServerMCPClient`` 每次调用工具时都会生成一个新的 stdio 子进程。
希望有一个统一的位置来管理子进程启动参数（Python 可执行文件、模块路径、传输方式），以便每个 Agent 构建器 / 测试 / 演示都使用完全相同的服务器接口。
在四个不同的调用点各自拼凑``sys.executable`` 和 ``-m`` 路径会成为维护陷阱。工厂模块的解决办法：把启动参数集中写在一个地方（_stdio_server_spec），4 个脚本都调这一个函数。改了一处，处处生效

用法
----
异步（生产 / 脚本 / 测试）::

    tools = await load_code_server_tools()
    supervisor = build_minimal_supervisor(
        model_router=router,
        coder_tools=tools,
    )

返回的工具是 ``langchain_core.tools.BaseTool`` 实例，可与``create_react_agent`` 和 ``langgraph_supervisor`` 配合使用，效果与本地定义的 ``@tool`` 函数完全相同。
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from langchain_mcp_adapters.client import MultiServerMCPClient

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


def _stdio_server_spec(module: str) -> dict[str, Any]:
    """为仓库内的 MCP 服务器模块构建 stdio 启动规格。

    使用 ``sys.executable`` 可确保子进程继承与父进程相同的虚拟环境（从而使用相同的 ``research_agent`` 安装）。
    使用 ``-m`` 可避免在 CI / 其他检出路径中失效的硬编码文件路径。
    """
    return {
        "command": sys.executable,
        "args": ["-m", module],
        "transport": "stdio",
    }


CODE_SERVER_MODULE = "research_agent.mcp_servers.code_server"
ECHO_SERVER_MODULE = "research_agent.mcp_servers.echo_server"
FIN_DATA_SERVER_MODULE = "research_agent.mcp_servers.fin_data_server"
PDF_REPORT_SERVER_MODULE = "research_agent.mcp_servers.pdf_report_server"
KNOWLEDGE_SERVER_MODULE = "research_agent.mcp_servers.knowledge_server"
NEWS_SERVER_MODULE = "research_agent.mcp_servers.news_server"
NEWS_SENTIMENT_SERVER_MODULE = "research_agent.mcp_servers.news_sentiment_server"


async def load_code_server_tools() -> list[BaseTool]:
    """通过 stdio 启动 ``code_server`` 并返回其工具列表。

    当前暴露一个工具：``code_execute_python``（``tool_name_prefix=True``标志会在工具名前添加服务器键 ``code``）。调用者应准备好工具名会带前缀。
    """
    client = MultiServerMCPClient(
        {"code": _stdio_server_spec(CODE_SERVER_MODULE)},
        tool_name_prefix=True,
    )
    return await client.get_tools()


async def load_echo_server_tools() -> list[BaseTool]:
    """通过 stdio 启动 ``echo_server`` 并返回其工具列表。

    主要用于 MCP 冒烟测试；生产环境的 Agent 不使用 echo 工具。
    """
    client = MultiServerMCPClient(
        {"echo": _stdio_server_spec(ECHO_SERVER_MODULE)},
        tool_name_prefix=True,
    )
    return await client.get_tools()


async def load_fin_data_server_tools() -> list[BaseTool]:
    """通过 stdio 启动 ``fin_data_server`` 并返回其工具列表。

    暴露五个 A 股数据工具，每个通过 ``tool_name_prefix=True`` 添加
    ``fin_`` 前缀：

    - ``fin_get_stock_basic_info``
    - ``fin_get_stock_price_history``
    - ``fin_get_financial_abstract``
    - ``fin_get_financial_indicators``
    - ``fin_search_stock_by_name``

    首次调用 MCP 子进程会加载 ``akshare``，其内部延迟导入可能耗时约 1 秒；后续工具调用复用已预热的进程。
    """
    client = MultiServerMCPClient(
        {"fin": _stdio_server_spec(FIN_DATA_SERVER_MODULE)},
        tool_name_prefix=True,
    )
    return await client.get_tools()


async def load_news_server_tools() -> list[BaseTool]:
    """通过 stdio 启动 ``news_server`` 并返回其工具列表。

    暴露五个 A 股新闻工具，每个通过 ``tool_name_prefix=True`` 添加
    ``news_`` 前缀：

    - ``news_get_stock_news``        — 东方财富个股新闻。
    - ``news_get_market_telegraph``  — 财联社实时快讯。
    - ``news_get_hot_keywords``      — 东方财富热搜关键词。
    - ``news_get_economic_news``     — 百度财经早晚报摘要。
    - ``news_get_xueqiu_discussion_hot_rank`` — 雪球讨论热度个股榜
      （``akshare.stock_hot_tweet_xq``）。

    与 ``fin_data_server`` 类似，子进程在首次工具调用时延迟导入 ``akshare``约 1 秒，后续调用复用已预热的解释器。
    """
    client = MultiServerMCPClient(
        {"news": _stdio_server_spec(NEWS_SERVER_MODULE)},
        tool_name_prefix=True,
    )
    return await client.get_tools()


async def load_news_sentiment_server_tools() -> list[BaseTool]:
    """通过 stdio 启动 ``news_sentiment_server`` 并返回其工具列表。

    暴露两个情感分析工具，每个通过 ``tool_name_prefix=True`` 添加
    ``sentiment_`` 前缀：

    - ``sentiment_analyze_text_sentiment`` — 纯文本评分：传入字符串列表，返回逐条情感分数 + 聚合统计。不依赖外部数据源。
    - ``sentiment_get_stock_sentiment_report`` — 一站式：获取东方财富个股新闻 → 逐条评分 → 返回结构化报告，含逐条分数 + 聚合 + 审计元数据（模型版本、文本指纹、时间戳）。

    子进程在首次工具调用时延迟导入 ``snownlp`` + ``akshare``，后续调用复用已预热的解释器。
    """
    client = MultiServerMCPClient(
        {"sentiment": _stdio_server_spec(NEWS_SENTIMENT_SERVER_MODULE)},
        tool_name_prefix=True,
    )
    return await client.get_tools()


async def load_pdf_report_server_tools() -> list[BaseTool]:
    """通过 stdio 启动 ``pdf_report_server`` 并返回其工具列表。

    暴露四个公告 PDF 工具，每个通过 ``tool_name_prefix=True`` 添加
    ``pdf_`` 前缀：

    - ``pdf_search_announcements`` — 列出 cninfo 公告，附带预推导的``pdf_url`` 字段。
    - ``pdf_download_pdf`` — 感知缓存的下载，存入 ``./data/pdf_cache/``。
    - ``pdf_parse_pdf_pages`` — 有界页码范围文本提取（每次调用最多 20 页）。
    - ``pdf_extract_pdf_metadata`` — 页数 / 标题 / 作者 / 文件大小。

    子进程在工具调用时延迟导入 ``pypdf`` 和 ``httpx``，因此即使冷启动也能快速启动。
    """
    client = MultiServerMCPClient(
        {"pdf": _stdio_server_spec(PDF_REPORT_SERVER_MODULE)},
        tool_name_prefix=True,
    )
    return await client.get_tools()


async def load_knowledge_server_tools() -> list[BaseTool]:
    """**已弃用** — 仅为向后兼容而保留。

    ``knowledge_server`` 的 MCP-stdio 传输路径在 Windows + Python 3.13上不稳定：
    成功执行 ``ingest_pdf`` 后，fastmcp 的 stdout 写入器无法将 JSON-RPC 响应刷新回父进程，见 ``knowledge_server.py``模块文档字符串中的完整排查记录。

    生产代码应改为调用 :func:`load_knowledge_tools_inproc`，它返回相同的四个工具、相同的 ``knowledge_*`` 名称，
    但以进程内方式调用（无子进程、无 JSON-RPC 帧、无 stdio 管道）。

    保留此函数是为了不破坏旧脚本 / 测试，但使用时会立即抛出异常，使弃用状态不可能被忽略。load_knowledge_tools_inproc()是替代方案。
    """
    raise RuntimeError(
        "load_knowledge_server_tools (MCP-stdio) is deprecated; "
        "use load_knowledge_tools_inproc() instead. The stdio path "
        "is known to deadlock on Windows + Python 3.13 with the "
        "FAISS/sentence-transformers import chain — see "
        "knowledge_server.py for the diagnosis."
    )


async def load_knowledge_tools_inproc() -> list[BaseTool]:
    """以进程内 LangChain 工具形式返回四个知识库工具。

    与（旧版）MCP-stdio 加载器产出的工具集完全相同——相同名称、相同参数形状、相同返回字典，
    但通过 ``research_agent.tools.knowledge_tools.KNOWLEDGE_TOOLS`` 而非 MCP子进程交付。
    签名保留为 ``async`` 以便调用者可以无缝切换加载器。

    返回的工具（每个都是 ``StructuredTool``）：

    - ``knowledge_ingest_pdf``      — 加载 → 分块 → 嵌入 → 写入``./data/knowledge_db/`` 下的持久化 FAISS 集合。
    - ``knowledge_search``          — 混合检索（向量 + BM25），含纠正式 RAG 质量信号（``quality`` ∈ ``{high, medium, low}``，加上每条命中的分数）。
    - ``knowledge_list_collections`` — 列举已持久化的集合。
    - ``knowledge_delete_collection`` — 幂等的清理操作。

    开销特征：导入此加载器会拉入 ``langchain_text_splitters`` + ``faiss-cpu``（首次约 9 秒，之后有缓存）。
    bge-small 嵌入模型在首次 ``ingest_pdf`` /``search`` 调用时才加载（冷启动约 3–17 秒，预热后亚秒级）。

    从 research_agent.tools.knowledge_tools 导入 4 个预先定义好的工具函数（knowledge_ingest_pdf、knowledge_search、knowledge_list_collections、knowledge_delete_collection），然后返回它们的列表。
    和 MCP 方式的区别：MCP 方式是启动一个子进程，通过管道通信拿到工具；这个函数是直接在当前进程里导入 Python 函数——效果一样，但不走子进程。


    """
    from research_agent.tools.knowledge_tools import KNOWLEDGE_TOOLS

    return list(KNOWLEDGE_TOOLS)


def extract_text_content(value: object) -> str:
    """展平 langchain-mcp-adapters 返回的内容块列表。

    较新版本的 ``langchain_mcp_adapters``（>=0.1）将每个工具响应包装为形如 ``[{'type': 'text', 'text': '...', 'id': '...'}]`` 的内容块列表。
    旧版本直接返回标量值。此辅助函数将两种形状统一为纯字符串，使下游断言无需关心安装的是哪个版本。

    旧版的 langchain-mcp-adapters：工具调用返回的结果是普通字符串 "hello"
    新版的 langchain-mcp-adapters（>=0.1）：返回结果变成了列表 [{"type": "text", "text": "hello"}]
    extract_text_content 函数的作用就是：不管你装的是旧版还是新版，它都统一转成纯字符串。这样下游代码不需要关心装了哪个版本的库。
    """
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(value)


__all__ = [
    "CODE_SERVER_MODULE",
    "ECHO_SERVER_MODULE",
    "FIN_DATA_SERVER_MODULE",
    "KNOWLEDGE_SERVER_MODULE",
    "NEWS_SENTIMENT_SERVER_MODULE",
    "NEWS_SERVER_MODULE",
    "PDF_REPORT_SERVER_MODULE",
    "extract_text_content",
    "load_code_server_tools",
    "load_echo_server_tools",
    "load_fin_data_server_tools",
    "load_knowledge_server_tools",
    "load_knowledge_tools_inproc",
    "load_news_sentiment_server_tools",
    "load_news_server_tools",
    "load_pdf_report_server_tools",
]
