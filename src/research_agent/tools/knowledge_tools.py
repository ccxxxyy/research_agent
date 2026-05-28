"""知识库 RAG 平面的进程内 LangChain 工具。

本模块存在的原因
----------------------
四个知识库工具（``ingest_pdf``、``search``、``list_collections``、``delete_collection``）定义在:mod:`research_agent.mcp_servers.knowledge_server` 中，
因为该模块同时充当规范的 MCP 契约,其文档字符串、校验规则和 JSON 返回结构具有权威性。

然而在运行时，以进程内 LangChain 工具的方式交付它们，而非通过 MCP-stdio 子进程。
原因在 ``knowledge_server.py`` 中有详细记录
（简而言之：在 Windows + Python 3.13 上，fastmcp 的 stdio JSON-RPC 写入器与 ``sentence-transformers`` / ``torch`` / ``faiss`` 引入的重量级导入链存在冲突，
导致 ``ingest_pdf`` 完成工作后 JSON-RPC 响应静默地无法到达父进程）。

遇到的问题：知识库工具需要导入 sentence-transformers（嵌入模型）、torch（深度学习框架）、faiss（向量索引）。这些库非常重，导入时会做很多底层操作。
在 Windows + Python 3.13 上，这些重量级导入和 fastmcp 的 stdio 传输产生了冲突 ——导入 PDF 完成后，JSON-RPC 的响应莫名其妙地丢了，父进程永远收不到回复。

本模块是桥接层：它从 ``knowledge_server`` 获取四个 ``@mcp.tool`` 装饰的协程，并以与 MCP 路径相同的前缀名称（``knowledge_*``）将其重新暴露为
``langchain_core.tools.BaseTool`` 实例。
这样 ``KNOWLEDGE_EXPERT_PROMPT``无需修改即可工作——无论工具以进程内还是跨进程方式交付，Agent 看到的工具集完全一致。

为何这样做是安全的
----------------
fastmcp 3.x 的 ``@mcp.tool()`` 装饰器会将函数注册到 FastMCP 实例上，
但不会改变函数本身（``type(ingest_pdf)`` 仍是 ``types.FunctionType``，无 ``.fn`` 垫片）。
因此直接import并调用它们与 MCP 传输层调用的代码路径完全一致，包括用于保护事件循环不被阻塞的 ``asyncio.to_thread`` 跳转。

返回结构
---------------
每个底层协程返回一个 ``dict``（成功时为结果，校验失败时为``{error, context}``）。LangChain 工具运行时将该 dict 序列化为 JSON供 LLM 使用。
提示词和 corrective-RAG 循环已读取该 JSON 结构，因此无需适配层。

知识库工具的逻辑写在 mcp_servers/knowledge_server.py 里，但因为 Windows + Python 3.13 上 fastmcp 的 stdio 传输和 torch/faiss 的重量级导入链有冲突（JSON-RPC 响应会丢失），所以不能通过 MCP 子进程调用。解决方案是：把 knowledge_server.py 里的函数直接 import 进来，包装成 LangChain 工具，在同一进程内调用。
解决方案：不走 MCP 子进程了，直接在主进程里调用这些函数。knowledge_tools.py 就是这个"桥接层"。


knowledge_server.py 定义了真正的逻辑（工具的逻辑、校验、返回格式、搜索、导入PDF等）
         │
         │  直接 import 函数
         ▼
knowledge_tools.py 桥接层：把这些函数包装成 LangChain 工具格式） 第一层：async 包装器（可加横切逻辑）
         │
         │  传入 StructuredTool.from_function()
         ▼
knowledge_tools.py 第二层：变成 LangChain BaseTool 对象
         │
         │  放入列表
         ▼
knowledge_tools.py 第三层：KNOWLEDGE_TOOLS 列表
         │
         │  暴露给 Agent ，被 knowledge_expert Agent 使用，调用这些工具
         ▼
Agent 运行时：LLM 看到 4 个工具的名字和描述 → 决定调哪个 → 调用对应函数


为什么要包装一层而不是直接用？
文档可以重写：MCP 的文档字符串是给 MCP 协议看的，桥接层可以写更适合 LLM 阅读的描述
统一扩展点：以后想加限流、审计日志、缓存，改这一个文件就行，不用动 MCP 模块
名字前缀对齐：工具统一命名为 knowledge_search、knowledge_ingest_pdf 等，和 MCP 路径生成的前缀一致，Agent 的提示词不需要任何修改


工具命名为 knowledge_search、knowledge_ingest_pdf 等。这和 MCP 路径生成的前缀一致（如果走 MCP，工具名自动变成 knowledge_search）。
这样 Agent 的提示词里写的工具名不需要任何修改——无论工具是通过 MCP 子进程还是进程内直接调用交付的。
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from research_agent.mcp_servers.knowledge_server import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
)
from research_agent.mcp_servers.knowledge_server import (
    delete_collection as _delete_collection_impl,
)
from research_agent.mcp_servers.knowledge_server import (
    ingest_pdf as _ingest_pdf_impl,
)
from research_agent.mcp_servers.knowledge_server import (
    list_collections as _list_collections_impl,
)
from research_agent.mcp_servers.knowledge_server import (
    search as _search_impl,
)

# ---------------------------------------------------------------------
# 轻量异步包装器
#
# StructuredTool 需要一个可等待对象，将 LLM 提供的参数映射到底层协程。
# 虽然可以直接传入导入的协程，但通过包装器可以获得以下好处：
#   1. 一个稳定的位置，用提示词友好的语言为每个工具重新编写文档（MCP 的文档字符串是权威的，但部分措辞针对 LLM 消费者做了调整）。
#   2. 一个横切关注点的汇聚点（限流、审计日志、缓存），无需修改 MCP 模块或 Agent 提示词。


# 直接调用 knowledge_server.py 里的同名函数，一行包装。
# 为什么不直接用原函数？ 两个原因：可以重写文档字符串——原函数的文档是给 MCP 协议看的，包装层可以写更适合 LLM 阅读的版本；将来加限流、审计日志、缓存，只需要改这一层，不动 MCP 模块
# ---------------------------------------------------------------------
async def _ingest_pdf(
    local_path: str,
    collection: str = "default",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    """将单个本地 PDF 导入到持久化 FAISS 集合中。

    Args:
        local_path: ``.pdf`` 文件的文件系统路径（通常是``pdf_download_pdf`` 返回的路径）。路径必须已存在；此工具不会自行创建文件。
        collection: 目标集合名称（首次使用时自动创建）。名称须匹配 ``[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]``，长度 3–63 个字符。
        chunk_size: 每个分块的字符数。默认值 800 针对 bge-small-zh 嵌入模型的 512-token 窗口进行了调优。
        chunk_overlap: 相邻分块之间的滑动字符数。

    Returns:
        成功时返回 ``{collection, source, num_pages, num_chunks_added,total_chunks_in_collection}``，
        校验失败时返回``{error, context}``。
    """
    return await _ingest_pdf_impl(
        local_path=local_path,
        collection=collection,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


async def _search(
    query: str,
    collection: str = "default",
    top_k: int = 5,
) -> dict[str, Any]:
    """对已导入的集合执行混合（向量 + BM25）检索。

    响应包含顶层 ``quality``（``high``/``medium``/``low``）以及每条命中的 ``vector_score`` / ``bm25_score`` / ``rrf_score``，
    供 Agent驱动 corrective-RAG 循环：当 ``quality == "low"`` 时，Agent 应重写查询（拆分复合问题、添加领域关键词、替换代词）并再次调用此工具，每轮用户对话最多三次尝试。

    Args:
        query: 自由格式的自然语言问题（中文或英文；bge-small-zh嵌入模型支持双语）。
        collection: 要检索的集合。空集合或不存在的集合将返回``quality='low'`` 并附带 ``warning`` 字段而非抛出异常——可安全探测。
        top_k: 融合后返回的最大命中数（上限 20）。

    Returns:
        ``{collection, query, top_k_returned, quality, top_score, mean_score, unique_sources, results: [...]}``。
    """
    return await _search_impl(
        query=query,
        collection=collection,
        top_k=top_k,
    )


async def _list_collections() -> dict[str, Any]:
    """列出当前持久化在磁盘上的所有 FAISS 集合。

    当用户暗示了某个知识库但未指定集合名称时（例如"我的 ESG 知识库里有什么？"），适合作为 Agent 的首次调用。每个条目包含 ``name`` 和 ``chunk_count``；
    如果 FAISS 文件对不可读则 chunk_count 为 ``-1``。

    Returns:
        ``{db_dir, collections: [{name, chunk_count}]}``。
    """
    return await _list_collections_impl()


async def _delete_collection(collection: str) -> dict[str, Any]:
    """删除集合及其内存缓存。幂等操作。

    仅在用户明确要求清除集合时使用。不存在的集合不会抛出异常,响应仅报告 ``existed=False``。

    Returns:
        ``{collection, existed, deleted}``。
    """
    return await _delete_collection_impl(collection=collection)


# ---------------------------------------------------------------------
# StructuredTool 导出。
#
# ``knowledge_`` 前缀与 MCP-stdio 加载器（``MultiServerMCPClient``）
# 通过 ``tool_name_prefix=True`` 配合 ``"knowledge"`` 服务器键生成的前缀一致。
# 这样 Agent 的 KNOWLEDGE_EXPERT_PROMPT——已将工具命名为 ``knowledge_ingest_pdf`` 等,无需修改即可工作。

# 把上面的 async def _search 包装成 LangChain 能识别的 BaseTool 对象。
# LangChain 的 Agent 不直接调用 Python 函数，它需要一个 BaseTool 对象，对象上有 name（LLM 看到的工具名）、description（LLM 读这个决定什么时候用这个工具）、args_schema（参数类型和校验规则，从类型提示自动生成）。
# ---------------------------------------------------------------------
knowledge_ingest_pdf: BaseTool = StructuredTool.from_function(
    coroutine=_ingest_pdf,
    name="knowledge_ingest_pdf",
    description=(
        "将单个本地 PDF 导入持久化 FAISS 知识库集合。"
        "参数：local_path (str)，collection "
        '(str，默认 "default")，chunk_size (int，默认 800)，'
        "chunk_overlap (int，默认 120)。返回包含 num_chunks_added 的字典，或错误信息。"
    ),
)

knowledge_search: BaseTool = StructuredTool.from_function(
    coroutine=_search,
    name="knowledge_search",
    description=(
        "对已导入的知识库集合执行混合（向量 + BM25）检索。"
        "参数：query (str)，collection (str，默认 "
        '"default")，top_k (int，默认 5)。返回包含 '
        "顶层 quality（'high'/'medium'/'low'）、top_score、mean_score 和 results 列表的字典。"
        "利用 quality 信号驱动 corrective-RAG 重试循环（quality 为 'low' 时 重写查询并重试，每轮最多 3 次）。"
    ),
)

knowledge_list_collections: BaseTool = StructuredTool.from_function(
    coroutine=_list_collections,
    name="knowledge_list_collections",
    description=(
        "列出当前持久化在磁盘上的所有 FAISS 集合。"
        "返回 {db_dir, collections: [{name, chunk_count}]}。"
        "当用户暗示某个知识库但未指定集合名称时，适合首先调用。"
    ),
)

knowledge_delete_collection: BaseTool = StructuredTool.from_function(
    coroutine=_delete_collection,
    name="knowledge_delete_collection",
    description=(
        "删除知识库集合（清理维护）。幂等操作：不存在的集合返回 existed=False 而非抛出异常。参数：collection (str)。仅在用户明确要求清除集合时使用。"
    ),
)


KNOWLEDGE_TOOLS: list[BaseTool] = [
    knowledge_ingest_pdf,
    knowledge_search,
    knowledge_list_collections,
    knowledge_delete_collection,
]
"""进程内知识库工具的规范清单。

把 4 个工具打包成一个列表，方便其他代码一行导入。
将此列表传递给 ``build_knowledge_expert(model_router, mcp_tools)``（参数名是历史遗留——它接受任意 ``Sequence[BaseTool]``，不关心交付机制）。"""


__all__ = [
    "KNOWLEDGE_TOOLS",
    "knowledge_delete_collection",
    "knowledge_ingest_pdf",
    "knowledge_list_collections",
    "knowledge_search",
]
