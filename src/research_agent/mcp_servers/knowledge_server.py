"""知识库 RAG — 工具定义 +（已弃用的）MCP-stdio 接口。

这是金融研究 Agent 的 知识层。
``pdf_report_server`` 处理公开披露PDF（巨潮资讯），``fin_data_server`` 处理公开市场数据（akshare），
而本模块允许用户上传自有 PDF 库（ESG 报告、券商研报、内部备忘录、招股说明书）并对其进行自由提问。

运行时模型：进程内工具，非 MCP 子进程
---------------------------------------------------
历史上本模块作为 MCP-stdio 子进程启动，以便父进程 ``MultiServerMCPClient``通过与 ``code_server`` / ``fin_data_server`` / ``pdf_report_server``
相同的协议发现工具。该路径不再是生产路径：

* 在 Windows + Python 3.13 上，fastmcp 的 stdio JSON-RPC 写入器与
  ``sentence-transformers`` / ``torch`` / ``faiss-cpu`` 引入的重度 import 链交互异常。
  ``ingest_pdf`` 完成工作后 JSON-RPC 响应始终无法到达父进程（静默 stdout 管道停滞）。
  诊断结论："进程内 35-40 秒可用，MCP 子进程即使配合 FAISS 且禁用 stdout 防火墙也永久挂起"。
* 跨语言/跨进程的协议价值本项目并未实际使用 — 四个 Agent 和 supervisor全部运行在同一个 Python 进程中。

因此下方四个 ``@mcp.tool`` 装饰的协程被保留为知识库能力的规范契约（其文档字符串、校验规则和返回形状具有权威性），
但``research_agent.tools.knowledge_tools`` 将它们重新导出为普通``langchain_core.tools.tool`` 装饰函数，供 ``knowledge_expert`` 进程内消费。
``@mcp.tool()`` 装饰器将函数注册到 FastMCP 实例但不改变函数本身（已验证 ``type(ingest_pdf) is types.FunctionType``），因此直接调用安全。

如果需要跨进程交付（如将其接入 Rust Agent），可恢复 MCP-stdio 启动路径；。

为何单独服务器（而非扩展 ``pdf_report_server``）（无状态/有状态）？
-------------------------------------------------------------
``pdf_report_server`` 是无状态的：从 cninfo 搜索派生 PDF URL，下载，解析有限页范围，返回文本。没有 索引 概念 — LLM 负责查找。

知识库服务器是有状态的：PDF 被分块、嵌入并一次性索引到持久化 FAISS 集合；
后续搜索命中混合（向量 + BM25）检索器，LLM 从不直接看原始 PDF。
不同的生命周期、不同的持久化方式、不同的存储成本特征 — 它们应属于不同进程。

暴露的4个工具(搜索、导入 PDF、列出集合、删除集合)
-------------
1. ``knowledge_ingest_pdf`` — 加载 → 分块 → 嵌入 → 写入 FAISS。返回 ``(collection, num_chunks_added, total_chunks_in_collection)``。
2. ``knowledge_search`` — 混合检索（向量 + BM25 上的 RRF），每个命中标注 ``source`` / ``page`` / ``vector_score`` / ``bm25_rank`` / ``rrf_rank``。
   响应还提供纠正式 RAG 信号 （``top_score``、``mean_score``、``unique_sources``、``quality``）以便调用 Agent 决定是否重写查询并重试。
3. ``knowledge_list_collections`` — 枚举持久化存储中当前集合及分块数。
4. ``knowledge_delete_collection`` — 清理；幂等操作。

纠正循环为何在 AGENT 中而非工具中（纠正式 RAG 循环的设计决策）
--------------------------------------------------------
在工具内部放置重写/重试循环会导致：
  (a) 需要在此子进程中内置 LLM 重写器，使凭据/网络攻击面翻倍，或
  (b) 硬编码基于规则的重写，缺乏灵活性。

取而代之的是，工具返回丰富的质量信号，``knowledge_expert`` 系统提示教导 Agent：
"如果 ``top_score < 0.4`` 或 ``quality == 'low'``，用更具体的关键词 REWRITE 查询并再次调用 ``knowledge_search``，最多 3 次尝试"。
这使纠正循环在 LangGraph 跟踪中以 knowledge_expert 子图内重复的``AIMessage → ToolMessage`` 周期可见 — 规范的 Corrective-RAG 模式。

存储布局
--------------
research_agent/data/knowledge_db\
├── my_esg_reports/           ← 一个"集合"（你上传的 ESG 报告）
│   ├── index.faiss           ← 向量索引文件（二进制，存向量）
│   └── index.pkl             ← 文档存储（pickle 格式，存原文+元数据）
├── annual_reports/           ← 另一个集合（你上传的年报）
│   ├── index.faiss
│   └── index.pkl
└── （每上传一类 PDF 就多一个文件夹）

``./data/knowledge_db/<collection_name>/`` 每个集合持有一个持久化 FAISS 索引。
每个子文件夹包含标准 LangChain FAISS 文件对：``index.faiss``（二进制索引）加 ``index.pkl``（docstore +``faiss_id -> doc_id`` 映射）。
因此列出集合只需枚举基路径的子目录（./data/knowledge_db/每个集合一个目录）。
BM25 在进程启动后首次搜索时（或集合被导入修改后）从持久化 FAISS docstore 在内存中重建，将开销分摊到会话的其余部分。

为何用 FAISS 而非 Chroma（Chroma 破坏 stdio 管道）？
~~~~~~~~~~~~~~~~~~~~~~
先尝试了 Chroma。在 Windows 上，``chromadb`` 的 import 链会生成 posthog 遥测守护线程，这些线程发出 ``print()`` 调用，破坏了 MCP-stdioJSON-RPC 通道。
FAISS 是纯 C++ + Python 绑定，无遥测、无后台线程，且持久化模型更简单（每个集合目录一个 ``index.faiss`` + ``index.pkl``）,在此规模下更合适。
即使 Chroma → FAISS 迁移后，stdio 路径在 Windows上仍不稳定，促使转向上文记录的进程内交付方式。

嵌入模型
---------------
本地 HuggingFace ``BAAI/bge-small-zh-v1.5`` — 双语（中英），体积小（约 100MB），免费，无需 API 密钥。首次调用下载权重（缓存于 ``~/.cache/huggingface``）；
后续进程启动即时可用。
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# 默认静默环境，用于抑制 ML 库的噪声输出。
#
# 尽管生产运行时已改为进程内（``research_agent.tools.knowledge_tools``），仍在模块导入时设置这些环境变量：
#
#   * ``transformers`` / ``tqdm`` / ``huggingface_hub`` 否则会向父进程的 stdout 输出进度条和 "BertModel LOAD REPORT" 横幅，
#   当父进程为 FastAPI worker 时很不合适（它们会混入请求日志流）。
#   * ``TOKENIZERS_PARALLELISM=false`` 静默 sentence-transformers 在 asyncio worker 线程中启动时的已知 fork 警告。
#   * ``HF_HUB_DISABLE_TELEMETRY=1`` 只是良好的卫生习惯。
#
# 这些 ``setdefault`` 调用永远不会覆盖运维人员在 shell 中显式设置的值。
# ---------------------------------------------------------------------
import os as _os

_os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
_os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
_os.environ.setdefault("TQDM_DISABLE", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import asyncio  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from fastmcp import FastMCP  # noqa: E402

# 在模块加载时预先导入重量级依赖（langchain text-splitter + FAISS 包装器），
# 而非在首次工具调用时。在进程内路径上，这只是将约 9 秒的 langchain-core导入前置到 worker 启动，使第一个 ``ingest_pdf`` 请求不会异常缓慢。
from langchain_community.vectorstores import FAISS as _PrewarmedFAISS  # noqa: E402, F401, N811
from langchain_text_splitters import (  # noqa: E402, F401
    RecursiveCharacterTextSplitter as _PrewarmedSplitter,
)
from loguru import logger  # noqa: E402

mcp = FastMCP("KnowledgeBase")

# ---------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------
DEFAULT_DB_DIR = Path("./data/knowledge_db").resolve()
"""持久化知识库根目录。

首次导入时为每个集合创建一个子目录。每个子目录包含一个 LangChain-FAISS 索引对（``index.faiss`` + ``index.pkl``）。
模块级以便测试可 monkey-patch，且子进程无论启动进程的 CWD 如何都继承相同路径。
"""

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
"""默认嵌入模型。本地、免费、双语。"""

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120

MAX_INGEST_BYTES = 50 * 1024 * 1024  # 50 MB safety bound for a single PDF
MAX_TOP_K = 20
"""每次搜索调用 ``top_k`` 的硬上限，防止 LLM 在 10 万分块的集合上请求 ``top_k=10000`` — 那将返回足以撑爆 LLM 上下文窗口的文本量。
"""

# 质量分类器阈值。针对 BAAI/bge-small-zh-v1.5 的归一化余弦相似度校准；
# 更换模型时需调整。
QUALITY_HIGH_THRESHOLD = 0.65
QUALITY_MEDIUM_THRESHOLD = 0.40

RERANK_OVERFETCH_MULTIPLIER = 3
"""重排序前从 RRF 多取的额外候选数量倍数。

cross-encoder 在有余量重排序时最有用：向 bi-encoder + BM25 请求``top_k * 3`` 个候选并在重排序后裁剪，通常每次查询可提升 1-2 个仅靠 RRF 会被埋没的结果。
使用较小倍数以控制每次调用延迟 ——见 :class:`CrossEncoderReranker` 自身的 ``max_pairs`` 保护作为第二层防护。
"""


def _reranker_enabled() -> bool:
    """在调用时读取 ``KNOWLEDGE_RERANKER_ENABLED`` 环境变量。

    每次调用重新读取（不在 import 时缓存），以便单测可通过 monkeypatch ``os.environ`` 在不同用例间切换开关而无需重新导入模块。
    """
    raw = _os.environ.get("KNOWLEDGE_RERANKER_ENABLED", "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


# 延迟单例：在首次 ``_maybe_rerank`` 调用时构建。
# 保持 import 延迟意味着从不请求重排序的进程（冒烟测试、工具）不必在模块加载时支付约 1 秒的 sentence_transformers import 开销。
_RERANKER: Any | None = None

# ---------------------------------------------------------------------
# 延迟模块级缓存
#
# 三个缓存在 MCP 子进程的整个生命周期内保持存活。
# 构建代价高（模型加载约 3 秒；FAISS 加载 + BM25 重建 = O(语料库大小)）而复用代价低。
# ---------------------------------------------------------------------
_EMBEDDER: Any | None = None
"""HuggingFaceEmbeddings 单例。首次使用前为 ``None``。"""

_FAISS_STORES: dict[str, Any] = {}
"""``collection_name -> langchain_community.vectorstores.FAISS`` 缓存。

每个集合在内存中保持一个预热的 FAISS，以便连续搜索不必重新读取索引文件。
每次成功导入后缓存会被清除/重新加载（FAISS 不支持并发原地修改）。
"""

_BM25_CACHE: dict[str, Any] = {}
"""``collection_name -> 内存 BM25 索引`` 缓存。

每次导入后通过 ``_invalidate_bm25(collection)`` 使其失效。
"""


def _fmt_error(exc: Exception, *, context: str) -> dict[str, Any]:
    """标准错误格式 — 抛出异常会终止 MCP 子进程。"""
    return {"error": f"{type(exc).__name__}: {exc}", "context": context}


def _validate_collection_name(name: str) -> None:
    """校验集合名称。

    每个集合对应磁盘上的一个目录名，因此限制为 ``[a-zA-Z0-9._-]``（与早期迭代使用的 Chroma 规则一致，以便现有测试夹具中的集合名继续可用）且长度 3–63 字符。
    额外禁止前导点和 ``..`` 以防止路径 遍历风险。
    """
    if not (3 <= len(name) <= 63):
        raise ValueError(f"collection name length must be 3..63, got {len(name)}")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]", name):
        raise ValueError(
            f"collection name {name!r} must match "
            r"[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]"
        )
    if ".." in name:
        raise ValueError(f"collection name {name!r} must not contain '..'")


# ---------------------------------------------------------------------
# 嵌入器 & FAISS 辅助函数（延迟加载）
# ---------------------------------------------------------------------
def _get_embedder() -> Any:
    """返回单例嵌入器，首次调用时构建。

    import 是延迟的，即使 sentence-transformers 需要下载模型权重，模块加载也很快 —
     约 3 秒的开销在首次 ``knowledge_ingest_pdf`` / ``knowledge_search`` 时支付，而非在 import 时。

    标准输出安全：聊天式 ML 库（``transformers``、``tqdm``、 ``huggingface_hub``）已通过模块导入时设置的环境变量静默（见本文件顶部的 env-var 块），
    因此不需要内联 ``redirect_stdout``包装器，即使 ``transformers`` 历史上在权重加载期间会打印"BertModel LOAD REPORT" 横幅。
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Loading embedding model: {}", DEFAULT_EMBEDDING_MODEL)
        _EMBEDDER = HuggingFaceEmbeddings(
            model_name=DEFAULT_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBEDDER


def _collection_dir(collection: str, *, db_dir: Path | None = None) -> Path:
    """返回 ``collection`` 的持久化路径。

    通过这一辅助函数统一解析路径，模块其余部分无需拼写约定 （``DEFAULT_DB_DIR / collection``）。
    测试可 monkey-patch``DEFAULT_DB_DIR``，此函数会自动使用。
    """
    base = db_dir or DEFAULT_DB_DIR
    return base / collection


def _faiss_index_exists(collection: str, *, db_dir: Path | None = None) -> bool:
    """当且仅当 ``collection`` 的已保存 FAISS 文件对存在于磁盘时返回 True。

    LangChain 的 ``FAISS.save_local(path)`` 始终同时写入``index.faiss`` 和 ``index.pkl``；任一缺失都表示写入不完整/损坏，
    此时将该集合视为不存在（而非半加载后以模糊方式失败）。
    """
    cdir = _collection_dir(collection, db_dir=db_dir)
    return (cdir / "index.faiss").exists() and (cdir / "index.pkl").exists()


def _load_faiss_store(collection: str, *, db_dir: Path | None = None) -> Any | None:
    """加载（并缓存）``collection`` 的 FAISS 存储，未导入过则返回 None。

    从未导入过的集合返回 ``None``（而非抛异常）。调用者须处理此情况 ——搜索工具将其转为 ``quality='low'`` 的空响应，导入工具将其视为 "创建"分支。

    ``allow_dangerous_deserialization=True`` 是必需的，因为 LangChain 的 FAISS 附属文件是 ``pickle`` 格式。
    本仓库只加载自己写入``DEFAULT_DB_DIR`` 的文件，从不加载来自互联网的不可信数据，因此 审计风险范围仅限于"已能写入数据目录的攻击者" —— 到那时候 pickle 已是最不值得担心的。
    """
    cached = _FAISS_STORES.get(collection)
    if cached is not None:
        return cached
    if not _faiss_index_exists(collection, db_dir=db_dir):
        return None

    from langchain_community.vectorstores import FAISS

    cdir = _collection_dir(collection, db_dir=db_dir)
    store = FAISS.load_local(
        folder_path=str(cdir),
        embeddings=_get_embedder(),
        allow_dangerous_deserialization=True,
    )
    _FAISS_STORES[collection] = store
    return store


def _save_faiss_store(collection: str, store: Any, *, db_dir: Path | None = None) -> None:
    """持久化 ``store`` 并原子刷新内存缓存。

    始终在 ``save_local`` 成功后才更新 ``_FAISS_STORES[collection]``，
    因此写入失败（如磁盘已满）时内存缓存仍指向磁盘上的先前状态 ——绝不会出现仅存在于内存中而磁盘上没有的幽灵存储。
    """
    cdir = _collection_dir(collection, db_dir=db_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    store.save_local(folder_path=str(cdir))
    _FAISS_STORES[collection] = store


# ---------------------------------------------------------------------
# 内容哈希去重
# ---------------------------------------------------------------------
_HASH_FILENAME = ".ingested_hashes.json"


def _load_ingested_hashes(collection: str, *, db_dir: Path | None = None) -> dict[str, str]:
    """加载集合的已导入文件哈希注册表。

    返回 ``{sha256_hex: source_path}`` 映射。文件不存在时返回空字典。
    """
    cdir = _collection_dir(collection, db_dir=db_dir)
    hash_file = cdir / _HASH_FILENAME
    if not hash_file.exists():
        return {}
    try:
        return json.loads(hash_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_ingested_hash(
    collection: str, file_hash: str, source_path: str, *, db_dir: Path | None = None
) -> None:
    """将新的文件哈希追加到集合的注册表中。"""
    cdir = _collection_dir(collection, db_dir=db_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    hashes = _load_ingested_hashes(collection, db_dir=db_dir)
    hashes[file_hash] = source_path
    (cdir / _HASH_FILENAME).write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _compute_file_hash(path: Path) -> str:
    """计算文件的 SHA-256 哈希值。"""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def _invalidate_bm25(collection: str) -> None:
    """清除 ``collection`` 的 BM25 缓存；下次搜索时重建。"""
    _BM25_CACHE.pop(collection, None)


# ---------------------------------------------------------------------
# BM25 索引 — 委托给 rag.retriever.BM25Index
# ---------------------------------------------------------------------
from research_agent.rag.retriever import BM25Index as _BM25Index  # noqa: E402


def _build_bm25_for_collection(collection: str) -> _BM25Index:
    """从 FAISS docstore 中当前所有文档构建 BM25。

    LangChain 的 FAISS 将文档对象保存在 ``store.docstore._dict`` 中，
    ``faiss_id -> doc_id`` 映射在 ``store.index_to_docstore_id`` 中。
    遍历 docstore 的值以确定性插入顺序产出文档，这对 BM25 已足够（BM25 索引不需要与 FAISS 的内部编号对齐）。
    对于典型用户库（10–1000 个分块）速度很快（<200 ms）。对于非常大的集合需要分页，留作 TODO，因为 Agent 很少会处理 >10k 个个人 PDF 分块。

    集合未导入时返回空索引 — 调用者须容忍 ``search`` /``BM25Index.search`` 返回零结果（搜索工具已处理 — FAISS 存储缺失时直接返回 ``quality='low'`` 响应）。
    """
    store = _load_faiss_store(collection)
    docs: list[dict[str, Any]] = []
    if store is None:
        return _BM25Index(docs)
    docstore = store.docstore
    for doc_id in store.index_to_docstore_id.values():
        doc = docstore.search(doc_id)
        if doc is None or isinstance(doc, str):
            continue
        docs.append(
            {
                "content": getattr(doc, "page_content", "") or "",
                "metadata": dict(getattr(doc, "metadata", None) or {}),
            }
        )
    return _BM25Index(docs)


def _get_bm25(collection: str) -> _BM25Index:
    """带缓存的 BM25 获取；缓存未命中/失效时延迟重建。"""
    bm25 = _BM25_CACHE.get(collection)
    if bm25 is None:
        bm25 = _build_bm25_for_collection(collection)
        _BM25_CACHE[collection] = bm25
    return bm25


# ---------------------------------------------------------------------
# PDF loading + chunking 加载+分块处理
# ---------------------------------------------------------------------
def _load_pdf_pages(local_path: Path) -> list[dict[str, Any]]:
    """返回用 pypdf 提取的逐页记录 ``[{page, text}]``。

    特意将文档的创建操作放在纯字典中进行，非使用 ``langchain_core.documents.Document`` ，这样这个辅助函数就可以在不涉及 LangChain API 接口的情况下进行单元测试。
    """
    import pypdf

    with local_path.open("rb") as fh:
        reader = pypdf.PdfReader(fh)
        out: list[dict[str, Any]] = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            out.append({"page": i, "text": text})
    return out


def _chunk_pages(
    pages: list[dict[str, Any]],
    *,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    """将每页文本切分为 ``chunk_size`` 字符的窗口。

    每个分块继承 ``page`` 和 ``source``，以便最终回答可以忠实引用 ``"source.pdf p.42"``。
    保留页面边界（分块不跨页）—— 这会对跨页句子损失少量召回，但使引用无歧义，这对金融 RAG 更有价值， 因为用户会核对页码。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
        length_function=len,
    )
    chunks: list[dict[str, Any]] = []
    for page in pages:
        for piece in splitter.split_text(page["text"] or ""):
            chunks.append(
                {
                    "content": piece,
                    "metadata": {"source": source, "page": page["page"]},
                }
            )
    return chunks


# ---------------------------------------------------------------------
# 工具 1：将 PDF 导入集合
# ---------------------------------------------------------------------
@mcp.tool()
async def ingest_pdf(
    local_path: str,
    collection: str = "default",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    """将单个 PDF 导入持久化知识库。

    PDF 用 ``pypdf`` 读取一次（页面保留为溯源单位），切分为 ``chunk_size``字符的窗口并有 ``chunk_overlap`` 字符的滑动，用 bge-small 中文模型嵌入，
    然后写入 FAISS 集合（``DEFAULT_DB_DIR`` 下每个集合一个文件夹）。基于文件 SHA-256 哈希去重：相同内容的 PDF 不会重复导入，返回 ``skipped=True``。

    Args:
        local_path: ``.pdf`` 文件的绝对或相对路径。通常为``pdf_download_pdf``（服务器）返回的路径，以便两个工具自然串联。
        collection: 目标集合名。首次使用时创建。须匹配 ``[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]``且长度 3..63 字符。
        chunk_size: 每个分块的字符数。800 对 bge-small-zh 的 512 token 窗口是合理默认值（中文约 1 字符/token，英文平均约 0.25 token/字符）。
        chunk_overlap: 相邻分块的滑动量。经验法则为 ``chunk_size`` 的 15%。

    Returns:
        成功时：``{collection, source, num_pages, num_chunks_added, total_chunks_in_collection}``。
        失败时：``{error, context}``。
    """
    try:
        _validate_collection_name(collection)
    except ValueError as e:
        return _fmt_error(e, context=f"ingest_pdf(collection={collection!r})")

    path = Path(local_path)
    if not path.exists():
        return _fmt_error(
            FileNotFoundError(f"no such file: {path}"),
            context=f"ingest_pdf(local_path={local_path!r})",
        )
    if path.suffix.lower() != ".pdf":
        return _fmt_error(
            ValueError(f"only .pdf is supported, got {path.suffix!r}"),
            context=f"ingest_pdf(local_path={local_path!r})",
        )
    if path.stat().st_size > MAX_INGEST_BYTES:
        return _fmt_error(
            ValueError(
                f"PDF size {path.stat().st_size} exceeds limit {MAX_INGEST_BYTES}"
            ),
            context=f"ingest_pdf(local_path={local_path!r})",
        )

    if chunk_size < 100 or chunk_size > 4000:
        return _fmt_error(
            ValueError(f"chunk_size must be 100..4000, got {chunk_size}"),
            context="ingest_pdf()",
        )
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        return _fmt_error(
            ValueError(
                f"chunk_overlap must be 0..chunk_size-1, got {chunk_overlap}"
            ),
            context="ingest_pdf()",
        )

    def _ingest() -> dict[str, Any]:
        file_hash = _compute_file_hash(path)
        existing_hashes = _load_ingested_hashes(collection)
        if file_hash in existing_hashes:
            return {
                "collection": collection,
                "source": str(path),
                "num_pages": 0,
                "num_chunks_added": 0,
                "total_chunks_in_collection": _collection_count(collection),
                "skipped": True,
                "reason": (
                    f"PDF content hash already ingested "
                    f"(original: {existing_hashes[file_hash]})"
                ),
            }

        pages = _load_pdf_pages(path)
        chunks = _chunk_pages(
            pages,
            source=str(path),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not chunks:
            return {
                "collection": collection,
                "source": str(path),
                "num_pages": len(pages),
                "num_chunks_added": 0,
                "total_chunks_in_collection": _collection_count(collection),
                "warning": "PDF contained no extractable text",
            }
        from langchain_community.vectorstores import FAISS

        texts = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        existing = _load_faiss_store(collection)
        if existing is None:
            embedder = _get_embedder()
            store = FAISS.from_texts(
                texts=texts,
                embedding=embedder,
                metadatas=metadatas,
            )
        else:
            existing.add_texts(texts=texts, metadatas=metadatas)
            store = existing
        _save_faiss_store(collection, store)
        _invalidate_bm25(collection)
        _save_ingested_hash(collection, file_hash, str(path))
        return {
            "collection": collection,
            "source": str(path),
            "num_pages": len(pages),
            "num_chunks_added": len(chunks),
            "total_chunks_in_collection": _collection_count(collection),
        }

    try:
        return await asyncio.to_thread(_ingest)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(
            e,
            context=(
                f"ingest_pdf(local_path={local_path!r}, collection={collection!r})"
            ),
        )


def _collection_count(collection: str) -> int:
    """返回 ``collection`` 的分块数量（尽力而为，出错返回 0）。

    FAISS 中计数 ``index_to_docstore_id`` 的条目而非 ``index.ntotal``：
    两者应相等，但前者与构建 BM25 时使用的数量一致，在未来可能添加的"软删除"下保持两层一致。
    """
    try:
        store = _load_faiss_store(collection)
        if store is None:
            return 0
        return len(store.index_to_docstore_id)
    except Exception:  # noqa: BLE001
        return 0


# ---------------------------------------------------------------------
# 工具 2：带纠正式 RAG 质量信号的混合检索
# ---------------------------------------------------------------------
from research_agent.rag.grader import RetrievalGrader as _RetrievalGrader  # noqa: E402

_GRADER = _RetrievalGrader(
    high_threshold=QUALITY_HIGH_THRESHOLD,
    medium_threshold=QUALITY_MEDIUM_THRESHOLD,
)


def _classify_quality(top_score: float, mean_score: float, unique_sources: int) -> str:
    """委托给 ``rag.grader.RetrievalGrader``。"""
    return _GRADER.grade(top_score, mean_score, unique_sources)


from research_agent.rag.retriever import hybrid_rrf_fuse as _hybrid_fuse  # noqa: E402, F811


async def _maybe_rerank(
    query: str, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """可选地使用本地 cross-encoder 对 ``candidates`` 重排序。

    以下情况返回原列表不变：
      * ``KNOWLEDGE_RERANKER_ENABLED`` 环境变量为假值，
      * 或重排序模型加载失败（如该主机无法 import ``sentence_transformers``），
      * 或底层 ``CrossEncoder.predict`` 调用抛出异常。

    在所有回退路径中，每个候选仍携带 ``rerank_score`` 键（设为 ``None``），
    因此无论重排序是否实际执行，响应形状保持稳定。调用者须容忍``rerank_score is None``。

    此函数有意设计为容错：搜索是 Agent 的主要工具，因可选的重排序器异常而中断搜索是错误的权衡。
    """
    if not candidates:
        return candidates
    if not _reranker_enabled():
        for c in candidates:
            c.setdefault("rerank_score", None)
        return candidates

    global _RERANKER
    try:
        if _RERANKER is None:
            from research_agent.rag.reranker import CrossEncoderReranker

            _RERANKER = CrossEncoderReranker()
            logger.info("Cross-encoder reranker initialised for knowledge_server")
        return await _RERANKER.rerank(query, candidates)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Reranker unavailable ({}); falling back to RRF order", exc
        )
        for c in candidates:
            c.setdefault("rerank_score", None)
        return candidates


@mcp.tool()
async def search(
    query: str,
    collection: str = "default",
    top_k: int = 5,
) -> dict:
    """混合检索（向量 + BM25 + 可选 cross-encoder 重排序）。

    Pipeline::

        FAISS (top_k * 3) ─┐
                           ├─ RRF fuse ─→ cross-encoder rerank ─→ trim
        BM25  (top_k * 3) ─┘    (over-fetch)        (optional)    (top_k)

    重排序步骤使用本地 ``BAAI/bge-reranker-base`` cross-encoder；
    通过 ``KNOWLEDGE_RERANKER_ENABLED`` 环境变量切换。禁用或不可用时，管线 优雅降级为 RRF 顺序 — 响应形状相同，只是每个命中的``rerank_score: null``。

    响应形状为纠正式 RAG Agent 量身设计：携带逐命中分数和顶级``quality`` 标签。预期的 Agent 循环为::

        result = call("knowledge_search", query=Q, collection=C, top_k=5)
        if result["quality"] == "low":
            Q' = rewrite(Q)        # Agent 执行此操作
            result = call("knowledge_search", query=Q', ...)

    Args:
        query: 自由格式自然语言问题。中英文均可 — bge-small-zh 嵌入器支持双语。
        collection: 要搜索的集合。集合不存在或为空时返回空的 ``quality='low'`` 响应（非错误），以便新 Agent 可无需异常处理即可探测集合。
        top_k: 融合后返回的最大命中数。上限为 ``MAX_TOP_K``（20） 以保护 LLM 上下文窗口。

    Returns:
        ``{
            collection, query, top_k_returned,
            quality,                  # "high" / "medium" / "low"
            top_score, mean_score, unique_sources,
            results: [
                {content, source, page,
                 vector_score, bm25_score, rrf_score, rerank_score,
                 vector_rank, bm25_rank}
            ]
        }``

        ``rerank_score`` 在 cross-encoder 执行时为浮点数，重排序禁用/不可用时为 ``None``。
        ``quality`` 标签基于``vector_score`` 计算（其区间针对归一化余弦校准），而非 cross-encoder 的 logits，
         因此纠正式 RAG Agent 循环在重排序器开/关配置间保持稳定信号。

        失败时返回：``{error, context}``。
    """
    try:
        _validate_collection_name(collection)
    except ValueError as e:
        return _fmt_error(e, context=f"search(collection={collection!r})")

    if not query or not query.strip():
        return _fmt_error(
            ValueError("query must be non-empty"),
            context="search()",
        )
    if top_k < 1:
        return _fmt_error(
            ValueError(f"top_k must be >= 1, got {top_k}"),
            context="search()",
        )
    top_k = min(top_k, MAX_TOP_K)

    def _collect_candidates() -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
        """同步阶段：加载 FAISS，运行向量 + BM25，通过 RRF 融合。


        在正常流程中，返回值为 ``(candidates, None)`` — 此时候选列表被过度提取（``top_k * RERANK_OVERFETCH_MULTIPLIER``），以便异步重排序器有空间进行重新排序。
        对于冷启动情况 (collection missing / empty) ，会返回 ``(None, early_response)`` ，以便调用者能够直接结束流程而无需支付重新排序器的启动费用。
        """
        store = _load_faiss_store(collection)
        if store is None or len(store.index_to_docstore_id) == 0:
            return None, {
                "collection": collection,
                "query": query,
                "top_k_returned": 0,
                "quality": "low",
                "top_score": 0.0,
                "mean_score": 0.0,
                "unique_sources": 0,
                "results": [],
                "warning": (
                    f"collection {collection!r} is empty; ingest a PDF first "
                    f"with knowledge_ingest_pdf"
                ),
            }

        # 从每个检索器多取一些候选，给 cross-encoder 留出空间 提升 bi-encoder 排名较低的条目。
        retrieve_k = max(top_k * RERANK_OVERFETCH_MULTIPLIER, 10)
        try:
            raw_vec = store.similarity_search_with_score(query, k=retrieve_k)
        except Exception:  # noqa: BLE001
            raw_vec = []
        # LangChain 的 FAISS 默认返回 (Document, L2 距离)。
        # 因为在``_get_embedder`` 中设置了 ``normalize_embeddings=True``，
        # 这些嵌入(embeddings)是单位向量，因此 L2 平方距离 ``d`` 和余弦相似度 ``s`` 满足``d = 2 - 2s`` ⇒ ``s = 1 - d / 2``。钳位到 [0, 1] 以吸收微小浮点漂移。
        vector_hits: list[tuple[dict[str, Any], float]] = []
        for doc, distance in raw_vec:
            similarity = max(0.0, min(1.0, 1.0 - float(distance) / 2.0))
            vector_hits.append(
                (
                    {"content": doc.page_content, "metadata": doc.metadata or {}},
                    similarity,
                )
            )

        bm25 = _get_bm25(collection)
        bm25_raw = bm25.search(query, top_k=retrieve_k)
        bm25_hits = [(idx, score, bm25.docs[idx]) for idx, score in bm25_raw]

        # 将完整融合列表交给重排序器 — 裁剪到 ``top_k`` 在重排序之后进行，否则会丢失 cross-encoder 本应提升的候选。
        candidates = _hybrid_fuse(vector_hits, bm25_hits)[:retrieve_k]
        return candidates, None

    try:
        candidates, early = await asyncio.to_thread(_collect_candidates)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(
            e,
            context=(
                f"search(query={query!r}, collection={collection!r}, top_k={top_k})"
            ),
        )

    if early is not None:
        return early

    assert candidates is not None  # narrows for type checkers

    # 重排序是尽力而为的：``_maybe_rerank`` 始终返回相同形状的列表（``rerank_score`` 已填充或为 ``None``），
    # 即使环境变量关闭或模型加载失败 — 搜索响应形状保持稳定。
    reranked = await _maybe_rerank(query, candidates)
    final_records = reranked[:top_k]

    results: list[dict[str, Any]] = []
    for rec in final_records:
        meta = rec["metadata"] or {}
        results.append(
            {
                "content": rec["content"],
                "source": meta.get("source", ""),
                "page": meta.get("page"),
                "vector_score": round(rec["vector_score"], 4),
                "bm25_score": round(rec["bm25_score"], 4),
                "rrf_score": round(rec["rrf_score"], 6),
                "rerank_score": rec.get("rerank_score"),
                "vector_rank": rec["vector_rank"],
                "bm25_rank": rec["bm25_rank"],
            }
        )

    # 质量分类基于 ``vector_score`` — 这些阈值是针对 bi-encoder 归一化余弦校准的。cross-encoder logits 处于不同的尺度，会悄悄使 high/medium/low 区间失效。
    scores = [r["vector_score"] for r in results]
    top_score = max(scores) if scores else 0.0
    mean_score = sum(scores) / len(scores) if scores else 0.0
    unique_sources = len({r["source"] for r in results if r["source"]})
    quality = _classify_quality(top_score, mean_score, unique_sources)

    return {
        "collection": collection,
        "query": query,
        "top_k_returned": len(results),
        "quality": quality,
        "top_score": round(top_score, 4),
        "mean_score": round(mean_score, 4),
        "unique_sources": unique_sources,
        "results": results,
    }


# ---------------------------------------------------------------------
# 工具 3：列出集合（含分块数量）
# ---------------------------------------------------------------------
@mcp.tool()
async def list_collections() -> dict:
    """列出持久化知识库中当前所有集合。

    当用户查询暗示某个库但未指定集合名（如"我的 ESG 库里有什么？"）时，可作为 Agent 的首个调用。Agent 随后可将后续 ``search`` 调用路由到正确的集合。

    此处"集合"指 ``DEFAULT_DB_DIR`` 下同时包含 ``index.faiss`` 和``index.pkl`` 的子目录。无关目录（如半删除残留）被静默跳过，从不向 LLM 暴露损坏的集合。

    Returns:
        ``{db_dir, collections: [{name, chunk_count}]}``。每个条目含
        ``name``（str）和 ``chunk_count``（int，尽力而为 —— FAISS 文件对不可读时为 ``-1``）。
    """

    def _list() -> dict[str, Any]:
        DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
        out: list[dict[str, Any]] = []
        for child in sorted(DEFAULT_DB_DIR.iterdir()):
            if not child.is_dir():
                continue
            if not _faiss_index_exists(child.name):
                continue
            try:
                chunk_count = _collection_count(child.name)
            except Exception:  # noqa: BLE001
                chunk_count = -1
            out.append({"name": child.name, "chunk_count": chunk_count})
        return {"db_dir": str(DEFAULT_DB_DIR), "collections": out}

    try:
        return await asyncio.to_thread(_list)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context="list_collections()")


# ---------------------------------------------------------------------
# 工具 4：删除集合
# ---------------------------------------------------------------------
@mcp.tool()
async def delete_collection(collection: str) -> dict:
    """删除集合及其内存缓存。幂等操作。

    用于用户想要从头重新导入语料库时的清理（如更改 chunk_size 后）。缺失的集合不会抛异常 — 响应仅报告 ``existed=False``。

    Returns:
        ``{collection, existed, deleted}``。
    """
    try:
        _validate_collection_name(collection)
    except ValueError as e:
        return _fmt_error(e, context=f"delete_collection({collection!r})")

    def _delete() -> dict[str, Any]:
        import shutil

        cdir = _collection_dir(collection)
        existed = _faiss_index_exists(collection)
        if not existed:
            # 磁盘上无数据；仍需清除过期的内存缓存，以便下次以此名称导入时真正从零开始。
            _FAISS_STORES.pop(collection, None)
            _BM25_CACHE.pop(collection, None)
            return {"collection": collection, "existed": False, "deleted": False}
        if cdir.exists():
            shutil.rmtree(cdir)
        _FAISS_STORES.pop(collection, None)
        _BM25_CACHE.pop(collection, None)
        return {"collection": collection, "existed": True, "deleted": True}

    try:
        return await asyncio.to_thread(_delete)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context=f"delete_collection({collection!r})")


# ---------------------------------------------------------------------
# 模块导出 — math 导入仅为避免 ruff 在未来迭代使用 log/exp 归一化时，报未使用导入警告。零开销。
# ---------------------------------------------------------------------
_ = math


if __name__ == "__main__":
    mcp.run(transport="stdio")
