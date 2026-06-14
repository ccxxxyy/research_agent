"""知识库管理端点 —— 导入、搜索、列举、删除。

这些端点暴露了与 ``knowledge_expert`` 智能体使用的相同 FAISS + BM25+ 交叉编码器流水线，但以直接 REST 接口的形式提供。这使前端可以直接操作知识库，而无需将每个操作路由到主管：

- ``POST /api/knowledge/ingest`` —— 上传 PDF → 分块 → 向量化 → FAISS。
- ``POST /api/knowledge/search`` —— 在集合上执行混合检索。
- ``GET  /api/knowledge/collections`` —— 枚举集合及其统计信息。
- ``DELETE /api/knowledge/collections/{name}`` —— 删除集合。

全部四个端点均委托给 ``research_agent.mcp_servers.knowledge_server`` 中定义的协程（规范实现）。
FastAPI 层仅处理 HTTP 关注点（文件上传、JSON序列化、状态码），不复制任何业务逻辑。

用户隔离
--------
每个端点均要求 ``X-User-ID`` 头。集合在磁盘上通过``{user_id}__{collection}`` 命名约定按用户隔离，确保用户无法通过REST 接口查看或修改他人的知识库。
"""

from __future__ import annotations

import contextlib
import re
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel, Field

from research_agent.mcp_servers.knowledge_server import (
    DEFAULT_DB_DIR as _DEFAULT_DB_DIR,
)
from research_agent.mcp_servers.knowledge_server import (
    delete_collection as _delete_collection,
)
from research_agent.mcp_servers.knowledge_server import (
    delete_document as _delete_document,
)
from research_agent.mcp_servers.knowledge_server import (
    ingest_pdf as _ingest_pdf,
)
from research_agent.mcp_servers.knowledge_server import (
    list_collections as _list_collections,
)
from research_agent.mcp_servers.knowledge_server import (
    search as _search,
)

_PDF_SUBDIR = "pdfs"


def _pdf_storage_dir(scoped_collection: str) -> Path:
    """持久化 PDF 存储目录：``data/knowledge_db/{collection}/pdfs/``。"""
    return _DEFAULT_DB_DIR / scoped_collection / _PDF_SUBDIR


def _safe_filename(name: str) -> str:
    """将文件名中不安全的字符替换为下划线。"""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)


router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

_USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,61}[a-zA-Z0-9]$")


def _scoped_collection(user_id: str, collection: str) -> str:
    """为集合名添加 user_id 前缀以实现磁盘级隔离。"""
    return f"{user_id}__{collection}"


def _validate_user_id(user_id: str) -> str:
    """校验并返回清理后的 user_id。"""
    uid = user_id.strip()
    if not uid or not _USER_ID_PATTERN.match(uid):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="X-User-ID must be 2-63 chars matching [a-zA-Z0-9._-].",
        )
    return uid


# =====================================================================
# 请求 / 响应模型（就近定义：仅本路由使用）
# =====================================================================


class SearchRequest(BaseModel):
    """``POST /api/knowledge/search`` 的请求体。"""

    query: str = Field(..., min_length=1, max_length=4000)
    collection: str = "default"
    top_k: int = Field(default=5, ge=1, le=20)


class IngestResponse(BaseModel):
    collection: str
    source: str
    num_pages: int
    num_chunks_added: int
    total_chunks_in_collection: int


class SearchResponse(BaseModel):
    collection: str
    query: str
    top_k_returned: int
    quality: str
    top_score: float
    mean_score: float
    unique_sources: int
    results: list[dict[str, Any]]


class CollectionInfo(BaseModel):
    name: str
    chunk_count: int
    sources: list[str] = []


class ListCollectionsResponse(BaseModel):
    db_dir: str
    collections: list[CollectionInfo]


class DeleteCollectionResponse(BaseModel):
    collection: str
    existed: bool
    deleted: bool


class DeleteDocumentResponse(BaseModel):
    collection: str
    source: str
    removed_chunks: int
    deleted: bool
    existed: bool = True


# =====================================================================
# 端点
# =====================================================================


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_pdf(
    file: UploadFile,
    collection: str = "default",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    x_user_id: str = Header(..., alias="X-User-ID"),
) -> IngestResponse:
    """上传 PDF 并导入到 FAISS 知识库集合。

    文件先写入临时路径，再传给 ``knowledge_server.ingest_pdf`` 流水线
    （加载 → 分块 → 向量化 → 写入）。
    成功后清理临时文件；失败时保留以便调试，并将错误以 422 返回。
    """
    user_id = _validate_user_id(x_user_id)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only .pdf files are supported for ingestion.",
        )

    scoped = _scoped_collection(user_id, collection)
    suffix = Path(file.filename).suffix
    # 保留原始文件名，用前缀避免冲突
    safe_name = re.sub(r'[<>:"/\\|?*]', "_", Path(file.filename).stem)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=f"{safe_name}_") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    logger.info(
        "Knowledge ingest: user={}, file={}, size={}, collection={}",
        user_id,
        file.filename,
        len(content),
        scoped,
    )

    result = await _ingest_pdf(
        local_path=tmp_path,
        collection=scoped,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source_name=file.filename,
    )

    if "error" in result:
        with contextlib.suppress(OSError):
            Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result["error"],
        )

    # 将 PDF 持久化到集合的 pdfs/ 子目录供后续查看
    pdf_dir = _pdf_storage_dir(scoped)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    dest = pdf_dir / _safe_filename(file.filename)
    try:
        import shutil

        shutil.move(str(tmp_path), str(dest))
    except OSError:
        with contextlib.suppress(OSError):
            Path(tmp_path).unlink(missing_ok=True)

    # 返回面向用户的集合名（不含内部前缀）
    result["collection"] = collection
    return IngestResponse(**result)


@router.post("/search", response_model=SearchResponse)
async def search_knowledge(
    body: SearchRequest,
    x_user_id: str = Header(..., alias="X-User-ID"),
) -> SearchResponse:
    """在知识库集合上执行混合检索（向量 + BM25 + 重排序）。

    返回最多 ``top_k`` 条命中结果及每条的分数，以及顶层 ``quality``标签（high / medium / low），前端可据此展示置信度指示器。
    """
    user_id = _validate_user_id(x_user_id)
    scoped = _scoped_collection(user_id, body.collection)

    result = await _search(
        query=body.query,
        collection=scoped,
        top_k=body.top_k,
    )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result["error"],
        )

    result["collection"] = body.collection
    return SearchResponse(**result)


@router.get("/collections", response_model=ListCollectionsResponse)
async def list_collections(
    x_user_id: str = Header(..., alias="X-User-ID"),
) -> ListCollectionsResponse:
    """列出当前认证用户拥有的 FAISS 集合。"""
    user_id = _validate_user_id(x_user_id)
    prefix = f"{user_id}__"

    result = await _list_collections()

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"],
        )

    # 仅保留当前用户的集合并去除前缀
    user_collections = [
        {
            "name": c["name"][len(prefix) :],
            "chunk_count": c["chunk_count"],
            "sources": c.get("sources", []),
        }
        for c in result.get("collections", [])
        if c["name"].startswith(prefix)
    ]
    result["collections"] = user_collections
    return ListCollectionsResponse(**result)


@router.delete(
    "/collections/{collection_name}",
    response_model=DeleteCollectionResponse,
)
async def delete_collection(
    collection_name: str,
    x_user_id: str = Header(..., alias="X-User-ID"),
) -> DeleteCollectionResponse:
    """删除知识库集合。幂等操作 —— 不存在的集合返回 ``existed=False``及 200 状态码。"""
    user_id = _validate_user_id(x_user_id)
    scoped = _scoped_collection(user_id, collection_name)

    result = await _delete_collection(collection=scoped)

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result["error"],
        )

    result["collection"] = collection_name
    return DeleteCollectionResponse(**result)


@router.delete("/documents", response_model=DeleteDocumentResponse)
async def delete_document(
    collection: str = "default",
    source: str = "",
    x_user_id: str = Header(..., alias="X-User-ID"),
) -> DeleteDocumentResponse:
    """从集合中删除单个 PDF 文档（按显示文件名匹配）。"""
    user_id = _validate_user_id(x_user_id)
    if not source.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="source query parameter is required",
        )
    scoped = _scoped_collection(user_id, collection)
    result = await _delete_document(collection=scoped, source=source)
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result["error"],
        )
    # 同时删除持久化的 PDF 文件
    if result.get("deleted"):
        pdf_path = _pdf_storage_dir(scoped) / _safe_filename(source)
        with contextlib.suppress(OSError):
            pdf_path.unlink(missing_ok=True)
    result["collection"] = collection
    return DeleteDocumentResponse(**result)


@router.get("/documents/view")
async def view_document(
    collection: str = "default",
    source: str = "",
    user_id: str = "",
) -> FileResponse:
    """以浏览器可渲染方式返回已上传的 PDF 原始文件。

    此端点通过 query param ``user_id`` 鉴权（而非 header），
    因为浏览器直接在新窗口打开链接时无法附加自定义请求头。
    """
    uid = user_id.strip()
    if not uid or not _USER_ID_PATTERN.match(uid):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="user_id query parameter is required",
        )
    if not source.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="source query parameter is required",
        )
    scoped = _scoped_collection(uid, collection)
    pdf_dir = _pdf_storage_dir(scoped)
    target = pdf_dir / _safe_filename(source)
    if not target.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF file not found: {source}",
        )
    return FileResponse(
        path=str(target),
        media_type="application/pdf",
        filename=source,
    )
