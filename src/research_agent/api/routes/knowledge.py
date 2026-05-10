"""Knowledge base management endpoints — ingest, search, list, delete.

These endpoints expose the **same** FAISS + BM25 + cross-encoder
pipeline the ``knowledge_expert`` agent uses, but as a direct REST
surface. This lets a frontend operate the knowledge base without
routing every action through the supervisor:

- ``POST /api/knowledge/ingest`` — upload a PDF → chunk → embed → FAISS.
- ``POST /api/knowledge/search`` — hybrid search over a collection.
- ``GET  /api/knowledge/collections`` — enumerate collections + stats.
- ``DELETE /api/knowledge/collections/{name}`` — drop a collection.

All four endpoints delegate to the coroutines defined in
``research_agent.mcp_servers.knowledge_server`` (the canonical
implementation). The FastAPI layer handles HTTP concerns (file
upload, JSON serialisation, status codes) and nothing else — no
business logic is duplicated.

User isolation
--------------
Every endpoint requires a ``X-User-ID`` header. Collections are
namespaced per user on disk via a ``{user_id}__{collection}`` naming
convention. This ensures users cannot see or modify each other's
knowledge bases through the REST surface.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, HTTPException, UploadFile, status
from loguru import logger
from pydantic import BaseModel, Field

from research_agent.mcp_servers.knowledge_server import (
    delete_collection as _delete_collection,
    ingest_pdf as _ingest_pdf,
    list_collections as _list_collections,
    search as _search,
)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

_USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,61}[a-zA-Z0-9]$")


def _scoped_collection(user_id: str, collection: str) -> str:
    """Prefix collection name with user_id for on-disk isolation."""
    return f"{user_id}__{collection}"


def _validate_user_id(user_id: str) -> str:
    """Validate and return a cleaned user_id."""
    uid = user_id.strip()
    if not uid or not _USER_ID_PATTERN.match(uid):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="X-User-ID must be 2-63 chars matching [a-zA-Z0-9._-].",
        )
    return uid


# =====================================================================
# Request / Response models (co-located: only this route uses them)
# =====================================================================


class SearchRequest(BaseModel):
    """Body for ``POST /api/knowledge/search``."""

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


class ListCollectionsResponse(BaseModel):
    db_dir: str
    collections: list[CollectionInfo]


class DeleteCollectionResponse(BaseModel):
    collection: str
    existed: bool
    deleted: bool


# =====================================================================
# Endpoints
# =====================================================================


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_pdf(
    file: UploadFile,
    collection: str = "default",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    x_user_id: str = Header(..., alias="X-User-ID"),
) -> IngestResponse:
    """Upload a PDF and ingest it into a FAISS knowledge-base collection.

    The file is written to a temporary path, then passed to the
    ``knowledge_server.ingest_pdf`` pipeline (load → chunk → embed →
    write). On success the temp file is cleaned up; on failure it is
    left for debugging and the error is surfaced as a 422.
    """
    user_id = _validate_user_id(x_user_id)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only .pdf files are supported for ingestion.",
        )

    scoped = _scoped_collection(user_id, collection)
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
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
    )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result["error"],
        )

    # Clean up temp file on success
    try:
        Path(tmp_path).unlink(missing_ok=True)
    except OSError:
        pass

    # Return the user-facing collection name (without internal prefix)
    result["collection"] = collection
    return IngestResponse(**result)


@router.post("/search", response_model=SearchResponse)
async def search_knowledge(
    body: SearchRequest,
    x_user_id: str = Header(..., alias="X-User-ID"),
) -> SearchResponse:
    """Hybrid (vector + BM25 + rerank) search over a knowledge-base collection.

    Returns up to ``top_k`` hits with per-hit scores and a top-level
    ``quality`` label (high / medium / low) that a frontend can use
    to display confidence indicators.
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
    """List FAISS collections belonging to the authenticated user."""
    user_id = _validate_user_id(x_user_id)
    prefix = f"{user_id}__"

    result = await _list_collections()

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"],
        )

    # Filter to only this user's collections and strip the prefix
    user_collections = [
        {"name": c["name"][len(prefix):], "chunk_count": c["chunk_count"]}
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
    """Delete a knowledge-base collection. Idempotent — missing collections
    return ``existed=False`` with a 200.
    """
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
