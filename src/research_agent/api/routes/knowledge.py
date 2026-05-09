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
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, status
from loguru import logger
from pydantic import BaseModel, Field

from research_agent.mcp_servers.knowledge_server import (
    delete_collection as _delete_collection,
    ingest_pdf as _ingest_pdf,
    list_collections as _list_collections,
    search as _search,
)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


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
) -> IngestResponse:
    """Upload a PDF and ingest it into a FAISS knowledge-base collection.

    The file is written to a temporary path, then passed to the
    ``knowledge_server.ingest_pdf`` pipeline (load → chunk → embed →
    write). On success the temp file is cleaned up; on failure it is
    left for debugging and the error is surfaced as a 422.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only .pdf files are supported for ingestion.",
        )

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    logger.info(
        "Knowledge ingest: file={}, size={}, collection={}",
        file.filename,
        len(content),
        collection,
    )

    result = await _ingest_pdf(
        local_path=tmp_path,
        collection=collection,
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

    return IngestResponse(**result)


@router.post("/search", response_model=SearchResponse)
async def search_knowledge(body: SearchRequest) -> SearchResponse:
    """Hybrid (vector + BM25 + rerank) search over a knowledge-base collection.

    Returns up to ``top_k`` hits with per-hit scores and a top-level
    ``quality`` label (high / medium / low) that a frontend can use
    to display confidence indicators.
    """
    result = await _search(
        query=body.query,
        collection=body.collection,
        top_k=body.top_k,
    )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result["error"],
        )

    return SearchResponse(**result)


@router.get("/collections", response_model=ListCollectionsResponse)
async def list_collections() -> ListCollectionsResponse:
    """List all FAISS collections persisted on disk with their chunk counts."""
    result = await _list_collections()

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"],
        )

    return ListCollectionsResponse(**result)


@router.delete(
    "/collections/{collection_name}",
    response_model=DeleteCollectionResponse,
)
async def delete_collection(collection_name: str) -> DeleteCollectionResponse:
    """Delete a knowledge-base collection. Idempotent — missing collections
    return ``existed=False`` with a 200.
    """
    result = await _delete_collection(collection=collection_name)

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result["error"],
        )

    return DeleteCollectionResponse(**result)
