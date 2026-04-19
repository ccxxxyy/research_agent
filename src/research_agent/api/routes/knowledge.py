"""Knowledge base management endpoints — upload, list, search, delete."""

from __future__ import annotations

from fastapi import APIRouter, UploadFile
from loguru import logger

from research_agent.api.dependencies import SettingsDep
from research_agent.api.schemas import KnowledgeListResponse
from research_agent.rag.chunker import ChunkStrategy, chunk_documents
from research_agent.rag.loader import load_file

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

# Temporary in-memory store; will be replaced by ChromaDB integration
_temp_docs: list[dict] = []


@router.post("/upload")
async def upload_document(
    file: UploadFile,
    collection: str = "default",
    settings: SettingsDep = None,
) -> dict:
    """Upload a document to the knowledge base.

    Supported formats: PDF, Markdown, TXT.
    The document is chunked, embedded, and stored in the vector database.
    """
    import tempfile
    from pathlib import Path

    suffix = Path(file.filename or "doc.txt").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    logger.info("Uploading document: {} ({} bytes)", file.filename, len(content))

    # Load → Chunk → (Embed + Store handled downstream)
    docs = await load_file(tmp_path)
    chunks = chunk_documents(docs, strategy=ChunkStrategy.RECURSIVE)

    # TODO: embed and store in ChromaDB
    _temp_docs.extend(
        {"content": c.page_content[:200], "metadata": c.metadata} for c in chunks
    )

    return {
        "status": "uploaded",
        "file_name": file.filename,
        "chunks": len(chunks),
        "collection": collection,
    }


@router.get("", response_model=KnowledgeListResponse)
async def list_documents(collection: str = "default") -> KnowledgeListResponse:
    """List all documents in a knowledge base collection."""
    # TODO: query ChromaDB for collection contents
    return KnowledgeListResponse(
        collection=collection,
        total=len(_temp_docs),
        documents=[],
    )


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, collection: str = "default") -> dict:
    """Remove a document from the knowledge base."""
    # TODO: delete from ChromaDB
    return {"status": "deleted", "doc_id": doc_id, "collection": collection}
