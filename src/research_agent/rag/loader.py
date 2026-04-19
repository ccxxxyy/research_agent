"""Document loaders for various file formats."""

from __future__ import annotations

import asyncio
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from loguru import logger

LOADER_MAP: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
}


async def load_file(path: str | Path) -> list[Document]:
    """Load a single file into LangChain Document objects."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    ext = file_path.suffix.lower()
    loader_cls = LOADER_MAP.get(ext)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {ext}")

    logger.info("Loading document: {}", file_path.name)
    loader = loader_cls(str(file_path))
    docs = await loader.aload()

    for doc in docs:
        doc.metadata.setdefault("source", str(file_path))
        doc.metadata.setdefault("file_name", file_path.name)

    return docs


async def load_url(url: str) -> list[Document]:
    """Load a web page into Document objects."""
    logger.info("Loading URL: {}", url)
    loader = WebBaseLoader(url)
    docs: list[Document] = await asyncio.to_thread(loader.load)
    for doc in docs:
        doc.metadata.setdefault("source", url)
    return docs


async def load_documents(sources: list[str]) -> list[Document]:
    """Load multiple files/URLs into Document objects."""
    all_docs: list[Document] = []
    for source in sources:
        if source.startswith(("http://", "https://")):
            docs = await load_url(source)
        else:
            docs = await load_file(source)
        all_docs.extend(docs)
    logger.info("Loaded {} documents from {} sources", len(all_docs), len(sources))
    return all_docs
