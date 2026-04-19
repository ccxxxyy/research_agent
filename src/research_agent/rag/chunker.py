"""Document chunking strategies for optimal retrieval."""

from __future__ import annotations

from enum import Enum

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from loguru import logger


class ChunkStrategy(str, Enum):
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"


def chunk_documents(
    documents: list[Document],
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into chunks using the specified strategy.

    Default uses recursive character splitting with sensible defaults
    for Chinese + English mixed content.
    """
    if strategy == ChunkStrategy.MARKDOWN:
        return _chunk_markdown(documents, chunk_size, chunk_overlap)
    return _chunk_recursive(documents, chunk_size, chunk_overlap)


def _chunk_recursive(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.info(
        "Chunked {} documents → {} chunks (size={}, overlap={})",
        len(documents), len(chunks), chunk_size, chunk_overlap,
    )
    return chunks


def _chunk_markdown(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Two-stage splitting: first by markdown headers, then by size."""
    headers_to_split = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split)

    header_chunks: list[Document] = []
    for doc in documents:
        splits = md_splitter.split_text(doc.page_content)
        for split in splits:
            split.metadata.update(doc.metadata)
            header_chunks.append(split)

    # Second pass: split oversized header chunks by character count
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    final_chunks = size_splitter.split_documents(header_chunks)

    logger.info(
        "Markdown chunked {} docs → {} header sections → {} final chunks",
        len(documents), len(header_chunks), len(final_chunks),
    )
    return final_chunks
