"""Tests for document chunking strategies."""

from langchain_core.documents import Document

from research_agent.rag.chunker import ChunkStrategy, chunk_documents


class TestChunker:
    def test_recursive_chunking(self):
        docs = [Document(page_content="Hello world. " * 200, metadata={"source": "test"})]
        chunks = chunk_documents(
            docs, strategy=ChunkStrategy.RECURSIVE, chunk_size=500, chunk_overlap=50
        )
        assert len(chunks) > 1
        assert all(len(c.page_content) <= 600 for c in chunks)

    def test_preserves_metadata(self):
        docs = [Document(page_content="x " * 500, metadata={"source": "a.pdf", "page": 1})]
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert chunk.metadata["source"] == "a.pdf"

    def test_empty_input(self):
        chunks = chunk_documents([])
        assert chunks == []
