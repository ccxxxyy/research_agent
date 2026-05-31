"""Tests for rag/vector_backend.py — FaissVectorBackend only (no PG required)."""

from __future__ import annotations

import pytest

from research_agent.rag.vector_backend import FaissVectorBackend, get_vector_backend


@pytest.fixture
def faiss_backend(tmp_path):
    """创建一个指向临时目录的 FAISS 后端。"""
    return FaissVectorBackend(db_dir=tmp_path)


class TestFaissVectorBackend:
    @pytest.mark.asyncio
    async def test_initial_state_empty(self, faiss_backend: FaissVectorBackend) -> None:
        assert await faiss_backend.collection_exists("test_coll") is False
        assert await faiss_backend.collection_count("test_coll") == 0

    @pytest.mark.asyncio
    async def test_add_and_search(self, faiss_backend: FaissVectorBackend) -> None:
        texts = ["宁德时代 2024 营收超过 4000 亿元", "比亚迪新能源汽车销量创新高"]
        metadatas = [{"source": "report_a"}, {"source": "report_b"}]

        added = await faiss_backend.add_texts("demo", texts, metadatas)
        assert added == 2
        assert await faiss_backend.collection_exists("demo") is True
        assert await faiss_backend.collection_count("demo") == 2

        results = await faiss_backend.similarity_search("demo", "宁德时代营收", k=2)
        assert len(results) > 0
        first_doc, score = results[0]
        assert "content" in first_doc
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_search_nonexistent_collection(self, faiss_backend: FaissVectorBackend) -> None:
        results = await faiss_backend.similarity_search("no_such", "query", k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_delete_collection(self, faiss_backend: FaissVectorBackend) -> None:
        await faiss_backend.add_texts("to_delete", ["foo"], [{"x": 1}])
        assert await faiss_backend.collection_exists("to_delete") is True

        deleted = await faiss_backend.delete_collection("to_delete")
        assert deleted is True
        assert await faiss_backend.collection_exists("to_delete") is False

        not_deleted = await faiss_backend.delete_collection("never_existed")
        assert not_deleted is False

    @pytest.mark.asyncio
    async def test_list_collections(self, faiss_backend: FaissVectorBackend) -> None:
        await faiss_backend.add_texts("alpha", ["x"], [{}])
        await faiss_backend.add_texts("beta", ["y", "z"], [{}, {}])

        collections = await faiss_backend.list_collections()
        names = [c["name"] for c in collections]
        assert "alpha" in names
        assert "beta" in names
        beta = next(c for c in collections if c["name"] == "beta")
        assert beta["chunk_count"] == 2


class TestGetVectorBackend:
    def test_default_is_faiss(self, monkeypatch) -> None:
        monkeypatch.delenv("KNOWLEDGE_VECTOR_BACKEND", raising=False)
        backend = get_vector_backend()
        assert isinstance(backend, FaissVectorBackend)

    def test_explicit_faiss(self, monkeypatch) -> None:
        monkeypatch.setenv("KNOWLEDGE_VECTOR_BACKEND", "faiss")
        backend = get_vector_backend()
        assert isinstance(backend, FaissVectorBackend)

    def test_pgvector_fallback_without_conn(self, monkeypatch) -> None:
        """当 pgvector 被请求但没有配置 postgres URI 时，回退到 FAISS。"""
        monkeypatch.setenv("KNOWLEDGE_VECTOR_BACKEND", "pgvector")

        from unittest.mock import MagicMock, patch

        fake_settings = MagicMock()
        fake_settings.database.postgres_sync_uri = ""
        with patch("research_agent.config.get_settings", return_value=fake_settings):
            backend = get_vector_backend()
        assert isinstance(backend, FaissVectorBackend)
