"""向量存储抽象层 — 支持 FAISS 和 pgvector 双后端。

通过统一接口隔离底层向量数据库实现，使知识库可在运行时通过配置切换存储引擎。

支持两种后端：
1. FAISS（默认）—— 文件持久化，零外部依赖，适合单机/开发环境。
2. pgvector —— 利用已有 Postgres 容器的 ``vector`` 扩展，支持并发读写和跨实例共享。

选择逻辑由 ``KNOWLEDGE_VECTOR_BACKEND`` 环境变量控制：
- ``"faiss"``（默认）：使用 ``FaissVectorBackend``。
- ``"pgvector"``：使用 ``PgvectorBackend``，要求 Postgres 可达且已安装 pgvector 扩展。
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path  # noqa: TC003
from typing import Any

from loguru import logger


class VectorBackend(ABC):
    """向量存储操作的统一接口。"""

    @abstractmethod
    async def add_texts(
        self,
        collection: str,
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int:
        """向集合添加文本并返回新增的向量数量。"""

    @abstractmethod
    async def similarity_search(
        self,
        collection: str,
        query: str,
        k: int = 5,
    ) -> list[tuple[dict[str, Any], float]]:
        """返回 [(doc_dict, similarity_score), ...] 按相似度降序排列。"""

    @abstractmethod
    async def collection_exists(self, collection: str) -> bool:
        """检查集合是否存在且非空。"""

    @abstractmethod
    async def collection_count(self, collection: str) -> int:
        """返回集合中的向量数量。"""

    @abstractmethod
    async def delete_collection(self, collection: str) -> bool:
        """删除集合，返回是否存在过。"""

    @abstractmethod
    async def list_collections(self) -> list[dict[str, Any]]:
        """返回 [{name, chunk_count}, ...]。"""


class FaissVectorBackend(VectorBackend):
    """基于 LangChain FAISS 的文件持久化后端（当前默认）。

    委托给 ``rag.faiss_store`` 共享模块。
    """

    def __init__(self, db_dir: Path | None = None) -> None:
        from research_agent.rag.faiss_store import DEFAULT_DB_DIR

        self._db_dir = db_dir or DEFAULT_DB_DIR

    async def add_texts(
        self,
        collection: str,
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int:
        import asyncio

        from research_agent.rag import faiss_store

        def _add() -> int:
            existing = faiss_store.load_store(collection, db_dir=self._db_dir)
            if existing is None:
                faiss_store.create_from_texts(collection, texts, metadatas, db_dir=self._db_dir)
            else:
                existing.add_texts(texts=texts, metadatas=metadatas)
                faiss_store.save_store(collection, existing, db_dir=self._db_dir)
            return len(texts)

        return await asyncio.to_thread(_add)

    async def similarity_search(
        self,
        collection: str,
        query: str,
        k: int = 5,
    ) -> list[tuple[dict[str, Any], float]]:
        import asyncio

        from research_agent.rag import faiss_store

        def _search() -> list[tuple[dict[str, Any], float]]:
            store = faiss_store.load_store(collection, db_dir=self._db_dir)
            if store is None:
                return []
            raw = store.similarity_search_with_score(query, k=k)
            results = []
            for doc, distance in raw:
                similarity = max(0.0, min(1.0, 1.0 - float(distance) / 2.0))
                results.append(
                    ({"content": doc.page_content, "metadata": doc.metadata or {}}, similarity)
                )
            return results

        return await asyncio.to_thread(_search)

    async def collection_exists(self, collection: str) -> bool:
        from research_agent.rag import faiss_store

        return faiss_store.index_exists(collection, db_dir=self._db_dir)

    async def collection_count(self, collection: str) -> int:
        from research_agent.rag import faiss_store

        return faiss_store.chunk_count(collection, db_dir=self._db_dir)

    async def delete_collection(self, collection: str) -> bool:
        import shutil

        from research_agent.rag import faiss_store

        existed = faiss_store.index_exists(collection, db_dir=self._db_dir)
        if existed:
            cdir = self._db_dir / collection
            shutil.rmtree(cdir, ignore_errors=True)
            faiss_store.invalidate_cache(collection)
        return existed

    async def list_collections(self) -> list[dict[str, Any]]:
        from research_agent.rag import faiss_store

        result = []
        if not self._db_dir.is_dir():
            return result
        for child in sorted(self._db_dir.iterdir()):
            if not child.is_dir():
                continue
            if not faiss_store.index_exists(child.name, db_dir=self._db_dir):
                continue
            count = faiss_store.chunk_count(child.name, db_dir=self._db_dir)
            result.append({"name": child.name, "chunk_count": count})
        return result


class PgvectorBackend(VectorBackend):
    """基于 pgvector 的 Postgres 后端。

    利用 docker-compose 中已有的 ``pgvector/pgvector:pg16`` 容器。
    底层委托 ``rag/pgvector_store.py``（psycopg + pgvector，项目已有依赖）。
    """

    def __init__(self, connection_string: str, embedding_model: str | None = None) -> None:
        # connection_string 保留以兼容工厂签名；pgvector_store 从 Settings 读取 URI。
        self._conn_str = connection_string
        self._embedding_model = embedding_model

    async def add_texts(
        self,
        collection: str,
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int:
        import asyncio

        from research_agent.rag import pgvector_store as pvs

        def _add() -> int:
            existing = pvs.load_store(collection)
            if existing is None:
                pvs.create_from_texts(collection, texts, metadatas)
            else:
                existing.add_texts(texts, metadatas)
            return len(texts)

        return await asyncio.to_thread(_add)

    async def similarity_search(
        self,
        collection: str,
        query: str,
        k: int = 5,
    ) -> list[tuple[dict[str, Any], float]]:
        import asyncio

        from research_agent.rag import pgvector_store as pvs

        def _search() -> list[tuple[dict[str, Any], float]]:
            store = pvs.load_store(collection)
            if store is None:
                return []
            raw = store.similarity_search_with_score(query, k=k)
            results = []
            for doc, distance in raw:
                similarity = max(0.0, min(1.0, 1.0 - float(distance) / 2.0))
                results.append(
                    ({"content": doc.page_content, "metadata": doc.metadata or {}}, similarity)
                )
            return results

        return await asyncio.to_thread(_search)

    async def collection_exists(self, collection: str) -> bool:
        import asyncio

        from research_agent.rag import pgvector_store as pvs

        return await asyncio.to_thread(pvs.collection_exists, collection)

    async def collection_count(self, collection: str) -> int:
        import asyncio

        from research_agent.rag import pgvector_store as pvs

        return await asyncio.to_thread(pvs.chunk_count, collection)

    async def delete_collection(self, collection: str) -> bool:
        import asyncio

        from research_agent.rag import pgvector_store as pvs

        return await asyncio.to_thread(pvs.delete_collection, collection)

    async def list_collections(self) -> list[dict[str, Any]]:
        import asyncio

        from research_agent.rag import pgvector_store as pvs

        names = await asyncio.to_thread(pvs.list_collections)
        result = []
        for name in names:
            count = await self.collection_count(name)
            result.append({"name": name, "chunk_count": count})
        return result


def get_vector_backend() -> VectorBackend:
    """根据环境变量返回对应的向量存储后端实例。"""
    backend_type = os.environ.get("KNOWLEDGE_VECTOR_BACKEND", "faiss").lower().strip()

    if backend_type == "pgvector":
        from research_agent.config import get_settings

        settings = get_settings()
        conn_str = settings.database.postgres_sync_uri
        if not conn_str:
            logger.warning(
                "KNOWLEDGE_VECTOR_BACKEND=pgvector but no POSTGRES_SYNC_URI; falling back to FAISS"
            )
            return FaissVectorBackend()
        logger.info("Vector backend: pgvector ({})", conn_str[:30] + "...")
        return PgvectorBackend(connection_string=conn_str)

    logger.info("Vector backend: FAISS (file-based)")
    return FaissVectorBackend()


__all__ = [
    "FaissVectorBackend",
    "PgvectorBackend",
    "VectorBackend",
    "get_vector_backend",
]
