"""Postgres pgvector backend for the knowledge base."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from loguru import logger

from research_agent.rag.embeddings import EMBEDDING_DIMENSION, get_embedder

_PG_STORES: dict[str, PgVectorCollection] = {}


class PgVectorCollection:
    """LangChain-FAISS-compatible wrapper around a pgvector table."""

    def __init__(self, collection: str, conninfo: str) -> None:
        self.collection = collection
        self._conninfo = conninfo
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        with psycopg.connect(self._conninfo) as conn:
            register_vector(conn)
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    collection TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({EMBEDDING_DIMENSION}) NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_collection
                ON knowledge_chunks (collection)
                """
            )
            conn.commit()

    @property
    def index_to_docstore_id(self) -> dict[str, str]:
        import psycopg

        with psycopg.connect(self._conninfo) as conn:
            rows = conn.execute(
                "SELECT id::text FROM knowledge_chunks WHERE collection = %s ORDER BY id",
                (self.collection,),
            ).fetchall()
        return {row[0]: row[0] for row in rows}

    def add_texts(self, texts: list[str], metadatas: list[dict[str, Any]]) -> None:
        import json

        import psycopg
        from pgvector.psycopg import register_vector

        embedder = get_embedder()
        vectors = embedder.embed_documents(texts)

        with psycopg.connect(self._conninfo) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                for text, meta, vector in zip(texts, metadatas, vectors, strict=True):
                    cur.execute(
                        """
                        INSERT INTO knowledge_chunks (collection, content, metadata, embedding)
                        VALUES (%s, %s, %s::jsonb, %s)
                        """,
                        (self.collection, text, json.dumps(meta), vector),
                    )
            conn.commit()

    def similarity_search_with_score(self, query: str, k: int) -> list[tuple[Document, float]]:
        import psycopg
        from pgvector.psycopg import register_vector

        embedder = get_embedder()
        query_vector = embedder.embed_query(query)

        with psycopg.connect(self._conninfo) as conn:
            register_vector(conn)
            rows = conn.execute(
                """
                SELECT content, metadata,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM knowledge_chunks
                WHERE collection = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vector, self.collection, query_vector, k),
            ).fetchall()

        hits: list[tuple[Document, float]] = []
        for content, metadata, similarity in rows:
            meta = metadata if isinstance(metadata, dict) else {}
            distance = max(0.0, 2.0 - 2.0 * float(similarity))
            hits.append((Document(page_content=content, metadata=meta), distance))
        return hits


def _conninfo() -> str:
    from research_agent.config import get_settings

    return get_settings().database.postgres_sync_uri


def collection_exists(collection: str) -> bool:
    import psycopg

    try:
        with psycopg.connect(_conninfo()) as conn:
            row = conn.execute(
                "SELECT 1 FROM knowledge_chunks WHERE collection = %s LIMIT 1",
                (collection,),
            ).fetchone()
            return row is not None
    except Exception as exc:  # noqa: BLE001
        logger.warning("pgvector collection_exists failed: {}", exc)
        return False


def load_store(collection: str) -> PgVectorCollection | None:
    cached = _PG_STORES.get(collection)
    if cached is not None:
        return cached
    if not collection_exists(collection):
        return None
    store = PgVectorCollection(collection, _conninfo())
    _PG_STORES[collection] = store
    return store


def save_store(collection: str, store: PgVectorCollection) -> None:
    _PG_STORES[collection] = store


def create_from_texts(
    collection: str,
    texts: list[str],
    metadatas: list[dict[str, Any]],
) -> PgVectorCollection:
    store = PgVectorCollection(collection, _conninfo())
    store.add_texts(texts, metadatas)
    save_store(collection, store)
    return store


def chunk_count(collection: str) -> int:
    import psycopg

    try:
        with psycopg.connect(_conninfo()) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM knowledge_chunks WHERE collection = %s",
                (collection,),
            ).fetchone()
            return int(row[0]) if row else 0
    except Exception:  # noqa: BLE001
        return 0


def invalidate_cache(collection: str) -> None:
    _PG_STORES.pop(collection, None)


def delete_collection(collection: str) -> bool:
    import psycopg

    invalidate_cache(collection)
    try:
        with psycopg.connect(_conninfo()) as conn:
            cur = conn.execute(
                "DELETE FROM knowledge_chunks WHERE collection = %s",
                (collection,),
            )
            conn.commit()
            return cur.rowcount > 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("pgvector delete_collection failed: {}", exc)
        return False


def list_collections() -> list[str]:
    import psycopg

    try:
        with psycopg.connect(_conninfo()) as conn:
            rows = conn.execute(
                "SELECT DISTINCT collection FROM knowledge_chunks ORDER BY collection"
            ).fetchall()
            return [row[0] for row in rows]
    except Exception:  # noqa: BLE001
        return []
