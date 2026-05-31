"""知识库的 FAISS -后端向量存储助手"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_agent.rag.embeddings import get_embedder

DEFAULT_DB_DIR = Path("./data/knowledge_db").resolve()

_FAISS_STORES: dict[str, Any] = {}


def collection_dir(collection: str, *, db_dir: Path | None = None) -> Path:
    base = db_dir or DEFAULT_DB_DIR
    return base / collection


def index_exists(collection: str, *, db_dir: Path | None = None) -> bool:
    cdir = collection_dir(collection, db_dir=db_dir)
    return (cdir / "index.faiss").exists() and (cdir / "index.pkl").exists()


def load_store(collection: str, *, db_dir: Path | None = None) -> Any | None:
    cached = _FAISS_STORES.get(collection)
    if cached is not None:
        return cached
    if not index_exists(collection, db_dir=db_dir):
        return None

    from langchain_community.vectorstores import FAISS

    cdir = collection_dir(collection, db_dir=db_dir)
    store = FAISS.load_local(
        folder_path=str(cdir),
        embeddings=get_embedder(),
        allow_dangerous_deserialization=True,
    )
    _FAISS_STORES[collection] = store
    return store


def save_store(collection: str, store: Any, *, db_dir: Path | None = None) -> None:
    cdir = collection_dir(collection, db_dir=db_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    store.save_local(folder_path=str(cdir))
    _FAISS_STORES[collection] = store


def create_from_texts(
    collection: str,
    texts: list[str],
    metadatas: list[dict[str, Any]],
    *,
    db_dir: Path | None = None,
) -> Any:
    from langchain_community.vectorstores import FAISS

    store = FAISS.from_texts(
        texts=texts,
        embedding=get_embedder(),
        metadatas=metadatas,
    )
    save_store(collection, store, db_dir=db_dir)
    return store


def chunk_count(collection: str, *, db_dir: Path | None = None) -> int:
    try:
        store = load_store(collection, db_dir=db_dir)
        if store is None:
            return 0
        return len(store.index_to_docstore_id)
    except Exception:  # noqa: BLE001
        return 0


def invalidate_cache(collection: str) -> None:
    _FAISS_STORES.pop(collection, None)


def list_collection_dirs(*, db_dir: Path | None = None) -> list[Path]:
    base = db_dir or DEFAULT_DB_DIR
    if not base.is_dir():
        return []
    return [p for p in base.iterdir() if p.is_dir() and index_exists(p.name, db_dir=db_dir)]
