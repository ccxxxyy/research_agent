"""分享 RAG 向量后端的嵌入模型单例。"""

from __future__ import annotations

from typing import Any

from loguru import logger

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIMENSION = 512

_EMBEDDER: Any | None = None


def get_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Any:
    """返回一个缓存的 HuggingFaceEmbeddings 实例。"""
    global _EMBEDDER
    if _EMBEDDER is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Loading embedding model: {}", model_name)
        _EMBEDDER = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBEDDER
