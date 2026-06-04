"""分享 RAG 向量后端的嵌入模型单例。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIMENSION = 512

_EMBEDDER: Any | None = None


def _resolve_local_model(model_id: str) -> str:
    """如果 HuggingFace 缓存中已有该模型的 snapshot，返回本地路径；否则返回原始 model_id。"""
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = hf_cache / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if model_dir.is_dir():
        snapshots = sorted(model_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if snapshots:
            local = str(snapshots[0])
            logger.info("Using local cached model: {}", local)
            return local
    return model_id


def get_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Any:
    """返回一个缓存的 HuggingFaceEmbeddings 实例。"""
    global _EMBEDDER
    if _EMBEDDER is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        resolved = _resolve_local_model(model_name)
        logger.info("Loading embedding model: {}", resolved)
        _EMBEDDER = HuggingFaceEmbeddings(
            model_name=resolved,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBEDDER
