"""Embedding generation for vector storage."""

from __future__ import annotations

from typing import Any

from langchain_openai import OpenAIEmbeddings
from loguru import logger

from research_agent.config import LLMConfig


def create_embeddings(config: LLMConfig, provider: str = "huggingface") -> Any:
    """Create an embedding model instance.

    Supports:
    - huggingface: Local sentence-transformers (free, no API calls)
    - openai: OpenAI text-embedding-3-small (paid, higher quality)
    """
    if provider == "openai":
        logger.info("Using OpenAI embeddings")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=config.openai_api_key,
            base_url=config.openai_api_base,
        )

    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("Using local HuggingFace embeddings")
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
