"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from research_agent.api.routes import chat, health, knowledge, research
from research_agent.config import get_settings
from research_agent.observability.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and tear down application resources."""
    settings = get_settings()
    setup_logging(settings.observability.log_level)

    from research_agent.memory.checkpointer import init_checkpointer
    from research_agent.memory.store import init_memory_store

    checkpointer = await init_checkpointer(settings.database.postgres_sync_uri)
    memory_store = await init_memory_store(settings.database.postgres_sync_uri)

    from research_agent.graph.supervisor import build_research_graph
    from research_agent.llm.provider import ModelRouter
    from research_agent.rag.embedder import create_embeddings
    from research_agent.rag.retriever import HybridRetriever

    from langchain_chroma import Chroma

    model_router = ModelRouter(settings.llm)

    embeddings = create_embeddings(settings.llm)
    vectorstore = Chroma(
        collection_name="research_agent",
        embedding_function=embeddings,
    )
    hybrid_retriever = HybridRetriever(vectorstore=vectorstore)

    graph = build_research_graph(
        model_router=model_router,
        hybrid_retriever=hybrid_retriever,
        checkpointer=checkpointer,
        memory_store=memory_store,
    )

    app.state.graph = graph
    app.state.model_router = model_router
    app.state.memory_store = memory_store
    app.state.settings = settings

    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Research Agent",
        description="Multi-agent deep research system with LangGraph, MCP, and Agentic RAG",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,  # type: ignore[arg-type]
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(research.router)
    app.include_router(chat.router)
    app.include_router(knowledge.router)

    return app


app = create_app()


def cli() -> None:
    settings = get_settings()
    uvicorn.run(
        "research_agent.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.is_dev,
    )


if __name__ == "__main__":
    cli()
