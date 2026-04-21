"""MCP Server — Generic vector DB (DEPRECATED).

.. deprecated:: Phase 3
    This generic Chroma wrapper was a Phase-0 skeleton placeholder whose
    tool implementations return synthetic placeholder data. In Phase 4
    the actual RAG path is provided directly inside the LangGraph nodes
    (``graph/nodes/retriever.py``) using the shared
    ``research_agent.rag.retriever.HybridRetriever`` backed by a real
    Chroma collection of A-share research reports, not through MCP.

    Exposing Chroma as an MCP tool was ultimately deemed an unnecessary
    indirection — the retriever is a LangChain ``Runnable`` already.

    Do NOT add new tools here. Do NOT wire this server into any Agent.
"""

from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("VectorDB")


@mcp.tool()
async def vector_search(
    query: str,
    collection: str = "default",
    top_k: int = 5,
) -> list[dict]:
    """Search the vector database for semantically similar documents.

    Args:
        query: The search query.
        collection: Name of the vector collection to search.
        top_k: Number of results to return.

    Returns:
        List of matching documents with content, metadata, and similarity score.
    """
    # TODO: Connect to ChromaDB instance — placeholder returns synthetic results
    return [
        {
            "content": f"Placeholder result {i + 1} for: {query}",
            "metadata": {"collection": collection},
            "score": 0.0,
        }
        for i in range(top_k)
    ]


@mcp.tool()
async def add_to_knowledge_base(
    content: str,
    metadata: dict | None = None,
    collection: str = "default",
) -> dict:
    """Add a document to the vector knowledge base.

    Args:
        content: The text content to store.
        metadata: Optional metadata dictionary (source, author, date, etc.).
        collection: Target vector collection name.

    Returns:
        Confirmation with document ID.
    """
    # TODO: Implement with ChromaDB client
    return {
        "status": "stored",
        "collection": collection,
        "metadata": metadata or {},
        "content_preview": content[:100],
    }


@mcp.tool()
async def list_collections() -> list[str]:
    """List all available vector collections.

    Returns:
        List of collection names.
    """
    # TODO: Implement with ChromaDB client
    return ["default"]


if __name__ == "__main__":
    mcp.run(transport="stdio")
