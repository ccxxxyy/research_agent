"""MCP Server — Web search tool for retrieving up-to-date information."""

from __future__ import annotations

import httpx
from fastmcp import FastMCP

mcp = FastMCP("WebSearch", description="Search the web for current information")


@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return relevant results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of search results with title, url, and snippet.
    """
    # TODO: Integrate with a real search API (SerpAPI / Tavily / Bing)
    # Placeholder implementation for skeleton
    async with httpx.AsyncClient(timeout=30) as client:
        # Example: Tavily search API
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced",
            },
            headers={"Content-Type": "application/json"},
        )
        if response.status_code == 200:
            data = response.json()
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                }
                for r in data.get("results", [])
            ]
    return []


@mcp.tool()
async def fetch_webpage(url: str) -> str:
    """Fetch and extract text content from a webpage.

    Args:
        url: The URL to fetch.

    Returns:
        Extracted text content from the page.
    """
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        # Basic HTML text extraction — production should use readability
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.texts: list[str] = []
                self._skip = False

            def handle_starttag(self, tag, attrs):
                self._skip = tag in ("script", "style", "nav", "footer", "header")

            def handle_endtag(self, tag):
                if tag in ("script", "style", "nav", "footer", "header"):
                    self._skip = False

            def handle_data(self, data):
                if not self._skip:
                    text = data.strip()
                    if text:
                        self.texts.append(text)

        extractor = TextExtractor()
        extractor.feed(response.text)
        return "\n".join(extractor.texts)[:5000]


if __name__ == "__main__":
    mcp.run(transport="stdio")
