"""MCP Server — Document parsing tools for PDF, Markdown, and text files."""

from __future__ import annotations

from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("DocumentParser", description="Parse and extract content from documents")


@mcp.tool()
async def parse_pdf(file_path: str) -> str:
    """Parse a PDF file and extract its text content.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted text content from the PDF.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(path))
    pages = await loader.aload()
    return "\n\n".join(page.page_content for page in pages)


@mcp.tool()
async def parse_markdown(file_path: str) -> dict:
    """Parse a Markdown file and extract structured sections.

    Args:
        file_path: Path to the Markdown file.

    Returns:
        Dictionary with extracted sections and metadata.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    content = path.read_text(encoding="utf-8")

    sections: dict[str, str] = {}
    current_heading = "introduction"
    current_lines: list[str] = []

    for line in content.split("\n"):
        if line.startswith("#"):
            if current_lines:
                sections[current_heading] = "\n".join(current_lines).strip()
            current_heading = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_heading] = "\n".join(current_lines).strip()

    return {
        "file_name": path.name,
        "total_sections": len(sections),
        "sections": sections,
    }


@mcp.tool()
async def extract_tables(file_path: str) -> list[dict]:
    """Extract tabular data from a document.

    Args:
        file_path: Path to the document file.

    Returns:
        List of extracted tables as dictionaries.
    """
    # TODO: Implement table extraction with camelot-py or tabula-py
    return [{"status": "not_implemented", "file": file_path}]


if __name__ == "__main__":
    mcp.run(transport="stdio")
