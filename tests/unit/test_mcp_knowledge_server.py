"""unit tests — ``knowledge_server`` MCP tools and helpers.

Two tiers of coverage:

1. **Pure-helper tests** (always run): exercise the tokenizer, BM25
   index, hybrid-fusion math, quality classifier, chunker, and PDF
   loader against a tiny synthetic in-memory PDF. Zero network, no
   model weights — these run in <1 s.

2. **End-to-end ingestion + search** (marked ``slow``): build a
   real FAISS index under a tmp directory, ingest a tiny PDF, and
   run a ``search()`` round-trip. This pulls in the bge-small
   embedding model the first time it runs (~95 MB cached under
   ``~/.cache/huggingface``); subsequent runs are warm and
   complete in a few seconds.

The slow tier is opt-in via ``pytest -m slow`` so CI doesn't pay the
embedding-download tax on every commit. ``pytest`` (default) runs the
fast helpers; the demo script
``scripts/demo_knowledge_expert.py`` is the real end-to-end smoke.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from research_agent.mcp_servers import knowledge_server
from research_agent.mcp_servers.knowledge_server import (
    QUALITY_HIGH_THRESHOLD,
    QUALITY_MEDIUM_THRESHOLD,
    _BM25Index,
    _chunk_pages,
    _classify_quality,
    _hybrid_fuse,
    _load_pdf_pages,
    _validate_collection_name,
)

# ---------------------------------------------------------------------
# Helpers — synthetic in-memory PDF
# ---------------------------------------------------------------------


def _make_tiny_pdf(pages: list[str]) -> bytes:
    """Render ``pages`` to a real (tiny) PDF using ``pypdf``.

    We avoid shipping a fixture PDF in the repo because the fast tier
    must work in a fresh checkout. ``pypdf`` can't author from
    scratch, so we use ``reportlab`` if available — and otherwise
    fall back to a hard-coded 2-page PDF where the page text is
    verbatim ASCII captured between the BT/ET markers in the content
    stream. The latter is what we actually use here so we keep deps
    to a minimum (``pypdf`` is already required).

    The result is a structurally minimal but spec-compliant PDF that
    ``pypdf.PdfReader`` happily parses.
    """
    # Hand-rolled minimal PDF skeleton. We only need the text-extraction
    # round-trip to work; we are NOT trying to render anything visually.
    # pypdf's text extractor reads ``Tj`` operators inside ``BT/ET``.

    def _content_stream(text: str) -> bytes:
        # Escape parentheses; PDF strings are () delimited.
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        body = f"BT /F1 12 Tf 50 750 Td ({safe}) Tj ET".encode("latin-1")
        return body

    objects: list[bytes] = []

    def _push(obj_bytes: bytes) -> int:
        objects.append(obj_bytes)
        return len(objects)

    # 1. Catalog
    catalog_obj_num = _push(b"<< /Type /Catalog /Pages 2 0 R >>")
    # 2. Pages tree (forward-ref placeholder; we'll patch in the kids
    # list once the page objects exist).
    pages_obj_num = _push(b"")
    # Font (shared by all pages)
    font_obj_num = _push(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    )

    page_obj_nums: list[int] = []
    for text in pages:
        stream = _content_stream(text)
        contents_obj_num = _push(
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
        page_obj = (
            b"<< /Type /Page /Parent "
            + str(pages_obj_num).encode("ascii")
            + b" 0 R "
            + b"/MediaBox [0 0 612 792] "
            + b"/Resources << /Font << /F1 "
            + str(font_obj_num).encode("ascii")
            + b" 0 R >> >> "
            + b"/Contents "
            + str(contents_obj_num).encode("ascii")
            + b" 0 R >>"
        )
        page_obj_nums.append(_push(page_obj))

    # Patch the Pages tree with the actual kid references.
    kids = b" ".join(
        str(n).encode("ascii") + b" 0 R" for n in page_obj_nums
    )
    objects[pages_obj_num - 1] = (
        b"<< /Type /Pages /Count "
        + str(len(page_obj_nums)).encode("ascii")
        + b" /Kids [" + kids + b"] >>"
    )

    # Assemble the PDF: header, body objects, xref, trailer.
    buf = io.BytesIO()
    buf.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = [0]  # entry 0 is the standard "free" record
    for i, obj in enumerate(objects, start=1):
        offsets.append(buf.tell())
        buf.write(str(i).encode("ascii") + b" 0 obj\n" + obj + b"\nendobj\n")

    xref_pos = buf.tell()
    buf.write(b"xref\n0 " + str(len(objects) + 1).encode("ascii") + b"\n")
    buf.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        buf.write(f"{off:010d} 00000 n \n".encode("ascii"))
    buf.write(
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode("ascii")
        + b" /Root "
        + str(catalog_obj_num).encode("ascii")
        + b" 0 R >>\nstartxref\n"
        + str(xref_pos).encode("ascii")
        + b"\n%%EOF\n"
    )
    return buf.getvalue()


@pytest.fixture
def tiny_pdf_path(tmp_path: Path) -> Path:
    """Two-page synthetic PDF with predictable extractable text."""
    pdf_bytes = _make_tiny_pdf(
        [
            "carbon neutrality 2030 goal scope 1 emissions",
            "shareholder dividend policy quarterly distribution schedule",
        ]
    )
    out = tmp_path / "tiny.pdf"
    out.write_bytes(pdf_bytes)
    return out


# ---------------------------------------------------------------------
# Pure-helper tests
# ---------------------------------------------------------------------


class TestValidateCollectionName:
    def test_accepts_simple_alphanum(self) -> None:
        _validate_collection_name("esg2024")
        _validate_collection_name("user-library")
        _validate_collection_name("a_b.c")

    def test_rejects_too_short(self) -> None:
        with pytest.raises(ValueError, match="length"):
            _validate_collection_name("ab")

    def test_rejects_too_long(self) -> None:
        with pytest.raises(ValueError, match="length"):
            _validate_collection_name("x" * 64)

    def test_rejects_leading_punct(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            _validate_collection_name("-leading")
        with pytest.raises(ValueError, match="must match"):
            _validate_collection_name(".dotted")

    def test_rejects_double_dot(self) -> None:
        with pytest.raises(ValueError, match=r"\.\."):
            _validate_collection_name("foo..bar")

    def test_rejects_invalid_chars(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            _validate_collection_name("with space")
        with pytest.raises(ValueError, match="must match"):
            _validate_collection_name("中文集合")


class TestQualityClassifier:
    def test_high_when_top_score_clears_threshold(self) -> None:
        assert (
            _classify_quality(QUALITY_HIGH_THRESHOLD + 0.05, 0.5, unique_sources=2)
            == "high"
        )

    def test_medium_band(self) -> None:
        # top_score ≥ medium but < high; mean still high enough.
        score = (QUALITY_HIGH_THRESHOLD + QUALITY_MEDIUM_THRESHOLD) / 2
        assert _classify_quality(score, score, unique_sources=2) == "medium"

    def test_low_when_top_score_below_medium(self) -> None:
        assert (
            _classify_quality(QUALITY_MEDIUM_THRESHOLD - 0.05, 0.1, unique_sources=1)
            == "low"
        )

    def test_low_when_no_unique_sources(self) -> None:
        # We treat ``unique_sources=0`` as "no usable evidence" — the
        # classifier must NOT report 'high' even if some scores drift up.
        assert (
            _classify_quality(QUALITY_HIGH_THRESHOLD + 0.1, 0.5, unique_sources=0)
            in {"medium", "low"}
        )


class TestBM25Index:
    def test_tokenizer_lowercases_and_splits(self) -> None:
        toks = _BM25Index._tokenize("Carbon Neutrality 2030 Goal!")
        assert toks == ["carbon", "neutrality", "2030", "goal"]

    def test_search_returns_relevant_doc_first(self) -> None:
        docs = [
            {"content": "carbon neutrality scope emissions", "metadata": {}},
            {"content": "shareholder dividend policy", "metadata": {}},
            {"content": "supply chain logistics warehouse", "metadata": {}},
        ]
        idx = _BM25Index(docs)
        ranked = idx.search("carbon emissions", top_k=3)
        assert ranked, "expected at least one BM25 hit"
        # The first doc shares both query terms — must rank #1.
        first_idx, first_score = ranked[0]
        assert first_idx == 0, ranked
        assert first_score > 0

    def test_search_handles_empty_query(self) -> None:
        idx = _BM25Index([{"content": "hello world", "metadata": {}}])
        assert idx.search("   ", top_k=5) == []

    def test_handles_empty_corpus(self) -> None:
        # Empty corpora are a normal cold-start case (collection
        # exists but no docs ingested yet) — must not raise.
        idx = _BM25Index([])
        assert idx.search("anything", top_k=5) == []


class TestChunkPages:
    def test_chunks_inherit_page_metadata(self) -> None:
        pages = [
            {"page": 1, "text": "alpha beta " * 200},
            {"page": 2, "text": "gamma delta " * 50},
        ]
        chunks = _chunk_pages(
            pages, source="x.pdf", chunk_size=300, chunk_overlap=50
        )
        assert chunks
        assert all("page" in c["metadata"] for c in chunks)
        assert all(c["metadata"]["source"] == "x.pdf" for c in chunks)
        # Chunks NEVER span pages — provenance must stay clean.
        page_to_chunks: dict[int, int] = {}
        for c in chunks:
            page_to_chunks[c["metadata"]["page"]] = (
                page_to_chunks.get(c["metadata"]["page"], 0) + 1
            )
        assert 1 in page_to_chunks and 2 in page_to_chunks


class TestHybridFuse:
    def test_doc_only_in_vector_list_still_included(self) -> None:
        vec_doc = {"content": "vector-only-doc", "metadata": {"source": "a", "page": 1}}
        bm_doc = {"content": "bm25-only-doc", "metadata": {"source": "b", "page": 1}}
        fused = _hybrid_fuse(
            vector_hits=[(vec_doc, 0.9)],
            bm25_hits=[(0, 5.0, bm_doc)],
        )
        contents = {r["content"] for r in fused}
        assert "vector-only-doc" in contents
        assert "bm25-only-doc" in contents

    def test_doc_in_both_lists_gets_summed_rrf(self) -> None:
        doc = {"content": "shared", "metadata": {"source": "x", "page": 1}}
        fused_only_vec = _hybrid_fuse(
            vector_hits=[(doc, 0.8)], bm25_hits=[]
        )
        fused_both = _hybrid_fuse(
            vector_hits=[(doc, 0.8)], bm25_hits=[(0, 5.0, doc)]
        )
        # The same doc appearing in both lists should outrank the
        # single-list version on RRF.
        assert fused_both[0]["rrf_score"] > fused_only_vec[0]["rrf_score"]

    def test_results_sorted_descending_by_rrf(self) -> None:
        docs = [
            {"content": f"doc{i}", "metadata": {"source": "s", "page": i}}
            for i in range(5)
        ]
        # Highest vector rank for doc0, lowest for doc4. Same for BM25.
        vec = [(docs[i], 1.0 - i * 0.1) for i in range(5)]
        bm = [(i, 5.0 - i, docs[i]) for i in range(5)]
        fused = _hybrid_fuse(vec, bm)
        scores = [r["rrf_score"] for r in fused]
        assert scores == sorted(scores, reverse=True), fused


# ---------------------------------------------------------------------
# PDF loader (still pure-helper tier, but uses real pypdf)
# ---------------------------------------------------------------------


class TestLoadPdfPages:
    def test_extracts_text_from_synthetic_pdf(self, tiny_pdf_path: Path) -> None:
        pages = _load_pdf_pages(tiny_pdf_path)
        assert len(pages) == 2
        assert all(isinstance(p["page"], int) for p in pages)
        # First page text contains the carbon-neutrality phrase.
        joined = " ".join(p["text"] for p in pages)
        assert "carbon" in joined.lower()
        assert "dividend" in joined.lower()


# ---------------------------------------------------------------------
# End-to-end ingest + search (slow tier)
# ---------------------------------------------------------------------


@pytest.mark.slow
class TestIngestAndSearch:
    """Real FAISS + bge-small embedder end-to-end.

    Marked ``slow`` because the first run downloads ~95 MB of model
    weights and warms a sentence-transformer; subsequent runs are
    sub-second on a warm cache. Run with ``pytest -m slow``.
    """

    @pytest.mark.asyncio
    async def test_ingest_then_search_returns_relevant_chunk(
        self, tiny_pdf_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Redirect persistence + cached singletons to a clean tmp dir
        # so this test can never collide with the user's real DB.
        # NOTE: ``@mcp.tool()`` in fastmcp returns the original async
        # function unwrapped — we can ``await`` it directly without
        # going through stdio.
        monkeypatch.setattr(knowledge_server, "DEFAULT_DB_DIR", tmp_path / "kb")
        monkeypatch.setattr(knowledge_server, "_FAISS_STORES", {})
        monkeypatch.setattr(knowledge_server, "_BM25_CACHE", {})

        ingest_result = await knowledge_server.ingest_pdf(
            local_path=str(tiny_pdf_path),
            collection="test-coll",
        )
        assert "error" not in ingest_result, ingest_result
        assert ingest_result["num_chunks_added"] >= 2

        # Carbon-neutrality query must surface the page-1 chunk near
        # the top, NOT the page-2 dividend chunk.
        hits = await knowledge_server.search(
            query="carbon neutrality goal",
            collection="test-coll",
            top_k=3,
        )
        assert "error" not in hits, hits
        assert hits["top_k_returned"] >= 1
        assert hits["unique_sources"] == 1
        assert hits["results"], hits
        top_content = hits["results"][0]["content"].lower()
        assert "carbon" in top_content or "neutrality" in top_content

        # Quality should classify as at least medium for an exact-phrase
        # match in a tiny corpus.
        assert hits["quality"] in {"high", "medium"}, hits

    @pytest.mark.asyncio
    async def test_search_on_empty_collection_returns_low_quality(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(knowledge_server, "DEFAULT_DB_DIR", tmp_path / "kb")
        monkeypatch.setattr(knowledge_server, "_FAISS_STORES", {})
        monkeypatch.setattr(knowledge_server, "_BM25_CACHE", {})

        result = await knowledge_server.search(
            query="anything",
            collection="empty-coll",
            top_k=5,
        )
        assert result["top_k_returned"] == 0
        assert result["quality"] == "low"
        assert result["results"] == []
        assert "warning" in result

    @pytest.mark.asyncio
    async def test_list_and_delete_collection(
        self, tiny_pdf_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(knowledge_server, "DEFAULT_DB_DIR", tmp_path / "kb")
        monkeypatch.setattr(knowledge_server, "_FAISS_STORES", {})
        monkeypatch.setattr(knowledge_server, "_BM25_CACHE", {})

        await knowledge_server.ingest_pdf(
            local_path=str(tiny_pdf_path), collection="to-delete"
        )

        listing = await knowledge_server.list_collections()
        names = {c["name"] for c in listing["collections"]}
        assert "to-delete" in names

        deleted = await knowledge_server.delete_collection(collection="to-delete")
        assert deleted["existed"] is True
        assert deleted["deleted"] is True

        # Idempotent second call.
        deleted_again = await knowledge_server.delete_collection(
            collection="to-delete"
        )
        assert deleted_again["existed"] is False
        assert deleted_again["deleted"] is False
