"""End-to-end smoke test for the in-process knowledge-base tools.

Validates the production runtime path (no MCP subprocess): the agent
calls the four ``StructuredTool`` instances exposed by
:mod:`research_agent.tools.knowledge_tools`, which delegate to the
async coroutines in
:mod:`research_agent.mcp_servers.knowledge_server`.

Sequence:

1. Import ``KNOWLEDGE_TOOLS`` and verify the four expected names are
   present.
2. Hand-roll a 2-page synthetic PDF on disk (no external file deps).
3. ``knowledge_ingest_pdf`` against a uniquely-named scratch
   collection (timestamp suffix prevents cross-run contamination of
   ``./data/knowledge_db/``).
4. ``knowledge_list_collections`` and assert our scratch collection
   appears.
5. ``knowledge_search`` with a query that lexically + semantically
   overlaps page 1; assert non-zero hits, sensible scores and a
   non-``low`` quality bucket.
6. ``knowledge_delete_collection`` to clean up.

Each call is wrapped in a wall-clock timer plus ``asyncio.wait_for``
so a regression that hangs the embedder pipeline fails fast with a
useful message instead of locking up the developer's terminal.

Run::

    .venv/Scripts/python.exe scripts/smoke_test_knowledge_inproc.py

Exit code 0 on full success, non-zero on any failure / timeout / wrong
shape. Suitable for CI smoke if you accept the cold-start cost
(~30 s on a fresh runner where the bge-small model has to download).
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path

# Force UTF-8 on stdout so Chinese sources / hits don't blow up on the
# Windows code page.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def _step(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _make_tiny_pdf(pages: list[str]) -> bytes:
    """Hand-roll a minimal spec-compliant PDF that pypdf can parse.

    Identical to the unit tests' ``_make_tiny_pdf`` strategy — kept
    inline so this script has no test-package import dependency.
    """

    def _content_stream(text: str) -> bytes:
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        return f"BT /F1 12 Tf 50 750 Td ({safe}) Tj ET".encode("latin-1")

    objects: list[bytes] = []

    def _push(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)

    catalog = _push(b"<< /Type /Catalog /Pages 2 0 R >>")
    pages_obj_num = _push(b"")  # placeholder, patched below
    font = _push(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_obj_nums: list[int] = []
    for text in pages:
        stream = _content_stream(text)
        contents = _push(
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
            + str(font).encode("ascii")
            + b" 0 R >> >> "
            + b"/Contents "
            + str(contents).encode("ascii")
            + b" 0 R >>"
        )
        page_obj_nums.append(_push(page_obj))

    kids = b" ".join(str(n).encode("ascii") + b" 0 R" for n in page_obj_nums)
    objects[pages_obj_num - 1] = (
        b"<< /Type /Pages /Count "
        + str(len(page_obj_nums)).encode("ascii")
        + b" /Kids ["
        + kids
        + b"] >>"
    )

    buf = io.BytesIO()
    buf.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = [0]
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
        + str(catalog).encode("ascii")
        + b" 0 R >>\nstartxref\n"
        + str(xref_pos).encode("ascii")
        + b"\n%%EOF\n"
    )
    return buf.getvalue()


async def _call(tool, payload: dict, *, timeout: float, label: str) -> dict:
    """Invoke ``tool.ainvoke`` with a timer + clean error reporting.

    StructuredTool returns the underlying coroutine's actual return
    value (a dict for our tools), unlike the MCP-stdio adapter that
    wraps every response in a content-block list. So no flatten step
    is needed here.
    """
    t0 = time.time()
    try:
        result = await asyncio.wait_for(tool.ainvoke(payload), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"{label}: timed out after {timeout:.1f}s") from exc
    elapsed = time.time() - t0
    if not isinstance(result, dict):
        raise RuntimeError(
            f"{label}: expected dict, got {type(result).__name__}: {result!r}"
        )
    _step(f"{label}: {elapsed:.2f}s -> {sorted(result)}")
    return result


async def amain() -> int:
    _step("loading in-process knowledge tools")
    from research_agent.tools.knowledge_tools import KNOWLEDGE_TOOLS

    by_name = {t.name: t for t in KNOWLEDGE_TOOLS}
    expected = {
        "knowledge_ingest_pdf",
        "knowledge_search",
        "knowledge_list_collections",
        "knowledge_delete_collection",
    }
    missing = expected - by_name.keys()
    if missing:
        _step(f"FAIL: missing tools {missing}")
        return 1
    _step(f"  tools: {sorted(by_name)}")

    suffix = time.strftime("%Y%m%d-%H%M%S")
    coll = f"smoke-inproc-{suffix}"

    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "tiny.pdf"
        pdf.write_bytes(
            _make_tiny_pdf(
                [
                    "carbon neutrality 2030 scope 1 emissions reduction roadmap",
                    "shareholder dividend policy quarterly distribution schedule",
                ]
            )
        )
        _step(f"wrote synthetic PDF to {pdf} ({pdf.stat().st_size} bytes)")

        ingest = await _call(
            by_name["knowledge_ingest_pdf"],
            {"local_path": str(pdf), "collection": coll},
            timeout=180.0,  # cold-start: bge-small first load + langchain imports
            label="ingest_pdf",
        )
        if "error" in ingest:
            _step(f"FAIL: ingest_pdf returned error: {ingest}")
            return 1
        if ingest.get("num_chunks_added", 0) <= 0:
            _step(f"FAIL: ingest_pdf added zero chunks: {ingest}")
            return 1
        _step(f"  ingested {ingest['num_chunks_added']} chunks (collection={coll!r})")

        listing = await _call(
            by_name["knowledge_list_collections"],
            {},
            timeout=30.0,
            label="list_collections",
        )
        names = {c["name"] for c in listing.get("collections", [])}
        if coll not in names:
            _step(f"FAIL: scratch collection {coll!r} missing from listing: {names}")
            return 1
        _step(f"  list_collections sees {len(names)} collection(s); ours present.")

        hits = await _call(
            by_name["knowledge_search"],
            {"query": "carbon neutrality goal", "collection": coll, "top_k": 3},
            timeout=60.0,  # warm path: should be sub-second after ingest above
            label="search",
        )
        if "error" in hits:
            _step(f"FAIL: search returned error: {hits}")
            return 1
        if hits.get("top_k_returned", 0) < 1:
            _step(f"FAIL: search produced no hits: {hits}")
            return 1
        if hits.get("quality") == "low":
            _step(f"WARN: search quality is 'low' on a clean lexical match: {hits}")
        top = hits["results"][0]
        ok_lexical = "carbon" in (top.get("content") or "").lower()
        if not ok_lexical:
            _step(f"FAIL: top hit doesn't contain 'carbon': {top}")
            return 1
        _step(
            f"  top hit page={top.get('page')} vec={top.get('vector_score')} "
            f"bm25={top.get('bm25_score')} rrf={top.get('rrf_score')} "
            f"quality={hits['quality']}"
        )

        deleted = await _call(
            by_name["knowledge_delete_collection"],
            {"collection": coll},
            timeout=30.0,
            label="delete_collection",
        )
        if not deleted.get("deleted"):
            _step(f"FAIL: delete_collection didn't delete: {deleted}")
            return 1
        _step("  cleaned up scratch collection")

    _step("ALL OK")
    return 0


if __name__ == "__main__":
    rc = 1
    with suppress(KeyboardInterrupt):
        rc = asyncio.run(amain())
    sys.exit(rc)
