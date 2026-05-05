"""end-to-end demo — knowledge_expert over a real PDF + Chroma.

This script is the **production flow-of-the-day for RAG**: it spawns
the ``knowledge_server`` MCP subprocess, wires its tools into a
``knowledge_expert`` specialist registered behind the research
supervisor, and asks a question that forces the agent to:

  1. ``knowledge_ingest_pdf`` — chunk + embed a freshly-written PDF
     into a brand-new Chroma collection.
  2. ``knowledge_search``     — hybrid (vector + BM25) retrieval.
  3. answer with a quoted excerpt + ``source`` + ``page``.

Run::

    uv run python scripts/demo_knowledge_expert.py

Requirements:
  - A working LLM config in ``.env`` (same file used by Phase-1/3
    smokes — DeepSeek / OpenAI / etc.). NO third-party network
    dependency for the data plane: the demo PDF is synthesised
    in-process so it runs offline once the embedding model weights
    are warm-cached.
  - First run downloads ``BAAI/bge-small-zh-v1.5`` (~95 MB) into
    ``~/.cache/huggingface``. Subsequent runs are instant.

Exit codes:
    0 → supervisor produced a final answer that passes soft checks.
    1 → any configuration / MCP / LLM error, OR the supervisor
        failed to route to ``knowledge_expert``, OR the final
        answer doesn't reference the seeded PDF content.

Soft checks (intentionally lenient — we do NOT pin LLM wording):
  * non-empty final answer
  * ``transfer_to_knowledge_expert`` appears in the supervisor trace
  * the answer cites the PDF (substring match on the path stem) OR a
    page number

Why we don't observe ``knowledge_*`` tool calls directly
--------------------------------------------------------
Same reason as ``demo_financial_research.py``: the supervisor
compiles with ``output_mode="last_message"``, so each specialist's
internal ``ToolMessage`` traffic stays inside its subgraph. We trust
the routing record and verify *correctness via final-answer content*.
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from loguru import logger

# Make the demo log unambiguously human-readable on Windows: force
# UTF-8 stdout/stderr so the Chinese prompt in the log file does not
# end up as mojibake when this script is invoked through PowerShell's
# ``>`` redirection.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    # ``reconfigure`` exists on TextIOWrapper since Py 3.7; if the
    # stream is already wrapped (e.g. uvicorn) we just skip it.
    pass


def _step(msg: str) -> None:
    """Print a one-line, time-stamped progress beacon.

    Used aggressively in this demo because it has historically hung
    in opaque places (Chroma SQLite locks, MCP subprocess re-spawn,
    LLM stream stall). With these beacons every step is visible
    even when stdout is redirected to a file and the process is
    killed mid-flight.
    """
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

from research_agent.config import get_settings
from research_agent.graph.research_supervisor import build_research_supervisor
from research_agent.llm.provider import ModelRouter
from research_agent.mcp_servers.client_factory import (
    load_knowledge_tools_inproc,
)


# ---------------------------------------------------------------------
# Synthetic-PDF builder
#
# We hand-roll the PDF rather than depending on reportlab / weasyprint
# so the demo is fully self-contained. The same trick is used by the
# unit tests for knowledge_server. The output is a structurally
# minimal but spec-compliant PDF that pypdf parses cleanly.
# ---------------------------------------------------------------------

PDF_PAGES = [
    # Page 1 — the answer-bearing chunk.
    "This is the 2024 ESG report for ExampleCorp. "
    "Our carbon neutrality goal is to achieve net-zero scope 1 and "
    "scope 2 emissions by the year 2030, with a 50 percent absolute "
    "reduction milestone by 2027. The board reviews progress annually.",
    # Page 2 — a deliberate distractor on a totally different topic
    # so the retriever has to actually discriminate.
    "Shareholder dividend policy: ExampleCorp distributes quarterly "
    "cash dividends of 0.12 USD per share, subject to free-cash-flow "
    "coverage. Dividend reinvestment plans are available to holders "
    "of record on the ex-dividend date.",
]


def _make_tiny_pdf(pages: list[str]) -> bytes:
    """Return raw bytes of a minimal multi-page PDF.

    Each page contains a single ``Tj`` text-show op so ``pypdf``
    reproduces the input string verbatim during ``extract_text()``.
    """

    def _content_stream(text: str) -> bytes:
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        return f"BT /F1 12 Tf 50 750 Td ({safe}) Tj ET".encode("latin-1")

    objects: list[bytes] = []

    def _push(obj_bytes: bytes) -> int:
        objects.append(obj_bytes)
        return len(objects)

    catalog_obj_num = _push(b"<< /Type /Catalog /Pages 2 0 R >>")
    pages_obj_num = _push(b"")  # back-patched once page objects exist
    font_obj_num = _push(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_obj_nums: list[int] = []
    for text in pages:
        stream = _content_stream(text)
        contents_obj_num = _push(
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n"
            + stream + b"\nendstream"
        )
        page_obj = (
            b"<< /Type /Page /Parent " + str(pages_obj_num).encode("ascii") + b" 0 R "
            + b"/MediaBox [0 0 612 792] "
            + b"/Resources << /Font << /F1 " + str(font_obj_num).encode("ascii") + b" 0 R >> >> "
            + b"/Contents " + str(contents_obj_num).encode("ascii") + b" 0 R >>"
        )
        page_obj_nums.append(_push(page_obj))

    kids = b" ".join(str(n).encode("ascii") + b" 0 R" for n in page_obj_nums)
    objects[pages_obj_num - 1] = (
        b"<< /Type /Pages /Count " + str(len(page_obj_nums)).encode("ascii")
        + b" /Kids [" + kids + b"] >>"
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
        b"trailer\n<< /Size " + str(len(objects) + 1).encode("ascii")
        + b" /Root " + str(catalog_obj_num).encode("ascii")
        + b" 0 R >>\nstartxref\n" + str(xref_pos).encode("ascii") + b"\n%%EOF\n"
    )
    return buf.getvalue()


# ---------------------------------------------------------------------
# Trace helpers (copied from demo_financial_research.py — same idiom)
# ---------------------------------------------------------------------
def _last_plain_assistant(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            tc = getattr(m, "tool_calls", None) or []
            if not tc and m.content:
                return str(m.content)
    return ""


def _trace_tool_calls(messages: list) -> list[str]:
    names: list[str] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            nm = getattr(m, "name", None) or ""
            if nm:
                names.append(nm)
        elif isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                nm = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if nm:
                    names.append(str(nm))
    return names


def _transfers_reached(calls: list[str]) -> set[str]:
    reached: set[str] = set()
    for n in calls:
        if n.startswith("transfer_to_") and n != "transfer_to_supervisor":
            reached.add(n[len("transfer_to_") :])
    return reached


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------
def _make_collection_name() -> str:
    """Return a fresh collection name unique to this run.

    Using a per-run name (timestamp suffix) sidesteps two real
    failure modes we have seen on Windows:

      a) **SQLite writer deadlock.** The MCP subprocess and the
         parent process must NOT both open Chroma's persistent
         SQLite file at the same time — the second writer can
         block indefinitely. By giving every run a fresh
         collection AND avoiding any in-process Chroma access,
         only ONE process (the MCP subprocess) ever touches the
         persistent store.

      b) **Stale state from previous runs.** Re-using the same
         collection name across runs would silently double-ingest
         the PDF on retry, polluting the BM25 stats and making
         "high-quality" thresholds easier to spuriously hit.
    """
    return f"demo_esg_{time.strftime('%Y%m%d_%H%M%S')}"


def _seed_pdf() -> Path:
    """Write the synthetic PDF under a system tempdir and return the path."""
    tmp_dir = Path(tempfile.gettempdir()) / "research_agent_demo"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / "examplecorp_esg_2024.pdf"
    out.write_bytes(_make_tiny_pdf(PDF_PAGES))
    _step(f"Seeded synthetic PDF: {out}")
    return out


# Hard ceiling for the supervisor's whole journey. The model has to
# complete:
#   ingest_pdf (cold-start embedder ~3s)
#   + 1-2 search calls (sub-second each, warm)
#   + LLM rounds for routing, planning, and the final synthesis.
# 4 minutes is plenty of headroom while still failing loudly if a
# subprocess deadlocks.
GRAPH_TIMEOUT_SECONDS: float = 240.0


async def main() -> int:
    settings = get_settings()
    router = ModelRouter(settings.llm)
    _step("Settings + ModelRouter initialised")

    pdf_path = _seed_pdf()
    collection = _make_collection_name()
    _step(f"Per-run collection name: {collection}")

    try:
        knowledge_tools = await load_knowledge_tools_inproc()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load in-process knowledge tools: {}", exc)
        return 1
    _step(f"knowledge tools loaded (in-process): count={len(knowledge_tools)}")
    for t in knowledge_tools:
        _step(f"  tool: {t.name}")

    graph = build_research_supervisor(
        model_router=router,
        knowledge_tools=knowledge_tools,
    )
    _step("Supervisor compiled (knowledge-only team)")

    question = (
        f"我刚把一份本地 PDF 路径 `{pdf_path}` 准备好了，请帮我：\n"
        f"  1) 把它灌入我的知识库（collection 名为 `{collection}`）；\n"
        "  2) 然后在该 collection 中检索 “2030 年碳中和目标” 相关原文，"
        "    返回最相关的 1-2 段，并在每段后用 `(source=…, page=…)` 标注出处；\n"
        "  3) 用一句中文总结公司的碳中和承诺要点。\n"
        "如果首次检索质量不高，请改写关键词后再试一次再回答。"
    )
    _step(f"Sending question (len={len(question)} chars) to supervisor...")
    print("---- prompt begin ----")
    print(question)
    print("---- prompt end ----", flush=True)

    try:
        result = await asyncio.wait_for(
            graph.ainvoke(
                {"messages": [HumanMessage(content=question)]},
                # ingest may take a few seconds; allow generous headroom.
                config={"recursion_limit": 40},
            ),
            timeout=GRAPH_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error(
            "Graph timed out after {}s — likely an MCP subprocess deadlock or "
            "an LLM stream stall. Re-run with finer logging if this repeats.",
            GRAPH_TIMEOUT_SECONDS,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Graph invocation crashed: {}", exc)
        return 1
    _step("Supervisor returned a final state")

    messages = result["messages"]
    final = _last_plain_assistant(messages)
    calls = _trace_tool_calls(messages)
    reached = _transfers_reached(calls)

    print("\n=== Final supervisor answer ===\n")
    print(final if final else "<empty>")
    print("\n=== Trace summary ===")
    print(f"  total messages              : {len(messages)}")
    print(f"  specialists reached         : {sorted(reached) or ['<none>']}")
    print(f"  total tool-name events      : {len(calls)}")
    print(f"  transfer_to_knowledge_expert: "
          f"{sum(1 for n in calls if n == 'transfer_to_knowledge_expert')}")

    print("\n=== Heuristic verification ===")
    ok_answer = bool(final.strip())
    ok_route = "knowledge_expert" in reached
    ok_topic = (
        "carbon" in final.lower()
        or "neutrality" in final.lower()
        or "碳中和" in final
        or "2030" in final
    )
    ok_citation = (
        "examplecorp_esg_2024" in final.lower()
        or "page" in final.lower()
        or "page=" in final.lower()
        or "p." in final.lower()
        or "页" in final
    )
    print(f"  non-empty final answer       : {ok_answer}")
    print(f"  routed to knowledge_expert   : {ok_route}")
    print(f"  topic mentioned (carbon/2030): {ok_topic}")
    print(f"  citation hint in answer      : {ok_citation}")

    if ok_answer and ok_route and ok_topic and ok_citation:
        print("\n  [PASS] knowledge_expert end-to-end RAG flow validated.")
        return 0
    print("\n  [WARN] Heuristic checks failed — inspect trace above.")
    return 1


if __name__ == "__main__":
    # NOTE: we deliberately do NOT auto-delete the demo collection
    # at exit. Doing so would require opening Chroma in the parent
    # process, which on Windows can deadlock against the still-warm
    # SQLite handles held by the MCP subprocess that just exited.
    # Each run uses a fresh timestamped collection (see
    # ``_make_collection_name``) so the on-disk directory cost stays
    # bounded; users can prune via ``knowledge_list_collections`` /
    # ``knowledge_delete_collection`` from inside the actual app.
    sys.exit(asyncio.run(main()))
