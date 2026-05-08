"""Knowledge-base RAG — tool definitions + (deprecated) MCP-stdio surface.

This is the **knowledge plane** of the financial-research agent.
Where ``pdf_report_server`` deals with PUBLIC disclosure PDFs (巨潮资讯)
and ``fin_data_server`` deals with PUBLIC market data (akshare), this
module lets a user upload **their own** PDF library (ESG reports,
broker research notes, internal memos, prospectuses) and ask
free-form questions over it.

Runtime model: in-process tools, NOT MCP subprocess
---------------------------------------------------
Historically this module was launched as an MCP-stdio subprocess so the
parent ``MultiServerMCPClient`` could discover the tools through the
same protocol as ``code_server`` / ``fin_data_server`` /
``pdf_report_server``. That path is **no longer the production path**:

* On Windows + Python 3.13, fastmcp's stdio JSON-RPC writer interacts
  badly with the heavy import chain pulled in by ``sentence-
  transformers`` / ``torch`` / ``faiss-cpu``. After ``ingest_pdf``
  finished its work the JSON-RPC response would never reach the parent
  (silent stdout-pipe stall). We diagnosed it down to "in-process
  works in 35-40s, MCP subprocess hangs forever even with FAISS,
  even with the stdout firewall disabled".
* The protocol value (cross-language / cross-process) is not actually
  exercised by this project — all four agents and the supervisor live
  in one Python process.

The four ``@mcp.tool`` decorated coroutines below are therefore kept
as the **canonical contract** for the knowledge-base capability (their
docstrings, validation rules and return shapes are authoritative), but
``research_agent.tools.knowledge_tools`` re-exports them as plain
``langchain_core.tools.tool``-decorated functions for in-process
consumption by ``knowledge_expert``. The ``@mcp.tool()`` decorator
registers the function with the FastMCP instance but otherwise leaves
it unchanged (we verified ``type(ingest_pdf) is types.FunctionType``),
so calling them directly is safe.

If a future workstream needs cross-process delivery (e.g. wiring this
into a Rust agent) the MCP-stdio launch path can be brought back; the
business logic doesn't change.

Why a *separate* server (vs. extending ``pdf_report_server``)?
-------------------------------------------------------------
``pdf_report_server`` is stateless: it derives a PDF URL from cninfo
search, downloads, parses bounded page ranges, and returns text. There
is no notion of an *index* — the LLM does the looking.

The knowledge server is stateful: PDFs are chunked, embedded, and
indexed once into a persistent FAISS collection; subsequent searches
hit a hybrid (vector + BM25) retriever that the LLM never sees the
raw PDF for. Different lifecycle, different persistence, different
storage-cost profile — they belong in different processes.

Tools exposed
-------------
1. ``knowledge_ingest_pdf`` — load → chunk → embed → write to FAISS.
   Returns ``(collection, num_chunks_added, total_chunks_in_collection)``.
2. ``knowledge_search`` — hybrid retrieval (RRF over vector + BM25),
   each hit annotated with ``source`` / ``page`` / ``vector_score``
   / ``bm25_rank`` / ``rrf_rank``. The response also surfaces
   **corrective-RAG signals** (``top_score``, ``mean_score``,
   ``unique_sources``, ``quality``) so the calling agent can decide
   whether to rewrite the query and retry.
3. ``knowledge_list_collections`` — enumerate collections currently in
   the persistent store with chunk counts.
4. ``knowledge_delete_collection`` — housekeeping; idempotent.

Why the corrective loop lives in the AGENT, not the tool
--------------------------------------------------------
Putting the rewrite/retry loop inside the tool would either:
  (a) require an LLM rewriter inside this subprocess, doubling the
      credentials / network surface area, or
  (b) hard-code rule-based rewrites, which is uninspiring.

Instead the tool returns rich quality signals and the
``knowledge_expert`` system prompt teaches the agent: "if
``top_score < 0.4`` or ``quality == 'low'``, REWRITE the query with
more specific keywords and call ``knowledge_search`` again, up to 3
attempts". This makes the corrective loop visible in the LangGraph
trace as repeat ``AIMessage → ToolMessage`` cycles inside the
knowledge_expert subgraph — the canonical Corrective-RAG story.

Storage layout
--------------
``./data/knowledge_db/<collection_name>/`` holds one persisted
FAISS index per collection. Each subfolder contains the standard
LangChain FAISS pair: ``index.faiss`` (binary index) plus
``index.pkl`` (docstore + ``faiss_id -> doc_id`` mapping). Listing
collections is therefore as simple as enumerating subdirectories
of the base path. BM25 is rebuilt in-memory from the persisted
FAISS docstore on first search after process start (or after an
ingestion that mutated the collection), amortising the cost across
the rest of the session.

Why FAISS, not Chroma?
~~~~~~~~~~~~~~~~~~~~~~
We tried Chroma first. On Windows, ``chromadb``'s import chain
spawns posthog-telemetry daemon threads that emit ``print()`` calls
which corrupted the MCP-stdio JSON-RPC channel. FAISS is pure C++ +
Python bindings, has no telemetry, no background threads, and a
simpler persistence model (one ``index.faiss`` + ``index.pkl`` per
collection directory) — a better engineering fit at this scale.
Even after the Chroma → FAISS migration the stdio path remained
unstable on Windows, which prompted the move to in-process delivery
documented above.

Embedding model
---------------
Local HuggingFace ``BAAI/bge-small-zh-v1.5`` — bilingual (Chinese +
English), small (~100MB), free, no API key. First call downloads the
weights (warm caches under ``~/.cache/huggingface``); subsequent
process spawns are instant.
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# Quiet-by-default environment for chatty ML libraries.
#
# Even though the production runtime is now in-process (``research_
# agent.tools.knowledge_tools``), we still set these env vars at module
# import time:
#
#   * ``transformers`` / ``tqdm`` / ``huggingface_hub`` would otherwise
#     spam progress bars and a "BertModel LOAD REPORT" banner into the
#     parent process's stdout, which is awkward when the parent is a
#     FastAPI worker (they end up in the request-log stream).
#   * ``TOKENIZERS_PARALLELISM=false`` silences the well-known fork-
#     warning when sentence-transformers boots inside an asyncio
#     worker thread.
#   * ``HF_HUB_DISABLE_TELEMETRY=1`` is just good hygiene.
#
# These ``setdefault`` calls never override an explicit value the
# operator has set in their shell.
# ---------------------------------------------------------------------
import os as _os

_os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
_os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
_os.environ.setdefault("TQDM_DISABLE", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import asyncio  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from fastmcp import FastMCP  # noqa: E402
from loguru import logger  # noqa: E402

# Eagerly import the heavy dependencies (langchain text-splitter +
# FAISS wrapper) at module load time rather than on first tool call.
# On the in-process path this just front-loads the ~9 s of langchain-
# core imports onto worker startup so the FIRST ``ingest_pdf`` request
# isn't unfairly slow. On the (legacy) MCP-stdio path it also dodged
# a Python-3.13 + anyio import-lock deadlock, but that path is no
# longer the production runtime — see the module docstring.
from langchain_community.vectorstores import FAISS as _PrewarmedFAISS  # noqa: E402, F401
from langchain_text_splitters import (  # noqa: E402, F401
    RecursiveCharacterTextSplitter as _PrewarmedSplitter,
)

mcp = FastMCP("KnowledgeBase")

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DEFAULT_DB_DIR = Path("./data/knowledge_db").resolve()
"""Persistent knowledge-base root directory.

One subdirectory per collection is created on first ingest. Each
subdirectory contains a single LangChain-FAISS index pair
(``index.faiss`` + ``index.pkl``). Module-level so tests can
monkey-patch and so subprocesses inherit the same path regardless
of the spawning process's CWD.
"""

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
"""Default embedding model. Local, free, bilingual."""

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120

MAX_INGEST_BYTES = 50 * 1024 * 1024  # 50 MB safety bound for a single PDF
MAX_TOP_K = 20
"""Hard cap on ``top_k`` per search call to prevent the LLM from
asking for ``top_k=10000`` on a 100k-chunk collection — that would
return enough text to blow the LLM's context window.
"""

# Quality classifier thresholds. Calibrated for normalised cosine
# similarity from BAAI/bge-small-zh-v1.5; adjust if you swap models.
QUALITY_HIGH_THRESHOLD = 0.65
QUALITY_MEDIUM_THRESHOLD = 0.40

RERANK_OVERFETCH_MULTIPLIER = 3
"""How many extra candidates to draw from RRF before reranking.

The cross-encoder is most useful when it has slack to re-order:
asking the bi-encoder + BM25 for ``top_k * 3`` candidates and
trimming after rerank typically promotes 1-2 items per query that
RRF alone would have buried. We cap at a small multiplier to keep
the per-call latency bounded — see :class:`CrossEncoderReranker`'s
own ``max_pairs`` guard for the second layer of protection.
"""


def _reranker_enabled() -> bool:
    """Read the ``KNOWLEDGE_RERANKER_ENABLED`` env var at call time.

    Read fresh on every call (not cached at import) so unit tests
    can monkeypatch ``os.environ`` to flip the switch between
    cases without re-importing the module.
    """
    raw = _os.environ.get("KNOWLEDGE_RERANKER_ENABLED", "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


# Lazy singleton: built on first ``_maybe_rerank`` call. Keeping the
# import deferred means a process that never asks for reranking
# (smoke tests, tooling) doesn't pay the ~1 s sentence_transformers
# import cost on module load.
_RERANKER: Any | None = None

# ---------------------------------------------------------------------
# Lazy module-level caches
#
# All three are kept ALIVE for the lifetime of the MCP subprocess.
# They are expensive to build (model load = ~3s; FAISS load + BM25
# rebuild = O(corpus size)) and cheap to reuse.
# ---------------------------------------------------------------------
_EMBEDDER: Any | None = None
"""HuggingFaceEmbeddings singleton. ``None`` until first use."""

_FAISS_STORES: dict[str, Any] = {}
"""``collection_name -> langchain_community.vectorstores.FAISS`` cache.

We hold one warmed FAISS in memory per collection so back-to-back
searches don't re-read the index file. The cache is dropped /
re-loaded after every successful ingest (FAISS is not designed for
concurrent in-place mutation).
"""

_BM25_CACHE: dict[str, "_BM25Index"] = {}
"""``collection_name -> in-memory BM25 index`` cache.

Invalidated by ``_invalidate_bm25(collection)`` after each ingest.
"""


def _fmt_error(exc: Exception, *, context: str) -> dict[str, Any]:
    """Canonical error shape — raising would kill the MCP subprocess."""
    return {"error": f"{type(exc).__name__}: {exc}", "context": context}


def _validate_collection_name(name: str) -> None:
    """Validate a collection name.

    Each collection becomes a directory name on disk so we restrict
    to ``[a-zA-Z0-9._-]`` (kept in line with the Chroma rules our
    earlier iteration used so collection names from existing test
    fixtures continue to work) and 3–63 chars. We additionally
    forbid leading dots and ``..`` to prevent path-traversal
    surprises.
    """
    if not (3 <= len(name) <= 63):
        raise ValueError(f"collection name length must be 3..63, got {len(name)}")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]", name):
        raise ValueError(
            f"collection name {name!r} must match "
            r"[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]"
        )
    if ".." in name:
        raise ValueError(f"collection name {name!r} must not contain '..'")


# ---------------------------------------------------------------------
# Embedder & FAISS helpers (lazy)
# ---------------------------------------------------------------------
def _get_embedder() -> Any:
    """Return the singleton embedder, building it on first call.

    Imports are deferred so the MCP subprocess starts fast even if
    sentence-transformers needs to download model weights — the
    ~3 s cost is paid on the first ``knowledge_ingest_pdf`` /
    ``knowledge_search``, not on subprocess launch.

    Stdout safety is provided by the module-level firewall (see
    ``_stdio_firewall``), so we do not need an inline
    ``redirect_stdout`` wrapper here even though ``transformers``
    historically prints a "BertModel LOAD REPORT" banner during
    weight loading.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Loading embedding model: {}", DEFAULT_EMBEDDING_MODEL)
        _EMBEDDER = HuggingFaceEmbeddings(
            model_name=DEFAULT_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBEDDER


def _collection_dir(collection: str, *, db_dir: Path | None = None) -> Path:
    """Return the persistence path for ``collection``.

    Resolving the path through this single helper means the rest of
    the module never has to spell out the convention
    (``DEFAULT_DB_DIR / collection``). Tests can monkey-patch
    ``DEFAULT_DB_DIR`` and this function picks it up automatically.
    """
    base = db_dir or DEFAULT_DB_DIR
    return base / collection


def _faiss_index_exists(collection: str, *, db_dir: Path | None = None) -> bool:
    """True iff a saved FAISS pair exists on disk for ``collection``.

    LangChain's ``FAISS.save_local(path)`` always writes BOTH
    ``index.faiss`` and ``index.pkl``; the absence of either is a
    half-written / corrupted state and we treat the collection as
    not present in that case (rather than half-load it and fail
    obscurely later).
    """
    cdir = _collection_dir(collection, db_dir=db_dir)
    return (cdir / "index.faiss").exists() and (cdir / "index.pkl").exists()


def _load_faiss_store(collection: str, *, db_dir: Path | None = None) -> Any | None:
    """Load (and cache) the FAISS store for ``collection``, or None.

    Returns ``None`` (instead of raising) when the collection has
    never been ingested. Callers must handle this — the search tool
    converts it into a ``quality='low'`` empty response, the ingest
    tool treats it as the "create" branch.

    ``allow_dangerous_deserialization=True`` is required because
    LangChain's FAISS sidecar is a ``pickle`` file. The repo only
    loads files we wrote ourselves under ``DEFAULT_DB_DIR``, never
    untrusted blobs from the internet, so the audit risk here is
    bounded to "an attacker who can already write into our data
    directory" — at which point pickle is the least of our worries.
    """
    cached = _FAISS_STORES.get(collection)
    if cached is not None:
        return cached
    if not _faiss_index_exists(collection, db_dir=db_dir):
        return None

    from langchain_community.vectorstores import FAISS

    cdir = _collection_dir(collection, db_dir=db_dir)
    store = FAISS.load_local(
        folder_path=str(cdir),
        embeddings=_get_embedder(),
        allow_dangerous_deserialization=True,
    )
    _FAISS_STORES[collection] = store
    return store


def _save_faiss_store(collection: str, store: Any, *, db_dir: Path | None = None) -> None:
    """Persist ``store`` and refresh the in-memory cache atomically.

    We always update ``_FAISS_STORES[collection]`` AFTER a successful
    ``save_local`` so a write failure (e.g. disk full) leaves the
    in-memory cache pointing at the previous on-disk state — never
    a phantom store that exists in memory but not on disk.
    """
    cdir = _collection_dir(collection, db_dir=db_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    store.save_local(folder_path=str(cdir))
    _FAISS_STORES[collection] = store


def _invalidate_bm25(collection: str) -> None:
    """Drop the BM25 cache for ``collection``; next search rebuilds it."""
    _BM25_CACHE.pop(collection, None)


# ---------------------------------------------------------------------
# BM25 index (rebuilt from FAISS docstore)
# ---------------------------------------------------------------------
class _BM25Index:
    """Wrapper around ``rank_bm25.BM25Okapi`` plus the originating docs.

    We keep ``docs`` parallel to the underlying ``BM25Okapi`` corpus so
    BM25 score lookups can be mapped back to ``(content, metadata)``
    without a second round-trip to the vector store.

    Tokenization is intentionally simple: lower-case + split on
    non-word characters. CJK characters survive as single-char tokens,
    which BM25 handles fine for queries that share noun phrases with
    the documents (the dominant case for finance-style RAG).
    """

    _SPLIT_RE = re.compile(r"\W+", flags=re.UNICODE)

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        from rank_bm25 import BM25Okapi

        self.docs = docs
        # Track whether we are running on a real corpus or just the
        # sentinel below; ``search`` uses this to short-circuit.
        self._is_empty: bool = not docs
        tokenized = [self._tokenize(d["content"]) for d in docs]
        # rank_bm25 cannot handle empty corpora — guard with a sentinel
        # whose only purpose is to satisfy ``BM25Okapi.__init__``. The
        # sentinel must NEVER surface as a real hit (it would inject
        # an empty-content "document" into the hybrid-fusion pipeline
        # downstream), so ``search`` checks ``_is_empty`` first.
        if not tokenized:
            tokenized = [[""]]
            self.docs = [{"content": "", "metadata": {}}]
        self._bm25 = BM25Okapi(tokenized)

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        return [t for t in cls._SPLIT_RE.split(text.lower()) if t]

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return ``[(corpus_index, bm25_score)]`` sorted desc.

        Returns an empty list when:
          - the query has no tokens after normalisation, or
          - the index was built from an empty corpus (no real docs).
        """
        if self._is_empty:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


def _build_bm25_for_collection(collection: str) -> _BM25Index:
    """Materialise BM25 from all docs currently in the FAISS docstore.

    LangChain's FAISS keeps the document objects in
    ``store.docstore._dict`` and a ``faiss_id -> doc_id`` mapping
    in ``store.index_to_docstore_id``. Walking the values of the
    docstore yields docs in deterministic insertion order, which
    is enough for BM25 (the BM25 indexing does not need to align
    with FAISS's internal numbering). For typical user libraries
    (10s–1000s of chunks) this is fast (<200 ms). For very large
    collections we would page; left as a TODO since the agent will
    rarely chew through >10k chunks of personal PDFs.

    Returns an empty index when the collection has not been
    ingested yet — callers must therefore tolerate ``search`` /
    ``BM25Index.search`` returning no hits (the search tool already
    does — it short-circuits to a ``quality='low'`` response when
    the FAISS store is missing).
    """
    store = _load_faiss_store(collection)
    docs: list[dict[str, Any]] = []
    if store is None:
        return _BM25Index(docs)
    docstore = store.docstore
    for doc_id in store.index_to_docstore_id.values():
        doc = docstore.search(doc_id)
        if doc is None or isinstance(doc, str):
            continue
        docs.append(
            {
                "content": getattr(doc, "page_content", "") or "",
                "metadata": dict(getattr(doc, "metadata", None) or {}),
            }
        )
    return _BM25Index(docs)


def _get_bm25(collection: str) -> _BM25Index:
    """Cached BM25 fetch; rebuilds lazily on cache miss / invalidation."""
    bm25 = _BM25_CACHE.get(collection)
    if bm25 is None:
        bm25 = _build_bm25_for_collection(collection)
        _BM25_CACHE[collection] = bm25
    return bm25


# ---------------------------------------------------------------------
# PDF loading + chunking
# ---------------------------------------------------------------------
def _load_pdf_pages(local_path: Path) -> list[dict[str, Any]]:
    """Return per-page records ``[{page, text}]`` extracted with pypdf.

    We deliberately keep the Document creation in pure dicts here
    rather than ``langchain_core.documents.Document`` so this helper
    is unit-testable without touching LangChain's API surface.
    """
    import pypdf

    with local_path.open("rb") as fh:
        reader = pypdf.PdfReader(fh)
        out: list[dict[str, Any]] = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            out.append({"page": i, "text": text})
    return out


def _chunk_pages(
    pages: list[dict[str, Any]],
    *,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    """Split each page's text into ``chunk_size``-character windows.

    Each chunk inherits ``page`` and ``source`` so the eventual
    answer can cite ``"source.pdf p.42"`` faithfully. We preserve the
    page boundary (chunks never span pages) — that loses a tiny bit
    of recall for cross-page sentences but makes the citation
    unambiguous, which is the more valuable property for finance RAG
    where users will sanity-check page numbers.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
        length_function=len,
    )
    chunks: list[dict[str, Any]] = []
    for page in pages:
        for piece in splitter.split_text(page["text"] or ""):
            chunks.append(
                {
                    "content": piece,
                    "metadata": {"source": source, "page": page["page"]},
                }
            )
    return chunks


# ---------------------------------------------------------------------
# Tool 1: ingest a PDF into a collection
# ---------------------------------------------------------------------
@mcp.tool()
async def ingest_pdf(
    local_path: str,
    collection: str = "default",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    """Ingest a single PDF into the persistent knowledge base.

    The PDF is read once with ``pypdf`` (pages preserved as the unit
    of provenance), split into ``chunk_size``-character windows with a
    ``chunk_overlap``-character slide, embedded with the bge-small
    Chinese model, and written to a FAISS collection (one folder
    per collection under ``DEFAULT_DB_DIR``). Re-ingesting the same
    PDF appends duplicate chunks — there is no content-hash dedup
    yet (TODO for a future iteration; current pattern is "one
    collection per ingestion pass").

    Args:
        local_path: Absolute or repo-relative path to a ``.pdf`` file.
            For convenience this is typically the path returned by
            ``pdf_download_pdf`` (Phase-4.2 server) so the two tools
            chain naturally.
        collection: Target collection name. Created on first use.
            Must match ``[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]`` and
            be 3..63 chars.
        chunk_size: Characters per chunk. 800 is a reasonable default
            for bge-small-zh's 512-token window (Chinese ≈ 1 char/
            token, English ≈ 0.25 token/char on average).
        chunk_overlap: Slide between adjacent chunks. 15% of
            ``chunk_size`` is the rule of thumb.

    Returns:
        On success: ``{collection, source, num_pages, num_chunks_added,
        total_chunks_in_collection}``.

        On failure: ``{error, context}``.
    """
    try:
        _validate_collection_name(collection)
    except ValueError as e:
        return _fmt_error(e, context=f"ingest_pdf(collection={collection!r})")

    path = Path(local_path)
    if not path.exists():
        return _fmt_error(
            FileNotFoundError(f"no such file: {path}"),
            context=f"ingest_pdf(local_path={local_path!r})",
        )
    if path.suffix.lower() != ".pdf":
        return _fmt_error(
            ValueError(f"only .pdf is supported, got {path.suffix!r}"),
            context=f"ingest_pdf(local_path={local_path!r})",
        )
    if path.stat().st_size > MAX_INGEST_BYTES:
        return _fmt_error(
            ValueError(
                f"PDF size {path.stat().st_size} exceeds limit {MAX_INGEST_BYTES}"
            ),
            context=f"ingest_pdf(local_path={local_path!r})",
        )

    if chunk_size < 100 or chunk_size > 4000:
        return _fmt_error(
            ValueError(f"chunk_size must be 100..4000, got {chunk_size}"),
            context="ingest_pdf()",
        )
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        return _fmt_error(
            ValueError(
                f"chunk_overlap must be 0..chunk_size-1, got {chunk_overlap}"
            ),
            context="ingest_pdf()",
        )

    def _ingest() -> dict[str, Any]:
        pages = _load_pdf_pages(path)
        chunks = _chunk_pages(
            pages,
            source=str(path),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not chunks:
            return {
                "collection": collection,
                "source": str(path),
                "num_pages": len(pages),
                "num_chunks_added": 0,
                "total_chunks_in_collection": _collection_count(collection),
                "warning": "PDF contained no extractable text",
            }
        from langchain_community.vectorstores import FAISS

        texts = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        existing = _load_faiss_store(collection)
        if existing is None:
            embedder = _get_embedder()
            store = FAISS.from_texts(
                texts=texts,
                embedding=embedder,
                metadatas=metadatas,
            )
        else:
            existing.add_texts(texts=texts, metadatas=metadatas)
            store = existing
        _save_faiss_store(collection, store)
        _invalidate_bm25(collection)
        return {
            "collection": collection,
            "source": str(path),
            "num_pages": len(pages),
            "num_chunks_added": len(chunks),
            "total_chunks_in_collection": _collection_count(collection),
        }

    try:
        return await asyncio.to_thread(_ingest)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(
            e,
            context=(
                f"ingest_pdf(local_path={local_path!r}, collection={collection!r})"
            ),
        )


def _collection_count(collection: str) -> int:
    """Return the chunk count for ``collection`` (best-effort, 0 on error).

    For FAISS we count entries in ``index_to_docstore_id`` rather
    than ``index.ntotal``: the two should be equal, but the former
    is the same number we used to build BM25, which keeps the two
    layers consistent under any future "soft delete" we might add.
    """
    try:
        store = _load_faiss_store(collection)
        if store is None:
            return 0
        return len(store.index_to_docstore_id)
    except Exception:  # noqa: BLE001
        return 0


# ---------------------------------------------------------------------
# Tool 2: hybrid search with corrective-RAG quality signals
# ---------------------------------------------------------------------
def _classify_quality(top_score: float, mean_score: float, unique_sources: int) -> str:
    """Heuristic three-bucket classifier for retrieval quality.

    The agent uses this label to decide whether to re-issue the search
    with a rewritten query. The thresholds are tuned for normalised
    cosine on bge-small-zh (we set ``normalize_embeddings=True`` in
    ``_get_embedder``); pasting raw cosine values from a different
    embedder will need re-calibration.

    "high"   → top hit is clearly on-topic; agent should answer.
    "medium" → mixed signal; agent should answer but warn user it may
               be partial.
    "low"    → top hit is weak; agent should rewrite and retry.
    """
    if top_score >= QUALITY_HIGH_THRESHOLD and unique_sources >= 1:
        return "high"
    if top_score >= QUALITY_MEDIUM_THRESHOLD and mean_score >= QUALITY_MEDIUM_THRESHOLD * 0.6:
        return "medium"
    return "low"


def _hybrid_fuse(
    vector_hits: list[tuple[dict[str, Any], float]],
    bm25_hits: list[tuple[int, float, dict[str, Any]]],
    *,
    k_rrf: int = 60,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list[dict[str, Any]]:
    """Fuse vector + BM25 result lists with weighted Reciprocal Rank Fusion.

    Returns one record per UNIQUE document (deduped by
    ``(source, page, content[:80])``) carrying:
        * ``content``, ``metadata``
        * ``vector_score``  — raw cosine similarity (∈ [0, 1] post-norm)
        * ``bm25_score``    — raw BM25 (unbounded, model-dependent)
        * ``rrf_score``     — fused rank score (the actual sort key)
        * ``vector_rank``, ``bm25_rank``  — 1-indexed rank in each list
                                            (None if not present)
    """
    fused: dict[str, dict[str, Any]] = {}

    def _key(meta: dict[str, Any], content: str) -> str:
        return f"{meta.get('source', '')}|p={meta.get('page', '?')}|{content[:80]}"

    for rank, (doc, score) in enumerate(vector_hits, start=1):
        k = _key(doc["metadata"], doc["content"])
        rec = fused.setdefault(
            k,
            {
                "content": doc["content"],
                "metadata": doc["metadata"],
                "vector_score": score,
                "bm25_score": 0.0,
                "rrf_score": 0.0,
                "vector_rank": rank,
                "bm25_rank": None,
            },
        )
        rec["vector_score"] = max(rec["vector_score"], score)
        rec["rrf_score"] += vector_weight / (k_rrf + rank)

    for rank, (_, score, doc) in enumerate(bm25_hits, start=1):
        k = _key(doc["metadata"], doc["content"])
        rec = fused.setdefault(
            k,
            {
                "content": doc["content"],
                "metadata": doc["metadata"],
                "vector_score": 0.0,
                "bm25_score": score,
                "rrf_score": 0.0,
                "vector_rank": None,
                "bm25_rank": rank,
            },
        )
        rec["bm25_score"] = max(rec["bm25_score"], score)
        rec["bm25_rank"] = rank if rec["bm25_rank"] is None else min(rec["bm25_rank"], rank)
        rec["rrf_score"] += bm25_weight / (k_rrf + rank)

    return sorted(fused.values(), key=lambda r: r["rrf_score"], reverse=True)


async def _maybe_rerank(
    query: str, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Optionally rerank ``candidates`` with a local cross-encoder.

    Returns the input list unchanged when:
      * the ``KNOWLEDGE_RERANKER_ENABLED`` env var is falsey, or
      * the reranker model fails to load (e.g. ``sentence_
        transformers`` not importable on this host), or
      * the underlying ``CrossEncoder.predict`` call raises.

    In every fallback path each candidate still ends up with a
    ``rerank_score`` key (set to ``None``) so the response shape is
    stable regardless of whether reranking actually ran. Callers
    must therefore tolerate ``rerank_score is None``.

    The function is intentionally tolerant: search is the agent's
    primary tool and breaking it because an optional reranker
    misbehaved would be the wrong trade-off.
    """
    if not candidates:
        return candidates
    if not _reranker_enabled():
        for c in candidates:
            c.setdefault("rerank_score", None)
        return candidates

    global _RERANKER
    try:
        if _RERANKER is None:
            from research_agent.rag.reranker import CrossEncoderReranker

            _RERANKER = CrossEncoderReranker()
            logger.info("Cross-encoder reranker initialised for knowledge_server")
        return await _RERANKER.rerank(query, candidates)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Reranker unavailable ({}); falling back to RRF order", exc
        )
        for c in candidates:
            c.setdefault("rerank_score", None)
        return candidates


@mcp.tool()
async def search(
    query: str,
    collection: str = "default",
    top_k: int = 5,
) -> dict:
    """Hybrid retrieval (vector + BM25 + optional cross-encoder rerank).

    Pipeline::

        FAISS (top_k * 3) ─┐
                           ├─ RRF fuse ─→ cross-encoder rerank ─→ trim
        BM25  (top_k * 3) ─┘    (over-fetch)        (optional)    (top_k)

    The reranker step uses a local ``BAAI/bge-reranker-base``
    cross-encoder; toggle via the ``KNOWLEDGE_RERANKER_ENABLED`` env
    var. When disabled or unavailable the pipeline degrades
    gracefully to RRF order — the response shape is identical, just
    with ``rerank_score: null`` on every hit.

    The response is shaped to make a CORRECTIVE-RAG agent's life easy:
    it carries per-hit scores AND a top-level ``quality`` label. The
    intended agent loop is::

        result = call("knowledge_search", query=Q, collection=C, top_k=5)
        if result["quality"] == "low":
            Q' = rewrite(Q)        # the agent does this
            result = call("knowledge_search", query=Q', ...)

    Args:
        query: Free-form natural-language question. Chinese or English
            both work — the bge-small-zh embedder is bilingual.
        collection: Collection to search. The tool returns an empty
            ``quality='low'`` response (NOT an error) when the
            collection does not exist or is empty, so a fresh
            agent can probe collections without exception handling.
        top_k: Maximum hits to return after fusion. Capped at
            ``MAX_TOP_K`` (20) to protect the LLM context window.

    Returns:
        ``{
            collection, query, top_k_returned,
            quality,                  # "high" / "medium" / "low"
            top_score, mean_score, unique_sources,
            results: [
                {content, source, page,
                 vector_score, bm25_score, rrf_score, rerank_score,
                 vector_rank, bm25_rank}
            ]
        }``

        ``rerank_score`` is a float when the cross-encoder ran and
        ``None`` when reranking was disabled / unavailable. The
        ``quality`` label is computed from ``vector_score`` (whose
        bands are calibrated for normalised cosine), not from the
        cross-encoder's logits, so the corrective-RAG agent loop
        keeps a stable signal across reranker on/off configurations.

        On failure: ``{error, context}``.
    """
    try:
        _validate_collection_name(collection)
    except ValueError as e:
        return _fmt_error(e, context=f"search(collection={collection!r})")

    if not query or not query.strip():
        return _fmt_error(
            ValueError("query must be non-empty"),
            context="search()",
        )
    if top_k < 1:
        return _fmt_error(
            ValueError(f"top_k must be >= 1, got {top_k}"),
            context="search()",
        )
    top_k = min(top_k, MAX_TOP_K)

    def _collect_candidates() -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
        """Sync phase: load FAISS, run vector + BM25, fuse via RRF.

        Returns ``(candidates, None)`` on the happy path — the
        candidate list is over-fetched (``top_k * RERANK_OVERFETCH_
        MULTIPLIER``) so the async reranker has slack to re-order.

        Returns ``(None, early_response)`` for cold-start cases
        (collection missing / empty) so the caller can short-circuit
        without paying the reranker startup cost.
        """
        store = _load_faiss_store(collection)
        if store is None or len(store.index_to_docstore_id) == 0:
            return None, {
                "collection": collection,
                "query": query,
                "top_k_returned": 0,
                "quality": "low",
                "top_score": 0.0,
                "mean_score": 0.0,
                "unique_sources": 0,
                "results": [],
                "warning": (
                    f"collection {collection!r} is empty; ingest a PDF first "
                    f"with knowledge_ingest_pdf"
                ),
            }

        # Over-fetch from each retriever so the cross-encoder has
        # room to promote items the bi-encoder ranked lower.
        retrieve_k = max(top_k * RERANK_OVERFETCH_MULTIPLIER, 10)
        try:
            raw_vec = store.similarity_search_with_score(query, k=retrieve_k)
        except Exception:  # noqa: BLE001
            raw_vec = []
        # LangChain's FAISS returns (Document, L2 distance) by
        # default. Because we set ``normalize_embeddings=True`` in
        # ``_get_embedder`` the embeddings are unit vectors, so the
        # squared L2 distance ``d`` and the cosine similarity ``s``
        # are related by ``d = 2 - 2s`` ⇒ ``s = 1 - d / 2``. Clamp
        # to [0, 1] to absorb tiny FP drift.
        vector_hits: list[tuple[dict[str, Any], float]] = []
        for doc, distance in raw_vec:
            similarity = max(0.0, min(1.0, 1.0 - float(distance) / 2.0))
            vector_hits.append(
                (
                    {"content": doc.page_content, "metadata": doc.metadata or {}},
                    similarity,
                )
            )

        bm25 = _get_bm25(collection)
        bm25_raw = bm25.search(query, top_k=retrieve_k)
        bm25_hits = [(idx, score, bm25.docs[idx]) for idx, score in bm25_raw]

        # Hand the full fused list to the reranker — trimming to
        # ``top_k`` happens AFTER reranking, otherwise we'd lose the
        # very candidates the cross-encoder is meant to promote.
        candidates = _hybrid_fuse(vector_hits, bm25_hits)[:retrieve_k]
        return candidates, None

    try:
        candidates, early = await asyncio.to_thread(_collect_candidates)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(
            e,
            context=(
                f"search(query={query!r}, collection={collection!r}, top_k={top_k})"
            ),
        )

    if early is not None:
        return early

    assert candidates is not None  # narrows for type checkers

    # Reranking is best-effort: ``_maybe_rerank`` always returns a
    # list of the same shape (with ``rerank_score`` populated or
    # ``None``), even when the env-var is off or the model fails to
    # load — the search response shape stays stable.
    reranked = await _maybe_rerank(query, candidates)
    final_records = reranked[:top_k]

    results: list[dict[str, Any]] = []
    for rec in final_records:
        meta = rec["metadata"] or {}
        results.append(
            {
                "content": rec["content"],
                "source": meta.get("source", ""),
                "page": meta.get("page"),
                "vector_score": round(rec["vector_score"], 4),
                "bm25_score": round(rec["bm25_score"], 4),
                "rrf_score": round(rec["rrf_score"], 6),
                "rerank_score": rec.get("rerank_score"),
                "vector_rank": rec["vector_rank"],
                "bm25_rank": rec["bm25_rank"],
            }
        )

    # Quality classification stays on ``vector_score`` — those
    # thresholds were calibrated against the bi-encoder's normalised
    # cosine. The cross-encoder logits live on a different scale
    # and would silently invalidate the high/medium/low bands.
    scores = [r["vector_score"] for r in results]
    top_score = max(scores) if scores else 0.0
    mean_score = sum(scores) / len(scores) if scores else 0.0
    unique_sources = len({r["source"] for r in results if r["source"]})
    quality = _classify_quality(top_score, mean_score, unique_sources)

    return {
        "collection": collection,
        "query": query,
        "top_k_returned": len(results),
        "quality": quality,
        "top_score": round(top_score, 4),
        "mean_score": round(mean_score, 4),
        "unique_sources": unique_sources,
        "results": results,
    }


# ---------------------------------------------------------------------
# Tool 3: list collections (with chunk counts)
# ---------------------------------------------------------------------
@mcp.tool()
async def list_collections() -> dict:
    """List all collections currently in the persistent knowledge base.

    Useful as the agent's first call when the user's query implies a
    library but doesn't name a collection (e.g. "what's in my ESG
    library?"). The agent can then route subsequent ``search`` calls
    at the right collection.

    A "collection" here is any subdirectory of ``DEFAULT_DB_DIR``
    that contains both ``index.faiss`` and ``index.pkl``. Stray
    directories (e.g. half-deleted leftovers) are silently skipped —
    we never want to surface a broken collection to the LLM.

    Returns:
        ``{db_dir, collections: [{name, chunk_count}]}``. Each entry
        has ``name`` (str) and ``chunk_count`` (int, best-effort —
        ``-1`` if the FAISS pair is unreadable).
    """

    def _list() -> dict[str, Any]:
        DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
        out: list[dict[str, Any]] = []
        for child in sorted(DEFAULT_DB_DIR.iterdir()):
            if not child.is_dir():
                continue
            if not _faiss_index_exists(child.name):
                continue
            try:
                chunk_count = _collection_count(child.name)
            except Exception:  # noqa: BLE001
                chunk_count = -1
            out.append({"name": child.name, "chunk_count": chunk_count})
        return {"db_dir": str(DEFAULT_DB_DIR), "collections": out}

    try:
        return await asyncio.to_thread(_list)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context="list_collections()")


# ---------------------------------------------------------------------
# Tool 4: delete a collection
# ---------------------------------------------------------------------
@mcp.tool()
async def delete_collection(collection: str) -> dict:
    """Delete a collection and its in-memory caches. Idempotent.

    Used for housekeeping when a user wants to re-ingest a corpus from
    scratch (e.g. after changing chunk_size). Missing collections do
    NOT raise — the response just reports ``existed=False``.

    Returns:
        ``{collection, existed, deleted}``.
    """
    try:
        _validate_collection_name(collection)
    except ValueError as e:
        return _fmt_error(e, context=f"delete_collection({collection!r})")

    def _delete() -> dict[str, Any]:
        import shutil

        cdir = _collection_dir(collection)
        existed = _faiss_index_exists(collection)
        if not existed:
            # Nothing on disk; still drop any stale in-memory caches
            # so the next ingest under this name starts truly fresh.
            _FAISS_STORES.pop(collection, None)
            _BM25_CACHE.pop(collection, None)
            return {"collection": collection, "existed": False, "deleted": False}
        if cdir.exists():
            shutil.rmtree(cdir)
        _FAISS_STORES.pop(collection, None)
        _BM25_CACHE.pop(collection, None)
        return {"collection": collection, "existed": True, "deleted": True}

    try:
        return await asyncio.to_thread(_delete)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context=f"delete_collection({collection!r})")


# ---------------------------------------------------------------------
# Module export — math import is here only so ruff doesn't warn about
# unused imports if a future iteration uses log/exp normalisation. It
# costs nothing.
# ---------------------------------------------------------------------
_ = math


if __name__ == "__main__":
    mcp.run(transport="stdio")
