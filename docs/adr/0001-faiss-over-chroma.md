# ADR 0001: Use FAISS (file-backed) instead of ChromaDB for the knowledge base

- **Status**: Accepted
- **Date**: 2026-05-10
- **Deciders**: research-agent maintainers
- **Phase**: 4.6 (RAG closure)

## Context

`research_agent` ships a per-user PDF knowledge base behind the
`knowledge_expert` specialist. The user uploads PDFs, the server
parses → chunks → embeds → stores them, and at search time the agent
does a hybrid (vector + BM25) lookup with reranking. The vector store
sits on the hot path of every research session, so its operational
properties dominate the developer experience.

The original Phase-0 plan picked **ChromaDB** for this role because:

- Persistent on-disk store with no separate process required for the
  "single user, single laptop" deployment we wanted first.
- HNSW-based ANN out of the box.
- Familiar API surface (`langchain-chroma`) that maps cleanly to the
  rest of the LangChain ecosystem.

Three things happened during Phase-4 that forced us to revisit this
choice:

1. **Stdio pipe corruption on Windows.** The `knowledge_server`
   MCP subprocess runs over JSON-RPC over `stdin`/`stdout`. Importing
   `chromadb` pulls in a chain that includes `posthog` (telemetry),
   `protobuf`, and `onnxruntime`. On Python 3.13 / Windows we
   observed the subprocess's `stdout` pipe becoming partially buffered
   in a way that delayed the JSON-RPC response — `_ingest` would
   complete, write the response to `sys.stdout`, and the parent would
   then time out at 90s having received nothing. We confirmed this
   was a stdio-layer issue (not a `_ingest` runtime issue) with an
   in-process diagnostic that called the exact same function and
   completed in seconds. A bespoke `_stdio_firewall.py` that
   redirected fd 1 to `NUL` and rerouted `sys.stdout` to a saved fd
   reduced but did not fully eliminate the symptom.
2. **Disabling telemetry was insufficient.** Setting
   `ANONYMIZED_TELEMETRY=False` and `CHROMA_TELEMETRY=False` removes
   the telemetry **daemon thread**, but `chromadb` itself spawns
   background workers (segment manager, embedding queue) whose
   logging — even silenced — went through Python's import-time
   `logging` configuration in a way that still affected the stdio
   pipe under fast `_ingest` workloads.
3. **Build/install footprint.** `chromadb` pulls in a transitive
   dependency set that's ~120 MB unzipped (`onnxruntime`,
   `chroma-hnswlib`, `tokenizers`, `tqdm`, `mmh3`, `bcrypt`,
   `pulsar-client`, ...). For a project that already ships
   `sentence-transformers`, `langchain-*`, and an MCP runtime, the
   additional weight made Docker image rebuilds noticeably slower
   and the `uv sync` cold path > 5 minutes on a fresh laptop.

## Decision

Replace `chromadb` + `langchain-chroma` with **FAISS** (via
`faiss-cpu` and `langchain_community.vectorstores.FAISS`) as the
on-disk vector store for the knowledge base.

Concretely:

- Each collection is one directory on disk
  (`./data/knowledge_db/<collection_name>/`) containing
  `index.faiss` + `index.pkl`. Loading is `FAISS.load_local(...)`,
  saving is `vs.save_local(...)`. No daemon, no telemetry, no
  background threads, no port to bind.
- Existing dual-retrieval logic (vector + BM25 + reranker) stays
  unchanged — only the vector backend swaps.
- Collection deletion is `shutil.rmtree` on the directory; collection
  listing is `os.listdir` on the root, gated by FAISS index file
  presence. Both operations are O(1) syscalls instead of round
  trips to a ChromaDB client.

## Alternatives considered

1. **Stay on ChromaDB, fix the Windows stdio issue properly.**
   We would need to either upstream a patch to chromadb's logging /
   init order or fork it to remove the segment manager's noisy
   side-effects. Both are out-of-band investments with no
   product value; the stdio issue is a Windows-specific developer
   experience problem that doesn't even exist on Linux/macOS, but
   the project's MVP must run on Windows.
2. **Run ChromaDB as a separate HTTP server.** The official
   `chromadb run` mode would sidestep the stdio issue entirely
   (no shared `stdout` between the agent and the vector store).
   Rejected because it pulls us back to multi-process deployment
   for a single-user dev environment, complicates `docker-compose`
   minimally but adds a port to the README and a permission story
   that the project doesn't want to own.
3. **Pinecone / Weaviate / Qdrant cloud.** Production-grade but
   requires API keys, a network, and per-call latency. Wrong choice
   for a project optimised for "clone and run".
4. **pgvector inside the existing Postgres container.** Attractive
   because the project already runs Postgres for checkpointing, but
   forces every user to bring up Postgres just to run the knowledge
   base — which contradicts the "knowledge_expert works standalone"
   property we wanted for the Phase-4.6 smoke test. We reserve
   pgvector for a future Phase-6 "production-grade" mode.

## Consequences

### Positive

- **Zero install friction.** `pip install faiss-cpu` is a single
  ~15 MB wheel; no `onnxruntime`, no telemetry, no daemon. Cold
  `uv sync` time dropped from > 5 min to ~90 s on Windows.
- **Stable stdio behaviour.** The `knowledge_server` import chain
  no longer touches background threads or fd-level logging. The
  remaining hang ([ADR 0002](0002-knowledge-server-inprocess.md))
  was unrelated to ChromaDB and survived the FAISS migration —
  which itself proved that ChromaDB was at least *one* of the
  causes.
- **Simpler operational model.** Backup = `tar` the collection
  directory. Migration = copy directory. Delete = `rm -rf`. There
  is no "vacuum" / "compact" / "consistency check" maintenance task.

### Negative

- **No incremental write durability.** FAISS persists by
  serialising the whole index back to disk. We do this synchronously
  after every successful `_ingest` call, which is fine at hundreds
  of MB of vectors but would not scale to streaming ingestion of
  millions of chunks. The current product target is "one analyst's
  PDF library, ≤ a few hundred documents" so this is acceptable; a
  future ADR will replace FAISS with pgvector or LanceDB if we
  cross the 10⁵-chunk threshold.
- **No multi-process write safety.** Two processes writing to the
  same collection directory will race on `save_local`. We mitigate
  by funnelling all knowledge-base writes through a single
  `knowledge_expert` agent inside the FastAPI process (since
  [ADR 0002](0002-knowledge-server-inprocess.md), this agent runs
  in-process anyway). A second writer would require a file lock —
  out of scope until multi-tenant.
- **HNSW tunables are less ergonomic.** ChromaDB exposes index
  parameters as a single config dict; FAISS exposes them at index
  construction time. We default to FAISS's `IndexFlatL2` for ≤ 10⁴
  chunks because exact search is fast at that size and the latency
  is dominated by embedding, not ANN. When (if) we cross to
  `IndexHNSWFlat` we'll document the trade-off in an updated ADR.

### Neutral

- Embedding model (`BAAI/bge-small-zh-v1.5`) and reranker
  (`BAAI/bge-reranker-base`) are unaffected — they were never tied
  to the vector store choice.
- BM25 sidecar (built from the same docstore the vector store
  holds) survives unchanged; the only edit was to load documents
  from FAISS's docstore instead of ChromaDB's collection.

## Status

Implemented in commit `<git rev-parse HEAD>` (Phase 4.6). Linux and
Windows smoke tests pass; ingest + search + delete + list cycle is
green in `tests/unit/test_mcp_echo_server.py`-style harnesses (the
knowledge tests run in-process, see ADR 0002).
