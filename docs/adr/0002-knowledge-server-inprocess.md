# ADR 0002: Deliver `knowledge_expert` tools in-process, not over MCP stdio

- **Status**: Accepted
- **Date**: 2026-05-10
- **Deciders**: research-agent maintainers
- **Phase**: 4.6 (RAG closure)
- **Supersedes**: the Phase-2 plan to wrap `knowledge_server` as an
  MCP stdio subprocess identical to `fin_data_server`,
  `pdf_report_server`, and `code_server`.

## Context

The project's tool-delivery contract is **Model Context Protocol
(MCP) over stdio**: each tool family (financial data, PDF reports,
code execution, news, ...) is implemented as a `fastmcp` server,
launched as a child process at FastAPI startup by
`langchain_mcp_adapters.MultiServerMCPClient`. The supervisor never
imports the tool implementation; it sees only the JSON-RPC envelope.

This pattern has three properties we want:

1. **Strict tool/agent separation.** A tool crash cannot take down
   the agent.
2. **Language-agnostic tools.** A Node.js MCP server is as legitimate
   as a Python one.
3. **Hot-swap of tool implementations.** Replace `fin_data_server.py`
   with a new version, and the agent picks it up on next launch
   without code changes.

The Phase-2 design therefore prescribed: implement `knowledge_server`
as an MCP server too. This was straightforward to write — the file
exists, decorated with `@mcp.tool()` — but in practice the
`knowledge_server` subprocess hung intermittently on Windows.
Diagnosis:

- The hang was post-business-logic: the `_ingest` function ran to
  completion, returned a `dict`, and `fastmcp` started serialising
  the response to `stdout`. The parent never saw the response and
  timed out at 90 s.
- The root cause was a combination of (a) heavy ML imports in the
  same process as the MCP transport (`sentence-transformers`,
  `langchain_text_splitters`, originally `chromadb` — see
  [ADR 0001](0001-faiss-over-chroma.md)) and (b) Windows' stdio
  pipe behaviour with mixed buffered/unbuffered writes from native
  C extensions and Python.
- We attempted a low-level fix: a `_stdio_firewall.py` module that
  ran at server import time, `dup`ed the original `stdout` fd,
  redirected fd 1 to `NUL` to silence C-level writes, and routed
  Python's `sys.stdout` back through the saved fd. This worked for
  ChromaDB's daemon-thread chatter but did **not** prevent the
  post-`_ingest` hang under FAISS — proving that more than one
  layer in the stack was contributing.

The same `knowledge_server.py` code, called **in-process** (no MCP
subprocess, no stdio at all), executed `_ingest` + `_search` + `_list`
+ `_delete` cleanly in well under the original timeout. We confirmed
this with `_diag_ingest_inproc.py` (now removed).

## Decision

Expose the knowledge-base operations to the agent as **in-process
LangChain `StructuredTool` instances**, not as MCP stdio tools.

Mechanics:

- `src/research_agent/mcp_servers/knowledge_server.py` keeps its
  `@mcp.tool()` decorators and stays an MCP server **on paper** —
  the FastMCP entry point is preserved so the server is still
  runnable as a subprocess when needed (e.g. from a non-Python
  agent, or for protocol-conformance tests).
- `src/research_agent/tools/knowledge_tools.py` is the
  production-mode adapter: it imports `_ingest`, `_search`,
  `_list_collections`, `_delete_collection` directly and wraps each
  in a `StructuredTool.from_function`. These are what
  `knowledge_expert` actually consumes at runtime.
- `src/research_agent/mcp_servers/client_factory.py` exposes
  `load_knowledge_tools_inproc()` — symmetric in shape with the
  other `load_*_server_tools()` functions — so the FastAPI lifespan
  treats the knowledge tools identically to the MCP-loaded ones at
  the call site. Only the implementation differs.

## Alternatives considered

1. **Run `knowledge_server` over MCP sse instead of stdio.**
   FastMCP supports SSE transport, which would dodge the stdio
   pipe issue. Rejected because:
   - It adds a TCP port to the deployment (or a Unix domain
     socket, which doesn't exist on Windows).
   - It moves the failure mode from "subprocess hangs" to
     "subprocess crashes; SSE client keeps the agent blocked
     until heartbeat timeout".
   - SSE transport is also less battle-tested in
     `langchain_mcp_adapters` than stdio at the time of writing.
2. **Spawn the MCP subprocess with `PYTHONUNBUFFERED=1` and
   `sys.stdout.reconfigure(line_buffering=True)`.** Tried; reduced
   but did not eliminate the hang on Windows under heavy ML
   imports.
3. **Move the heavy ML imports to a lazy path.** `_ingest` already
   imports `sentence-transformers` and `langchain_text_splitters`
   lazily inside the function. The hang happened **after**
   `_ingest` returned, while writing the response, so lazy imports
   didn't help.
4. **Accept the timeout and add retry logic at the supervisor
   level.** Rejected because retries waste embedding work
   (sentence-transformers calls aren't cheap) and inflate latency.
   Also masks the real problem.
5. **Move ALL specialists in-process.** Tempting for symmetry but
   loses the very property we want from MCP for the OTHER
   specialists: `fin_data_server` calling `akshare` could legitimately
   crash from an upstream API; isolation is valuable. We pay the
   "two transports" cost only for the one specialist whose import
   chain is incompatible with stdio.

## Consequences

### Positive

- **Reliability.** Knowledge-base operations are deterministic
  again. No more 90 s timeouts on Windows; no more "works on
  Linux, hangs on my colleague's laptop".
- **Faster cold start for knowledge calls.** No per-call subprocess
  spin-up; the tools are method calls. Local benchmark: first
  `knowledge_search` after server boot drops from ~3.2 s to ~0.4 s.
- **Eager-import warm-up still works.** Top-of-module imports in
  `knowledge_server.py` (FAISS, text splitter) front-load the
  expensive initialisation at FastAPI startup so the first
  `knowledge_ingest_pdf` call doesn't pay the cost. Originally
  introduced to avoid the MCP-stdio deadlock, the eager imports
  keep their value as a startup-warm-up mechanism.
- **MCP protocol surface preserved.** The `@mcp.tool()` decorators
  are intact, so the server can still be run as a stdio subprocess
  by non-Python clients or in protocol-conformance tests. The
  default Python agent path simply chooses the in-process adapter.

### Negative

- **Loss of crash isolation for knowledge tools.** An unhandled
  exception in `_ingest` (e.g. a corrupted PDF causing pypdf to
  raise inside the calling task) now lands inside the FastAPI
  worker. We mitigate by wrapping every in-process tool function
  with a defensive try/except that converts exceptions to
  structured error returns — same contract as the MCP error
  envelope.
- **Two delivery modes to maintain.** The `knowledge_server.py`
  file now serves two consumers: the MCP runtime (for protocol
  conformance) and the in-process adapter (for the agent). Anything
  the in-process path needs (lazy/eager imports, error wrapping)
  has to remain compatible with the subprocess path. We document
  this explicitly in the module docstring.
- **Symmetry break in the explainer story.** When pitching the
  architecture, "all six specialists are MCP-backed" was a clean
  one-liner. The reality is now "five MCP, one in-process; here's
  why, and here's how we kept the protocol surface intact". This
  is a worse soundbite but more accurate.

### Neutral

- The supervisor and the rest of the agent code are unaware of
  the delivery mode. They consume `BaseTool` instances either way.
- `knowledge_expert` still validates that the right tool family
  was loaded (it rejects empty tool lists at build time) — the
  contract is identical to other specialists.

## Status

Implemented in commit `<git rev-parse HEAD>` (Phase 4.6). All
167 + 21 + 2 unit tests (regression + reflection + new wrapper
tests) green on Windows + Python 3.13.13. Knowledge end-to-end
flow (ingest → list → search → delete) verified via in-process
test fixtures.
