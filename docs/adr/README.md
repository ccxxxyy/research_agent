# Architecture Decision Records

This directory holds the project's ADRs — the load-bearing
architectural choices, written down at the moment they were made,
with the alternatives we rejected and **why**. The goal is twofold:

1. **For future maintainers (including future-you):** answer the
   "why didn't we just do X?" question without having to
   re-litigate it from scratch.
2. **For interviews / code reviews:** demonstrate that the
   non-obvious decisions in the codebase were deliberate trade-offs,
   not accidents.

## Format

We use the [Michael Nygard ADR template](https://github.com/joelparkerhenderson/architecture-decision-record/tree/main/locales/en/templates/decision-record-template-by-michael-nygard)
in spirit, lightly adapted:

- **Status**: `Proposed` → `Accepted` → `Superseded by …` /
  `Deprecated`. Once written, an ADR is immutable; subsequent
  decisions get their own file.
- **Context**: the problem we were solving.
- **Decision**: what we picked.
- **Alternatives considered**: what we rejected, with reasons.
- **Consequences**: positive, negative, and neutral fallout.

## Index

| #    | Title                                                                    | Status   | Phase |
|------|--------------------------------------------------------------------------|----------|-------|
| 0001 | [Use FAISS (file-backed) instead of ChromaDB for the knowledge base](0001-faiss-over-chroma.md) | Accepted | 4.6   |
| 0002 | [Deliver `knowledge_expert` tools in-process, not over MCP stdio](0002-knowledge-server-inprocess.md) | Accepted | 4.6   |
| 0003 | [Add a Writer / Reasoner reflection loop as a post-supervisor subgraph](0003-reflection-loop.md) | Accepted | 5.2   |

## When to write a new ADR

Write one when **any** of these apply:

- We chose a non-default approach (e.g. picked library A over
  library B; ran something in-process instead of out-of-process).
- The decision touches more than one module / package.
- A reasonable reader would later ask "why is it this way?"
- The decision has consequences that are noticeable in operations
  (extra latency, extra deps, extra failure mode).

Do **not** write one for routine refactors, bug fixes, or
implementation details that don't have meaningful alternatives.
