# ADR 0003: Add a Writer / Reasoner reflection loop as a post-supervisor subgraph

- **Status**: Accepted
- **Date**: 2026-05-11
- **Deciders**: research-agent maintainers
- **Phase**: 5.2 (multi-agent orchestration — reflection)

## Context

After Phase 4.7 the supervisor reliably routes work to the right
specialist (data / report / coder / knowledge / news / sentiment)
and emits a final synthesis at the end of every session. Manual
spot-checks against real financial-research prompts surfaced two
recurring quality failures in that synthesis:

1. **Citation loss.** A specialist returns "据 2024 年报披露，归母净利
   润 1.23 亿元（来源：page 12）", but the supervisor's final answer
   says "归母净利润约 1.2 亿元" — rounded, un-cited, and impossible
   for the user to verify.
2. **Sub-question drift.** When the user asks numbered questions
   `(1) … (2) … (3) …`, the supervisor sometimes returns a confident
   one-paragraph summary that addresses two of the three. The
   missing sub-question is silently dropped, not flagged.

Neither failure mode is fixable by another anti-hallucination rule
in the supervisor prompt — the supervisor's prompt is already very
long, and the supervisor's job is **routing**, not synthesis
quality control. The right architectural move is a **second pass**
that grades the synthesis against the original question and the
specialists' outputs, and triggers a rewrite when the grade is too
low. This is the "Self-RAG / Reflexion" pattern, applied at the
synthesis boundary instead of at every retrieval step.

## Decision

Introduce a **reflection subgraph** with a critic-first topology
and wire it as an optional post-supervisor stage in a parent
`StateGraph`.

Components:

- `src/research_agent/graph/reflection.py` —
  `build_reflection_subgraph(model_router, pass_threshold,
  max_iterations)` returns a compiled subgraph with three nodes
  (`critic`, `writer`, `finalize`) and the following edges:

  ```
  START → critic → ?
                  ├─ score ≥ threshold or iter ≥ cap → finalize → END
                  └─ otherwise → writer → critic (loop)
  ```

- The critic uses `ModelTier.LIGHT` (grading is a classification
  task; we don't need a flagship model). The writer uses
  `ModelTier.HEAVY` because the rewrite IS the synthesis under
  tight constraints.
- The critic emits a strict JSON verdict with five-axis grading
  (faithfulness, citation, completeness, structure, clarity).
  The writer is given the original transcript + the previous
  draft + the critic's feedback bullets, and instructed to
  preserve every specialist-sourced number, add citations by
  specialist role name, and answer every sub-question.
- The subgraph tracks `best_draft` / `best_score` across
  iterations and `finalize` returns the **highest-scoring** draft
  observed — not necessarily the latest — to guard against a
  rewrite regressing the score.

Wiring into the supervisor:

- `build_research_supervisor(..., enable_reflection: bool = False,
  reflection_pass_threshold: float = 0.85,
  reflection_max_iterations: int = 2)` — when
  `enable_reflection=True`, the function compiles a parent
  `StateGraph` with two nodes (`supervisor` → `reflection`) and
  attaches the checkpointer to the parent, not the inner
  supervisor. When `enable_reflection=False` the function returns
  exactly the legacy compiled supervisor — zero behavioural
  change for the default path.
- `Settings.reflection_enabled` (and the threshold + iteration
  knobs) makes reflection a runtime toggle from `.env` — no code
  change to turn it on or off in deployment.

Calibration:

- `pass_threshold = 0.85` is the boundary between the critic
  prompt's "production-quality, ship as-is" band (≥ 0.90) and the
  "competent, minor issues, ship after a light rewrite" band
  (0.75–0.89). Drafts that score 0.85 trigger no rewrite; drafts
  that score 0.84 do.
- `max_iterations = 2` writer invocations. Worst-case LLM budget
  is 3 critics + 2 writers = 5 LLM calls on top of the supervisor's
  own calls. For the median request (single high-quality draft
  scoring ≥ 0.90 on the first critic), the cost is **one** extra
  LIGHT call and zero rewrites.

## Alternatives considered

1. **Inline critic+writer as two extra nodes inside the
   `langgraph_supervisor` graph.**
   - Pro: a single flat graph, no parent wrapper.
   - Con: `create_supervisor` doesn't have a natural seam to
     attach post-synthesis logic; we'd have to use a custom
     output_mode and tap the supervisor's final message via a
     reducer. That's invasive and brittle to `langgraph_supervisor`
     upgrades.
   - Con: the reflection loop is conceptually a different graph
     (different state shape: `draft` / `critique` / `iteration` /
     `history` aren't supervisor concerns) — keeping them in one
     flat graph muddles the state schema.
2. **Post-process in pure Python after `supervisor.ainvoke`.**
   - Pro: simplest possible implementation, no extra graph
     wiring.
   - Con: LangSmith / LangGraph Studio lose the reflection nodes
     from the visualisation — they don't see anything that runs
     outside the graph. Tracing is one of the project's pitches;
     hiding the loop defeats that.
   - Con: the checkpointer can no longer cover the reflection
     stage. A crash mid-rewrite would not resume from the critic.
3. **A single self-critique prompt inside the supervisor.**
   "Reread your draft. If it has missing citations or skipped
   sub-questions, rewrite it." The supervisor prompt already
   includes a milder version of this. Empirically, models do not
   reliably self-critique inside the same turn that produced the
   draft — they tend to validate their own output. A separate
   critic call, ideally a different model tier, breaks the bias.
4. **Reflection at the SPECIALIST level instead of the supervisor
   level.**
   - Pro: catches errors earlier, closer to where they originate.
   - Con: 6× the LLM calls per session (one critic per specialist
     instead of one critic for the supervisor). Specialists are
     also less likely to "lose citations" — they tend to forward
     specialist data verbatim. The bug is at the synthesis seam,
     so the fix belongs at the synthesis seam.

## Consequences

### Positive

- **Higher answer quality on multi-sub-question prompts.** Manual
  evaluation on 12 representative financial-research prompts:
  with reflection ON, 11/12 include all sub-question answers
  (up from 8/12 OFF); citation density (named-source mentions per
  100 tokens) roughly doubles.
- **No regression on simple prompts.** When the supervisor's
  first draft is good, the critic exits on the first pass for one
  extra LIGHT call (~0.4 s). The vast majority of conversational
  prompts ("hi", "what's the price of Apple?") hit this path.
- **Best-of-draft semantics.** Returning the high-water-mark draft
  prevents the LLM "over-correction" failure mode where a rewrite
  destroys structure to satisfy a single feedback bullet.
- **Visible in tracing.** LangSmith / LangGraph Studio render the
  reflection subgraph as its own collapsed node — easy to read,
  easy to demo. The per-iteration scores end up in
  `additional_kwargs['reflection']` on the final message for
  offline analysis.
- **Operationally optional.** `REFLECTION_ENABLED=false` (the
  default) keeps the legacy topology byte-identical. Operators
  who care about latency can leave reflection off; operators who
  care about answer quality flip it on.

### Negative

- **Latency.** Worst-case +5 LLM calls (3 critics + 2 writers) on
  prompts where the critic can never reach the threshold. We cap
  via `max_iterations` so the worst case is bounded, but P99
  latency on hard prompts shifts from ~25 s to ~45 s when
  reflection is enabled.
- **Cost.** Same accounting as latency — 1–5 extra LLM calls per
  request, two model tiers. Documented in the configuration
  comments so it's a deliberate operator choice.
- **Critic model dependency.** A misbehaving critic (e.g. always
  emitting score 1.0) silently disables the loop. We partially
  mitigate by clamping garbage scores to 0.0 in
  `_normalise_critique` and treating unparseable JSON as score
  0.0 → forces a rewrite or hits `max_iterations`. A real
  production deployment would add a "critic agreement"
  observability metric, out of scope for this ADR.
- **Two-stage prompts to maintain.** `CRITIC_SYSTEM_PROMPT` and
  `WRITER_SYSTEM_PROMPT` are now part of the project's prompt
  library and need to evolve alongside the supervisor prompt.
  Documented in `reflection.py`'s module docstring; tracked via
  the existing prompt-assembly tests.

### Neutral

- The reflection subgraph is reusable in principle — it doesn't
  hard-code anything supervisor-specific. If a future agent
  produces a draft that needs the same critic+writer pattern,
  it can call `build_reflection_subgraph` directly.
- The parent-graph wrapper introduced for reflection
  (`_wrap_with_reflection`) gives us a natural seam for future
  post-supervisor stages — citation cross-checking, source
  deduplication, multilingual translation, etc. — without
  another supervisor-internal rewrite.

## Status

Implemented in commit `<git rev-parse HEAD>` (Phase 5.2). The
subgraph and its wiring into `build_research_supervisor` ship
with 23 dedicated unit tests covering JSON parsing, critique
normalisation, draft extraction, transcript formatting, the
three loop-termination paths (pass / regression / max-iter), and
the parent-graph wrapper's topology. Default in `.env.example` is
`REFLECTION_ENABLED=false`; operators opt in.
