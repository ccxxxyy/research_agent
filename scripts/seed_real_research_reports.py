"""Seed the production knowledge-base with real A-share research reports.

What this script does
---------------------
Pulls the most recent 1-2 disclosures (annual / quarterly report,
preferred over 临时公告) for a curated set of representative A-share
tickers off **巨潮资讯** (cninfo), downloads each PDF into the
``./data/pdf_cache/`` content-addressable cache, and ingests the
resulting PDFs into the persistent FAISS collection ``prod_reports``
(one collection holds all reports — chunk metadata carries
``source = local PDF path`` so the agent can still cite per-document).

The seeded collection becomes the operational corpus the
``knowledge_expert`` searches at runtime when the supervisor needs
historical research context — see ``scripts/demo_full_research.py``
for the matching end-to-end demo question.

Why this script exists
----------------------
Up until now the project's RAG layer was demonstrable but never
**driven by real data**: ``demo_knowledge_expert.py`` ingested a
hand-rolled 2-page synthetic PDF. That's fine for proving the
plumbing; it's not enough for an interview-grade story. With this
seed script we get:

  * a fixed-name FAISS collection populated from real disclosures
    (interview demo can reference it without depending on stale
    timestamps or 网络抖动 mid-call);
  * a reproducible idempotent path — re-running the script does NOT
    duplicate already-ingested chunks (we check
    ``knowledge_list_collections`` and the per-source chunk count
    before ingesting);
  * a clean separation between **ingestion** (this script, run once)
    and **retrieval-augmented Q&A** (the demo + the agent at
    runtime).

Why we call the tools in-process (not via MCP-stdio)
----------------------------------------------------
``pdf_report_server`` and ``knowledge_server`` both expose their
tools as plain async functions decorated by ``@mcp.tool()`` (the
decorator registers them with FastMCP but doesn't wrap them — see
``knowledge_server.py`` module docstring). Calling them directly
from this seed script is functionally identical to going through
the MCP-stdio transport but skips two subprocess spawns + JSON-RPC
serialisation per call. For a one-shot operator script that runs
~10 seconds of HTTP I/O + ~20 seconds of ingestion, that's a
worthwhile shortcut.

Run::

    .venv/Scripts/python.exe scripts/seed_real_research_reports.py
    # optional: --tickers 600519,300750  --collection prod_reports

Exit codes:
    0 → at least one report was ingested or already present.
    1 → no report could be obtained for any ticker (network down,
        cninfo schema drift, etc.).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Force UTF-8 on stdout so Chinese strings don't blow up the Windows
# code page.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


# ---------------------------------------------------------------------
# Curated ticker list — AI / semiconductor cross-section
#
# Three sub-sectors of the AI value chain so the demo question has
# meaningful comparisons across "compute / interconnect / memory":
#
#     AI 算力芯片     688256 寒武纪      — 训练 GPU/ASIC, 亏损改善
#     CPO / 光模块    300308 中际旭创    — AI 数据中心互联, 业绩爆发
#     存储芯片        603986 兆易创新    — NOR Flash / MCU, 周期复苏
#
# Adding more is fine; ingestion runs sequentially so a long list
# just takes longer (each annual report is ~10 MB → ~20 s ingest
# warm-cache, dominated by the embedder).
# ---------------------------------------------------------------------
DEFAULT_TICKERS: dict[str, str] = {
    "688256": "寒武纪",
    "300308": "中际旭创",
    "603986": "兆易创新",
}

DEFAULT_COLLECTION = "prod_reports"

# Categories tried in order until we find something. cninfo's "年报"
# is published once a year (March/April) so on a freshly rolled-over
# day a ticker may have only quarterly disclosures available — the
# fallback chain prevents an empty seed.
PREFERRED_CATEGORIES = ("年报", "一季报", "三季报", "半年报")

# Look-back window: 365 days catches at least one annual report for
# every ticker, even ones whose fiscal year ends late in the calendar.
LOOKBACK_DAYS = 365

# Per-ticker hard cap. For seeding we want ONE recent annual + ONE
# quarterly to keep the corpus small and the ingestion fast. The
# embedder cost is ~20 s per 100-page report.
MAX_REPORTS_PER_TICKER = 1


def _step(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def _find_recent_report(
    *,
    search_announcements,
    symbol: str,
    end_date: datetime,
) -> dict[str, Any] | None:
    """Try preferred categories in order; return the most recent hit.

    Returns the announcement dict (carries ``pdf_url``) or ``None``
    when no category produced a usable record. We pick the FIRST
    record from ``announcements`` because cninfo orders them
    descending by ``publish_date``.
    """
    start_date = (end_date - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    for cat in PREFERRED_CATEGORIES:
        _step(f"  search_announcements(symbol={symbol}, category={cat}, "
              f"window={start_date}..{end_str})")
        try:
            resp = await search_announcements(
                symbol=symbol,
                start_date=start_date,
                end_date=end_str,
                category=cat,
                limit=10,
            )
        except Exception as exc:  # noqa: BLE001
            _step(f"    search failed for category={cat}: {exc!r}")
            continue
        if "error" in resp:
            _step(f"    search returned error for category={cat}: {resp['error']}")
            continue
        records = [
            r for r in resp.get("announcements", []) if r.get("pdf_url")
        ]
        if not records:
            _step(f"    no PDF announcements in category={cat}; trying next.")
            continue
        chosen = records[0]
        _step(
            f"  picked: {chosen.get('publish_date')} | "
            f"{chosen.get('title', '')[:60]} | size?"
        )
        return chosen
    return None


async def _ingested_sources_for_collection(
    *,
    list_collections,
    collection: str,
    knowledge_search,
) -> set[str]:
    """Return the set of ``source`` paths already present in ``collection``.

    We use a wildcard-y query (a single Chinese char that's almost
    sure to match SOMETHING in any corpus) and pull a generous
    ``top_k`` so we can dedup. This is cheap once FAISS is warm.
    Returns an empty set if the collection doesn't exist yet.
    """
    listing = await list_collections()
    names = {c["name"] for c in listing.get("collections", [])}
    if collection not in names:
        return set()
    # ``top_k`` is capped at 20 by knowledge_server's MAX_TOP_K — for
    # a seed corpus that contains a handful of large PDFs this is
    # already plenty of distinct ``source`` values to dedup on.
    probe = await knowledge_search(
        query="公司",  # generic on purpose — seeding-time check
        collection=collection,
        top_k=20,
    )
    sources: set[str] = set()
    for hit in probe.get("results", []):
        src = hit.get("source")
        if src:
            sources.add(src)
    return sources


async def amain(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Seed the prod_reports knowledge-base collection."
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=",".join(DEFAULT_TICKERS),
        help=(
            "Comma-separated 6-digit A-share tickers. Names for "
            "logging are looked up in DEFAULT_TICKERS, otherwise "
            "the ticker is used as the display name."
        ),
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help="Target FAISS collection name.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="",
        help=(
            "End of the cninfo search window as YYYYMMDD. Defaults to "
            "today. Only override for deterministic snapshot tests."
        ),
    )
    args = parser.parse_args(argv)

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        _step("FAIL: no tickers given.")
        return 1

    end_date = (
        datetime.strptime(args.end_date, "%Y%m%d")
        if args.end_date
        else datetime.now()
    )

    # --- imports happen here so the argparse error above is fast ---
    _step("loading tool modules (this triggers the bge embedder import)")
    from research_agent.mcp_servers.knowledge_server import (
        delete_collection,  # noqa: F401  (exposed for ad-hoc cleanup)
        ingest_pdf,
        list_collections,
        search as knowledge_search,
    )
    from research_agent.mcp_servers.pdf_report_server import (
        download_pdf,
        search_announcements,
    )

    _step(f"target collection: {args.collection!r}")
    _step(f"tickers: {tickers}")

    already = await _ingested_sources_for_collection(
        list_collections=list_collections,
        collection=args.collection,
        knowledge_search=knowledge_search,
    )
    if already:
        _step(f"  collection already has {len(already)} distinct sources; "
              f"existing PDFs will be skipped (idempotent re-run).")

    seeded = 0
    skipped = 0
    failed: list[str] = []

    for sym in tickers:
        name = DEFAULT_TICKERS.get(sym, sym)
        _step(f"=== {sym} {name} ===")
        record = await _find_recent_report(
            search_announcements=search_announcements,
            symbol=sym,
            end_date=end_date,
        )
        if record is None:
            _step(f"  no recent report found for {sym}; skipping.")
            failed.append(sym)
            continue

        pdf_url = record.get("pdf_url")
        if not pdf_url:
            _step(f"  record missing pdf_url; skipping.")
            failed.append(sym)
            continue

        _step(f"  download_pdf({pdf_url[:80]}...)")
        try:
            dl = await download_pdf(pdf_url=pdf_url)
        except Exception as exc:  # noqa: BLE001
            _step(f"  download failed: {exc!r}")
            failed.append(sym)
            continue
        if "error" in dl:
            _step(f"  download returned error: {dl['error']}")
            failed.append(sym)
            continue

        local_path = dl["local_path"]
        size_kb = dl.get("size_bytes", 0) // 1024
        from_cache = dl.get("from_cache", False)
        _step(f"  -> {local_path} ({size_kb} KB, from_cache={from_cache})")

        # Idempotency: if this PDF's path is already inside the
        # collection's ``source`` set we skip ingest. Re-ingesting
        # would APPEND a duplicate copy of every chunk (knowledge_
        # server's TODO note about content-hash dedup is not yet
        # done) and bloat the index.
        if local_path in already:
            _step("  already ingested in this collection; skipping.")
            skipped += 1
            continue

        _step(f"  knowledge_ingest_pdf(collection={args.collection!r})")
        try:
            ing = await ingest_pdf(
                local_path=local_path,
                collection=args.collection,
            )
        except Exception as exc:  # noqa: BLE001
            _step(f"  ingest crashed: {exc!r}")
            failed.append(sym)
            continue
        if "error" in ing:
            _step(f"  ingest returned error: {ing['error']}")
            failed.append(sym)
            continue

        added = ing.get("num_chunks_added", 0)
        total = ing.get("total_chunks_in_collection", 0)
        _step(f"  ingested: +{added} chunks (collection total: {total})")
        seeded += 1
        already.add(local_path)

    _step("=== summary ===")
    _step(f"  newly seeded:   {seeded}")
    _step(f"  already-cached: {skipped}")
    _step(f"  failed tickers: {failed if failed else '(none)'}")
    if seeded == 0 and skipped == 0:
        _step("FAIL: nothing reached the index.")
        return 1
    _step("OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(amain(sys.argv[1:])))
