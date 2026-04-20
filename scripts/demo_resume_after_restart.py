"""Demonstrate conversation RESUME across a fresh Python process.

Run:
    uv run python scripts/demo_resume_after_restart.py

What makes this demo honest:

*Instead of pretending* that "rebuilding the agent in the same process"
equals "surviving a crash", we actually spawn a **second Python process**
with ``subprocess.run`` and show it reads state written by the first
process. The two processes only share one thing — a SQLite file on disk.
No in-memory trickery.

Scenario timeline:

    [process A, role=writer]
        build_simple_agent(checkpointer=SqliteSaver("./data/resume.db"))
        turn 1: "Remember: the project codename is 'North Star'."
        turn 2: calculate 999 * 37  → 36963
        exit(0)         <-- process A dies here; nothing in memory survives

    [process B, role=reader, spawned by subprocess.run]
        SAME SQLite file, SAME thread_id, BRAND NEW ModelRouter & agent
        turn 3: "What is the codename? What was the multiplication result?"
        <-- correct recall from DB proves the resume works

The scenario is written so the reader can distinguish a real resume
from an accidentally-shared in-memory object: the writer and reader
never touch the same Python objects.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger


DB_PATH = Path("data/demo_resume.sqlite")
THREAD_ID = "demo-resume-codename"


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


async def _build_stateful_agent():
    from research_agent.agents.simple import build_simple_agent
    from research_agent.config import get_settings
    from research_agent.llm.provider import ModelRouter
    from research_agent.memory.checkpointer import init_checkpointer

    settings = get_settings()
    router = ModelRouter(settings.llm)
    checkpointer = await init_checkpointer(sqlite_path=DB_PATH)
    agent = build_simple_agent(router, checkpointer=checkpointer)
    return agent


def _last_ai_text(result: dict) -> str:
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            return str(msg.content)
    return "<no final answer>"


async def role_writer() -> None:
    _banner("Process A (writer): storing facts into SQLite and exiting")

    agent = await _build_stateful_agent()
    cfg = {"configurable": {"thread_id": THREAD_ID}}

    for step, question in enumerate(
        [
            "Remember this fact: the project codename is 'North Star'. Confirm you will remember.",
            "Now calculate 999 * 37 using your tool. Store the result in context.",
        ],
        start=1,
    ):
        result = await agent.ainvoke({"messages": [HumanMessage(content=question)]}, config=cfg)
        print(f"  writer turn {step}")
        print(f"    user : {question}")
        print(f"    ai   : {_last_ai_text(result)}")

    snapshot = await agent.aget_state(cfg)
    print(
        f"\n  writer exit — persisted {len(snapshot.values.get('messages', []))} messages "
        f"to {DB_PATH.resolve()}"
    )
    print("  --> Process A terminating now. Memory wiped.")


async def role_reader() -> None:
    _banner("Process B (reader): a brand-new Python process opens the DB")

    agent = await _build_stateful_agent()
    cfg = {"configurable": {"thread_id": THREAD_ID}}

    snapshot = await agent.aget_state(cfg)
    persisted = len(snapshot.values.get("messages", []))
    print(f"  reader startup — found {persisted} pre-existing messages in SQLite")

    if persisted == 0:
        print("  [FATAL] nothing to resume from. Did the writer run first?")
        sys.exit(2)

    probe = (
        "Two questions: (a) what is the project codename I told you? "
        "(b) what was the multiplication result from earlier?"
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=probe)]}, config=cfg)

    print(f"\n  reader probe")
    print(f"    user : {probe}")
    print(f"    ai   : {_last_ai_text(result)}")

    # Sanity checks that the answer references the facts from process A.
    answer = _last_ai_text(result).lower()
    checks = {
        "codename recalled": "north star" in answer,
        "multiplication recalled (36963)": "36963" in answer or "36,963" in answer,
    }
    print("\n  resume verification:")
    for check, passed in checks.items():
        verdict = "PASS" if passed else "FAIL"
        print(f"    [{verdict}] {check}")

    if not all(checks.values()):
        sys.exit(1)


async def role_orchestrator() -> None:
    """Entry point: clean slate, run writer in subprocess A, then reader in B."""
    import subprocess

    if DB_PATH.exists():
        logger.info("Removing stale DB at {}", DB_PATH)
        DB_PATH.unlink()

    _banner("Orchestrator — two-process resume demo begins")
    print(f"  SQLite path : {DB_PATH.resolve()}")
    print(f"  thread_id   : {THREAD_ID}")

    # ---- Process A ----
    print("\n  >>> spawning Process A (writer)...")
    proc_a = subprocess.run(
        [sys.executable, __file__, "writer"],
        check=True,
    )
    print(f"  <<< Process A exited with code {proc_a.returncode}")

    # ---- Process B ----
    print("\n  >>> spawning Process B (reader)...")
    proc_b = subprocess.run(
        [sys.executable, __file__, "reader"],
    )
    print(f"  <<< Process B exited with code {proc_b.returncode}")

    _banner("Takeaway")
    if proc_b.returncode == 0:
        print(
            """
  The reader process — which NEVER shared a single Python object with
  the writer — correctly recalled both:
    * a plain-text fact ("North Star")
    * a tool-computed numeric result (36963 = 999 * 37)

  The only channel between the two processes is the SQLite file. That
  means the entire conversation state, including tool-call artefacts,
  survived a simulated crash / redeploy / horizontal scale-out event.

  In a production FastAPI deployment, swap SqliteSaver for PostgresSaver
  and the exact same code path works — multi-instance, HA-grade.
            """.rstrip()
        )
    else:
        print("  Demo FAILED — reader process did not recover expected facts.")
        sys.exit(proc_b.returncode)


def main() -> None:
    role = sys.argv[1] if len(sys.argv) > 1 else "orchestrator"

    if role == "orchestrator":
        asyncio.run(role_orchestrator())
    elif role == "writer":
        asyncio.run(role_writer())
    elif role == "reader":
        asyncio.run(role_reader())
    else:
        print(f"Unknown role: {role}. Use one of: orchestrator, writer, reader.")
        sys.exit(2)


if __name__ == "__main__":
    main()
