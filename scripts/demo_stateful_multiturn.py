"""Demonstrate multi-turn conversation with a checkpointer-backed agent.

Run:
    uv run python scripts/demo_stateful_multiturn.py

What this demo proves:

1. The Phase-1 ReAct agent, when given a LangGraph checkpointer and a
   ``thread_id``, automatically REMEMBERS every prior turn of the
   conversation — no manual history management on the application side.

2. The SAME agent, invoked on a DIFFERENT ``thread_id``, sees an empty
   history. This is thread isolation: users and sessions are sandboxed
   from each other inside the same process.

3. The saved state is rich: it contains the full message list PLUS any
   intermediate tool calls and tool results from prior turns. The LLM
   can reason over that history on the next turn.

We use ``MemorySaver`` here so the demo is fully self-contained. The
"does it survive a process restart?" question is answered in a separate
demo (``demo_resume_after_restart.py``) using ``SqliteSaver``.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from research_agent.agents.simple import build_simple_agent
from research_agent.config import get_settings
from research_agent.llm.provider import ModelRouter


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _last_ai_text(result: dict) -> str:
    """Extract the text of the final AIMessage from an agent invocation."""
    from langchain_core.messages import AIMessage

    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            return str(msg.content)
    return "<no final answer>"


async def turn(agent, thread_id: str, question: str, step: int) -> None:
    cfg = {"configurable": {"thread_id": thread_id}}
    result = await agent.ainvoke({"messages": [HumanMessage(content=question)]}, config=cfg)

    print(f"\n  turn {step}  [thread={thread_id}]")
    print(f"    user : {question}")
    print(f"    ai   : {_last_ai_text(result)}")
    print(f"    history length after turn : {len(result['messages'])} messages")


async def peek_state(agent, thread_id: str) -> None:
    """Directly inspect the checkpointed state to prove it's persisted."""
    cfg = {"configurable": {"thread_id": thread_id}}
    snapshot = await agent.aget_state(cfg)

    print(f"\n  state snapshot [thread={thread_id}]")
    if snapshot.values and snapshot.values.get("messages"):
        msgs = snapshot.values["messages"]
        print(f"    messages persisted : {len(msgs)}")
        print(f"    message types      : {[type(m).__name__ for m in msgs]}")
        print(f"    checkpoint id      : {snapshot.config['configurable'].get('checkpoint_id')}")
    else:
        print("    (empty — no checkpoint yet)")


async def main() -> None:
    settings = get_settings()
    logger.info("Light model: {}", settings.llm.light_model)

    router = ModelRouter(settings.llm)
    checkpointer = MemorySaver()
    agent = build_simple_agent(router, checkpointer=checkpointer)

    # ------------------------------------------------------------------
    _banner("Scenario 1 — thread 'alice': three sequential turns")
    # Each turn only sends ONE new HumanMessage. The checkpointer auto-
    # prepends prior messages. We can tell memory works if turn 2 and 3
    # correctly reference earlier facts ("my name is Alice", "6, 9, 12").
    # ------------------------------------------------------------------
    await turn(agent, "alice", "Hi! My name is Alice. Please remember that.", 1)
    await turn(agent, "alice", "Now please calculate 6 + 9 + 12 using your tool.", 2)
    await turn(agent, "alice", "What's my name, and what was the sum I just asked about?", 3)
    await peek_state(agent, "alice")

    # ------------------------------------------------------------------
    _banner("Scenario 2 — thread 'bob': fully isolated from alice")
    # Bob has NEVER interacted before. If thread isolation works, the LLM
    # will NOT know Alice's name or the 6+9+12 sum.
    # ------------------------------------------------------------------
    await turn(agent, "bob", "Do you know my name, or any previous calculation?", 1)
    await peek_state(agent, "bob")

    # ------------------------------------------------------------------
    _banner("Scenario 3 — alice thread continues: memory is still there")
    # ------------------------------------------------------------------
    await turn(
        agent,
        "alice",
        "Multiply the previous sum by 10 using your tool. State the result clearly.",
        4,
    )
    await peek_state(agent, "alice")

    _banner("Takeaway")
    print(
        """
  * The application never stored messages itself — the LangGraph
    checkpointer owns the entire history and replays it into every
    agent invocation under the same thread_id.

  * Thread isolation is the unit of multi-tenancy: one user, one
    session, one document review — pick whatever maps to your product.

  * Because the state is a LIST of LangChain messages (Human / AI / Tool),
    the LLM sees tool-call artefacts from earlier turns too. That's how
    it could recall "the previous sum" and chain the calculation.

  * With MemorySaver this all evaporates on process exit. See
    ``demo_resume_after_restart.py`` for the SQLite variant that survives.
        """.rstrip()
    )


if __name__ == "__main__":
    asyncio.run(main())
