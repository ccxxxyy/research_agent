"""演示使用 checkpointer 支持的多轮有状态对话。

运行::

    uv run python scripts/demo_stateful_multiturn.py

本演示证明的内容：

1. ReAct Agent 在给定 LangGraph checkpointer 和``thread_id`` 后，会自动记住对话的每一轮历史 — 应用侧无需手动管理历史记录。

2. 同一个 Agent 在不同的 ``thread_id`` 上调用时，看到的是空历史。这就是线程隔离：用户和会话在同一进程内相互沙箱化。

3. 保存的状态是丰富的：它包含完整的消息列表，加上之前轮次的所有中间工具调用和工具结果。LLM 在下一轮可以基于这些历史进行推理。

此处使用 ``MemorySaver``，使演示完全自包含。"能否在进程重启后恢复？"这个问题在另一个演示（``demo_resume_after_restart.py``）中通过 ``SqliteSaver`` 来回答。
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
    """从 Agent 调用结果中提取最后一条 AIMessage 的文本。"""
    from langchain_core.messages import AIMessage

    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            return str(msg.content)
    return "<无最终回答>"


async def turn(agent, thread_id: str, question: str, step: int) -> None:
    cfg = {"configurable": {"thread_id": thread_id}}
    result = await agent.ainvoke({"messages": [HumanMessage(content=question)]}, config=cfg)

    print(f"\n  第 {step} 轮  [thread={thread_id}]")
    print(f"    用户 : {question}")
    print(f"    AI   : {_last_ai_text(result)}")
    print(f"    本轮后历史长度 : {len(result['messages'])} 条消息")


async def peek_state(agent, thread_id: str) -> None:
    """直接检查 checkpoint 状态以证明其已被持久化。"""
    cfg = {"configurable": {"thread_id": thread_id}}
    snapshot = await agent.aget_state(cfg)

    print(f"\n  状态快照 [thread={thread_id}]")
    if snapshot.values and snapshot.values.get("messages"):
        msgs = snapshot.values["messages"]
        print(f"    已持久化消息数  : {len(msgs)}")
        print(f"    消息类型        : {[type(m).__name__ for m in msgs]}")
        print(f"    checkpoint id   : {snapshot.config['configurable'].get('checkpoint_id')}")
    else:
        print("    （空 — 尚无 checkpoint）")


async def main() -> None:
    settings = get_settings()
    logger.info("Light model: {}", settings.llm.light_model)

    router = ModelRouter(settings.llm)
    checkpointer = MemorySaver()
    agent = build_simple_agent(router, checkpointer=checkpointer)

    # ------------------------------------------------------------------
    _banner("场景 1 — 线程 'alice'：三轮连续对话")
    # 每轮只发送一条新的 HumanMessage。checkpointer 自动在前面拼接之前的消息。如果第 2、3 轮能正确引用之前的事实（"我叫 Alice"、"6, 9, 12"），则说明记忆生效。
    # ------------------------------------------------------------------
    await turn(agent, "alice", "Hi! My name is Alice. Please remember that.", 1)
    await turn(agent, "alice", "Now please calculate 6 + 9 + 12 using your tool.", 2)
    await turn(agent, "alice", "What's my name, and what was the sum I just asked about?", 3)
    await peek_state(agent, "alice")

    # ------------------------------------------------------------------
    _banner("场景 2 — 线程 'bob'：与 alice 完全隔离")
    # Bob 从未交互过。如果线程隔离有效，LLM 将不知道 Alice 的
    # 名字或 6+9+12 的结果。
    # ------------------------------------------------------------------
    await turn(agent, "bob", "Do you know my name, or any previous calculation?", 1)
    await peek_state(agent, "bob")

    # ------------------------------------------------------------------
    _banner("场景 3 — alice 线程继续：记忆仍然存在")
    # ------------------------------------------------------------------
    await turn(
        agent,
        "alice",
        "Multiply the previous sum by 10 using your tool. State the result clearly.",
        4,
    )
    await peek_state(agent, "alice")

    _banner("要点总结")
    print(
        """
  * 应用层从未自行存储消息 — LangGraph checkpointer 拥有完整的 历史记录，并在相同 thread_id 下的每次 Agent 调用时回放。

  * 线程隔离是多租户的基本单位：一个用户、一个会话、一次文档审查 — 选择与产品匹配的粒度即可。

  * 由于状态是 LangChain 消息列表（Human / AI / Tool），LLM也能看到之前轮次的工具调用产物。这就是它能回忆"上次的求和结果"并链式计算的原因。

  * 使用 MemorySaver 时，以上所有内容在进程退出后即消失。
    见 ``demo_resume_after_restart.py`` 了解可持久化的 SQLite 变体。
        """.rstrip()
    )


if __name__ == "__main__":
    asyncio.run(main())
