"""演示跨 Python 进程重启后恢复对话。

运行::

    uv run python scripts/demo_resume_after_restart.py

本演示的诚实之处：

不假装 "在同一进程中重建 Agent"等于"从崩溃中恢复"，而是实际启动第二个 Python 进程（通过 ``subprocess.run``），并展示它能读取第一个进程写入的状态。
两个进程唯一共享的是磁盘上的一个 SQLite 文件。没有内存层面的花招。

场景时间线：

    [进程 A，角色=写入者]
        build_simple_agent(checkpointer=SqliteSaver("./data/resume.db"))
        第 1 轮: "记住：项目代号是 'North Star'。"
        第 2 轮: 计算 999 * 37  → 36963
        exit(0)         <-- 进程 A 在此退出；内存中的一切都不复存在

    [进程 B，角色=读取者，由 subprocess.run 启动]
        同一 SQLite 文件，同一 thread_id，全新的 ModelRouter 和 Agent
        第 3 轮: "项目代号是什么？之前的乘法结果是多少？"
        <-- 从数据库正确回忆证明恢复有效

场景经过精心设计，使读者能区分真正的恢复和意外共享的内存对象：写入者和读取者从未接触过相同的 Python 对象。
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
    return "<无最终回答>"


async def role_writer() -> None:
    _banner("进程 A（写入者）：将事实存入 SQLite 并退出")

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
        print(f"  写入者第 {step} 轮")
        print(f"    用户 : {question}")
        print(f"    AI   : {_last_ai_text(result)}")

    snapshot = await agent.aget_state(cfg)
    print(
        f"\n  写入者退出 — 已持久化 {len(snapshot.values.get('messages', []))} 条消息 "
        f"到 {DB_PATH.resolve()}"
    )
    print("  --> 进程 A 即将终止。内存已清除。")


async def role_reader() -> None:
    _banner("进程 B（读取者）：全新的 Python 进程打开数据库")

    agent = await _build_stateful_agent()
    cfg = {"configurable": {"thread_id": THREAD_ID}}

    snapshot = await agent.aget_state(cfg)
    persisted = len(snapshot.values.get("messages", []))
    print(f"  读取者启动 — 在 SQLite 中发现 {persisted} 条已有消息")

    if persisted == 0:
        print("  [致命错误] 无可恢复的内容。写入者是否运行过？")
        sys.exit(2)

    probe = (
        "Two questions: (a) what is the project codename I told you? "
        "(b) what was the multiplication result from earlier?"
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=probe)]}, config=cfg)

    print("\n  读取者探测")
    print(f"    用户 : {probe}")
    print(f"    AI   : {_last_ai_text(result)}")

    # 检查回答是否引用了进程 A 中的事实。
    answer = _last_ai_text(result).lower()
    checks = {
        "代号已回忆": "north star" in answer,
        "乘法结果已回忆 (36963)": "36963" in answer or "36,963" in answer,
    }
    print("\n  恢复验证:")
    for check, passed in checks.items():
        verdict = "PASS" if passed else "FAIL"
        print(f"    [{verdict}] {check}")

    if not all(checks.values()):
        sys.exit(1)


async def role_orchestrator() -> None:
    """入口：清理旧数据，在子进程 A 中运行写入者，然后在子进程 B 中运行读取者。"""
    import subprocess

    if DB_PATH.exists():
        logger.info("Removing stale DB at {}", DB_PATH)
        DB_PATH.unlink()

    _banner("编排器 — 双进程恢复演示开始")
    print(f"  SQLite 路径  : {DB_PATH.resolve()}")
    print(f"  thread_id    : {THREAD_ID}")

    # ---- 进程 A ----
    print("\n  >>> 启动进程 A（写入者）...")
    proc_a = subprocess.run(
        [sys.executable, __file__, "writer"],
        check=True,
    )
    print(f"  <<< 进程 A 退出，返回码 {proc_a.returncode}")

    # ---- 进程 B ----
    print("\n  >>> 启动进程 B（读取者）...")
    proc_b = subprocess.run(
        [sys.executable, __file__, "reader"],
    )
    print(f"  <<< 进程 B 退出，返回码 {proc_b.returncode}")

    _banner("要点总结")
    if proc_b.returncode == 0:
        print(
            """
  读取者进程 — 与写入者从未共享过任何 Python 对象 — 正确
  回忆了两者：
    * 纯文本事实（"North Star"）
    * 工具计算的数值结果（36963 = 999 * 37）

  两个进程之间唯一的通道是 SQLite 文件。这意味着完整的对话状态（包括工具调用产物）经受住了模拟的崩溃 / 重部署 /水平扩展事件。

  在生产 FastAPI 部署中，将 SqliteSaver 换成 PostgresSaver，
  完全相同的代码路径即可工作 — 支持多实例、高可用级别。
            """.rstrip()
        )
    else:
        print("  演示失败 — 读取者进程未能恢复预期事实。")
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
        print(f"未知角色: {role}。请使用: orchestrator, writer, reader。")
        sys.exit(2)


if __name__ == "__main__":
    main()
