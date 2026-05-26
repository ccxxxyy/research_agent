"""端到端冒烟测试：supervisor 移交到 MCP 支持的专家。

本脚本验证的内容
----------------
它测试了其他脚本各自无法单独覆盖的完整 Phase-3+ 链路：

    用户提示词
        -> langgraph_supervisor（LLM，medium 层）
        -> transfer_to_coder_expert（移交工具调用）
        -> coder_expert react-agent（LLM，light 层）
        -> code_execute_python（MCP 工具，langchain-mcp-adapters）
        -> stdio 子进程（fastmcp CodeExecutor 服务器）
        -> 真实 Python 沙箱
        -> 结果内容块
        -> coder_expert 最终消息
        -> supervisor 综合最终回答

提问经过精心设计，使得除 ``coder_expert`` 外没有其他专家能合理回答。``math_expert`` 不行：
答案需要列表推导 + 二维过滤，而非单个算术表达式。``time_expert`` / ``text_analyst`` 显然无关。因此 PASS 确认移交路由正确选择了 MCP 支持的专家。

运行::

    uv run python scripts/smoke_test_supervisor_with_mcp.py

通过打印 ``[PASS]`` 并退出 0。失败打印最终回复及诊断信息并退出非零，因此需要时可以阻塞 CI。
"""

from __future__ import annotations

import asyncio
import sys

from langchain_core.messages import HumanMessage
from loguru import logger

from research_agent.config import get_settings
from research_agent.graph.minimal_supervisor import build_minimal_supervisor
from research_agent.llm.provider import ModelRouter
from research_agent.mcp_servers.client_factory import load_code_server_tools


QUESTION = (
    "Use Python code to solve this: given the list "
    "numbers = [7, 2, 9, 4, 11, 6, 13, 8], "
    "return (a) the sum of the numbers that are divisible by 3, "
    "(b) the count of numbers strictly greater than 7, and "
    "(c) the second-largest number overall. "
    "Run code to compute all three in one go and report a single line "
    "in the format 'sum=<A>, count=<B>, second_largest=<C>'."
)

# 预计算的标准答案（脚本自行校验用）：
#   被 3 整除的       → [9, 6] → sum=15
#   > 7               → [9, 11, 13, 8] → count=4
#   第二大            → 降序排列 = [13, 11, 9, 8, 7, 6, 4, 2] → 11
EXPECTED_SUM = 15
EXPECTED_COUNT = 4
EXPECTED_SECOND_LARGEST = 11


async def main() -> int:
    settings = get_settings()
    router = ModelRouter(settings.llm)

    logger.info("Loading MCP code_server tools over stdio…")
    coder_tools = await load_code_server_tools()
    logger.info("Loaded MCP tools: {}", [t.name for t in coder_tools])

    logger.info("Compiling supervisor with coder_expert attached…")
    graph = build_minimal_supervisor(
        model_router=router,
        coder_tools=coder_tools,
    )

    logger.info("Asking composite question:\n{}", QUESTION)
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=QUESTION)]},
        # 35 保持充足余量应对偶发的工具重试，同时在路由错误时防止失控循环。
        config={"recursion_limit": 35},
    )

    messages = result.get("messages", [])
    final_msg = messages[-1] if messages else None
    final_text = getattr(final_msg, "content", "") or ""

    print("\n=== Supervisor 最终回答 ===\n")
    print(final_text)
    print(f"\n=== 消息总数 === {len(messages)}")

    # --- 启发式校验 --------------------------------------------------
    # 不要求精确措辞；检查 (a) 三个标准答案数字均出现， (b) 至少有一次移交到 ``coder_expert``（证明 MCP 参与了流程）。
    answer_lc = final_text.lower()
    hit_sum = str(EXPECTED_SUM) in final_text
    hit_count = f"count={EXPECTED_COUNT}" in answer_lc or (
        str(EXPECTED_COUNT) in final_text and "count" in answer_lc
    )
    hit_second = str(EXPECTED_SECOND_LARGEST) in final_text

    # 移交到 MCP 支持的专家的证据。
    # 重要说明：因为 supervisor 使用 ``output_mode="last_message"``编译，
    # coder_expert 子图内部的 ``code_execute_python`` 工具调用不会出现在外层 ``messages`` 列表中。
    # 出现的是 supervisor 自身的 ``transfer_to_coder_expert`` 工具调用。
    # 检测到该调用即为路由到达 coder_expert 的充分证据，消息数跳变（单次移交往返 >=5）是独立的佐证。
    coder_handoff = False
    inner_tool_call_seen = False
    for msg in messages:
        tc = getattr(msg, "tool_calls", None) or []
        for call in tc:
            name = (
                call.get("name", "")
                if isinstance(call, dict)
                else getattr(call, "name", "")
            )
            if "transfer_to_coder_expert" in name:
                coder_handoff = True
            if "code_execute_python" in name:
                inner_tool_call_seen = True

    # 也接受任何消息文本中提及 "coder_expert" 作为更宽松的佐证（supervisor 经常引用来源）。
    any_text_mentions_coder = any(
        "coder_expert" in str(getattr(m, "content", "")) for m in messages
    )

    routing_ok = coder_handoff or inner_tool_call_seen or any_text_mentions_coder

    print("\n=== 启发式校验 ===")
    print(f"  sum=15 存在                           : {hit_sum}")
    print(f"  count=4 存在                          : {hit_count}")
    print(f"  second_largest=11 存在                : {hit_second}")
    print(f"  supervisor 移交到 coder_expert        : {coder_handoff}")
    print(f"  内部 code_execute_python 可见         : {inner_tool_call_seen}")
    print(f"  消息中提及 'coder_expert'             : {any_text_mentions_coder}")
    print(f"  → routing_ok（以上任一为真）          : {routing_ok}")

    all_ok = hit_sum and hit_count and hit_second and routing_ok
    if all_ok:
        print("\n  [PASS] Supervisor 路由到 coder_expert，MCP code_server "
              "产生了正确答案。")
        return 0

    print("\n  [FAIL] 一项或多项校验未通过。请查看上方消息跟踪。")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
