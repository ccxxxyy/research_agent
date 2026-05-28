"""端到端演示 — knowledge_expert 基于真实 PDF + FAISS 的 RAG 流程。

本脚本是 RAG 日常生产流程：以进程内方式加载 ``knowledge_server``工具，将它们接入注册在研究 supervisor 后面的 ``knowledge_expert`` 专家，
然后提出一个问题迫使 Agent 执行：

  1. ``knowledge_ingest_pdf`` — 对一份新生成的 PDF 进行分块 + 向量化，写入全新的 Chroma collection。
  2. ``knowledge_search``     — 混合检索（FAISS 向量 + BM25 + RRF），可选交叉编码器重排序。
  3. 用引用原文 + ``source`` + ``page`` 作答。

运行::

    uv run python scripts/demo_knowledge_expert.py

前置条件:
  - ``.env`` 中已配置可用的 LLM（与 Phase-1/3 冒烟测试相同 — DeepSeek / OpenAI 等）。
    数据面无第三方网络依赖：演示用 PDF 在进程内合成，因此一旦 embedding 模型权重已缓存即可离线运行。
  - 首次运行会下载 ``BAAI/bge-small-zh-v1.5``（约 95 MB）到 ``~/.cache/huggingface``，后续运行即时完成。

退出码:
    0 → supervisor 给出的最终回答通过了宽松校验。
    1 → 任何配置 / MCP / LLM 错误，或 supervisor 未路由到 ``knowledge_expert``，或最终回答未引用已灌入的 PDF 内容。

宽松校验（故意不固定 LLM 措辞）：
  * 最终回答非空
  * supervisor 跟踪中出现 ``transfer_to_knowledge_expert``
  * 回答中引用了 PDF（路径名子串匹配）或包含页码

为何不直接观测 ``knowledge_*`` 工具调用
----------------------------------------
原因与 ``demo_financial_research.py`` 相同：supervisor 使用``output_mode="last_message"`` 编译，因此各专家内部的``ToolMessage`` 通信保留在其子图内。
通过路由记录验证转发，并通过最终回答内容验证正确性。
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import tempfile
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from loguru import logger

# 使演示日志在 Windows 上可读：强制 UTF-8 stdout/stderr，避免通过 PowerShell 的 ``>`` 重定向时中文提示变为乱码。
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    # ``reconfigure`` 自 Py 3.7 起存在于 TextIOWrapper；
    # 如果流已被包装（如 uvicorn）则跳过。
    pass


def _step(msg: str) -> None:
    """打印一行带时间戳的进度标记。

    在本演示中大量使用，因为历史上曾在多个不透明的位置卡住（embedder 冷启动、FAISS I/O、LLM 流式阻塞）。
    有了这些标记，即使 stdout 重定向到文件且进程中途被终止，每一步也清晰可见。
    """
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

from research_agent.config import get_settings
from research_agent.graph.research_supervisor import build_research_supervisor
from research_agent.llm.provider import ModelRouter
from research_agent.mcp_servers.client_factory import (
    load_knowledge_tools_inproc,
)

# ---------------------------------------------------------------------
# 合成 PDF 构造器
#
# 手工构建 PDF，不依赖 reportlab / weasyprint，使演示完全自包含。
# 知识库服务器的单元测试也用了同样的技巧。输出是结构上最小但符合规范的 PDF，pypdf 可以正常解析。
# ---------------------------------------------------------------------

PDF_PAGES = [
    # 第 1 页 — 包含答案的段落。
    "This is the 2024 ESG report for ExampleCorp. 这是一份示例 PDF，用于演示基于真实 PDF + FAISS 的 RAG 流程。我们在这里描述公司的碳中和承诺，"
    "Our carbon neutrality goal is to achieve net-zero scope 1 and "
    "scope 2 emissions by the year 2030, with a 50 percent absolute "
    "reduction milestone by 2027. The board reviews progress annually.",
    # 第 2 页 — 故意设置的干扰段落，主题完全不同，
    # 让检索器必须真正区分相关性。
    "Shareholder dividend policy: ExampleCorp distributes quarterly "
    "cash dividends of 0.12 USD per share, subject to free-cash-flow "
    "coverage. Dividend reinvestment plans are available to holders "
    "of record on the ex-dividend date.",
]


def _make_tiny_pdf(pages: list[str]) -> bytes:
    """返回一个最小多页 PDF 的原始字节。

    每页包含一个 ``Tj`` 文本显示操作，使 ``pypdf`` 在 ``extract_text()`` 时能逐字还原输入字符串。
    """

    def _content_stream(text: str) -> bytes:
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        return f"BT /F1 12 Tf 50 750 Td ({safe}) Tj ET".encode("latin-1")

    objects: list[bytes] = []

    def _push(obj_bytes: bytes) -> int:
        objects.append(obj_bytes)
        return len(objects)

    catalog_obj_num = _push(b"<< /Type /Catalog /Pages 2 0 R >>")
    pages_obj_num = _push(b"")  # 后面会回填
    font_obj_num = _push(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_obj_nums: list[int] = []
    for text in pages:
        stream = _content_stream(text)
        contents_obj_num = _push(
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n"
            + stream + b"\nendstream"
        )
        page_obj = (
            b"<< /Type /Page /Parent " + str(pages_obj_num).encode("ascii") + b" 0 R "
            + b"/MediaBox [0 0 612 792] "
            + b"/Resources << /Font << /F1 " + str(font_obj_num).encode("ascii") + b" 0 R >> >> "
            + b"/Contents " + str(contents_obj_num).encode("ascii") + b" 0 R >>"
        )
        page_obj_nums.append(_push(page_obj))

    kids = b" ".join(str(n).encode("ascii") + b" 0 R" for n in page_obj_nums)
    objects[pages_obj_num - 1] = (
        b"<< /Type /Pages /Count " + str(len(page_obj_nums)).encode("ascii")
        + b" /Kids [" + kids + b"] >>"
    )

    buf = io.BytesIO()
    buf.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(buf.tell())
        buf.write(str(i).encode("ascii") + b" 0 obj\n" + obj + b"\nendobj\n")

    xref_pos = buf.tell()
    buf.write(b"xref\n0 " + str(len(objects) + 1).encode("ascii") + b"\n")
    buf.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        buf.write(f"{off:010d} 00000 n \n".encode("ascii"))
    buf.write(
        b"trailer\n<< /Size " + str(len(objects) + 1).encode("ascii")
        + b" /Root " + str(catalog_obj_num).encode("ascii")
        + b" 0 R >>\nstartxref\n" + str(xref_pos).encode("ascii") + b"\n%%EOF\n"
    )
    return buf.getvalue()


# ---------------------------------------------------------------------
# 跟踪辅助函数（与 demo_financial_research.py 相同的范式）
# ---------------------------------------------------------------------
def _last_plain_assistant(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            tc = getattr(m, "tool_calls", None) or []
            if not tc and m.content:
                return str(m.content)
    return ""


def _trace_tool_calls(messages: list) -> list[str]:
    names: list[str] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            nm = getattr(m, "name", None) or ""
            if nm:
                names.append(nm)
        elif isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                nm = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if nm:
                    names.append(str(nm))
    return names


def _transfers_reached(calls: list[str]) -> set[str]:
    reached: set[str] = set()
    for n in calls:
        if n.startswith("transfer_to_") and n != "transfer_to_supervisor":
            reached.add(n[len("transfer_to_") :])
    return reached


# ---------------------------------------------------------------------
# 流水线
# ---------------------------------------------------------------------
def _make_collection_name() -> str:
    """返回本次运行专属的 collection 名称。

    使用每次运行独立的名称（时间戳后缀）防止旧状态残留 — 跨运行复用同一 collection 名称会静默地重复灌入 PDF，污染 BM25 统计数据，使"高质量"阈值更容易被虚假触发。
    """
    return f"demo_esg_{time.strftime('%Y%m%d_%H%M%S')}"


def _seed_pdf() -> Path:
    """将合成 PDF 写入系统临时目录并返回路径。"""
    tmp_dir = Path(tempfile.gettempdir()) / "research_agent_demo"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / "examplecorp_esg_2024.pdf"
    out.write_bytes(_make_tiny_pdf(PDF_PAGES))
    _step(f"已生成合成 PDF: {out}")
    return out


# supervisor 整体运行的硬性超时上限。模型需要完成：
# ingest_pdf（embedder 冷启动约 3 秒） + 1-2 次 search 调用（预热后亚秒级） + LLM 的路由、规划和最终综合轮次。
# 4 分钟留有充足余量，同时在子进程死锁时能快速失败。
GRAPH_TIMEOUT_SECONDS: float = 240.0


async def main() -> int:
    settings = get_settings()
    router = ModelRouter(settings.llm)
    _step("Settings + ModelRouter 已初始化")

    pdf_path = _seed_pdf()
    collection = _make_collection_name()
    _step(f"本次运行 collection 名称: {collection}")

    try:
        knowledge_tools = await load_knowledge_tools_inproc()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load in-process knowledge tools: {}", exc)
        return 1
    _step(f"知识库工具已加载（进程内）: count={len(knowledge_tools)}")
    for t in knowledge_tools:
        _step(f"  tool: {t.name}")

    graph = build_research_supervisor(
        model_router=router,
        knowledge_tools=knowledge_tools,
    )
    _step("Supervisor 已编译（仅知识库团队）")

    question = (
        f"我刚把一份本地 PDF 路径 `{pdf_path}` 准备好了，请帮我：\n"
        f"  1) 把它灌入我的知识库（collection 名为 `{collection}`）；\n"
        "  2) 然后在该 collection 中检索 2030 年碳中和目标 相关原文，返回最相关的 1-2 段，并在每段后用 `(source=…, page=…)` 标注出处；\n"
        "  3) 用一句中文总结公司的碳中和承诺要点。\n"
        "如果首次检索质量不高，请改写关键词后再试一次再回答。"
    )
    _step(f"正在向 supervisor 发送问题（长度={len(question)} 字符）...")
    print("---- 提示词开始 ----")
    print(question)
    print("---- 提示词结束 ----", flush=True)

    try:
        result = await asyncio.wait_for(
            graph.ainvoke(
                {"messages": [HumanMessage(content=question)]},
                # 灌入可能需要几秒；留出充足余量。
                config={"recursion_limit": 40},
            ),
            timeout=GRAPH_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        logger.error(
            "Graph timed out after {}s — likely an MCP subprocess deadlock or "
            "an LLM stream stall. Re-run with finer logging if this repeats.",
            GRAPH_TIMEOUT_SECONDS,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Graph invocation crashed: {}", exc)
        return 1
    _step("Supervisor 返回了最终状态")

    messages = result["messages"]
    final = _last_plain_assistant(messages)
    calls = _trace_tool_calls(messages)
    reached = _transfers_reached(calls)

    print("\n=== Supervisor 最终回答 ===\n")
    print(final if final else "<空>")
    print("\n=== 跟踪摘要 ===")
    print(f"  消息总数                    : {len(messages)}")
    print(f"  已到达的专家                : {sorted(reached) or ['<无>']}")
    print(f"  工具调用事件总数            : {len(calls)}")
    print(f"  transfer_to_knowledge_expert: "
          f"{sum(1 for n in calls if n == 'transfer_to_knowledge_expert')}")

    # -----------------------------------------------------------------
    # 重排序管线可视化
    #
    # 直接调用 knowledge_server.search() 以表格形式展示每条命中的 rerank_score。这与 LLM 选择引用的内容无关 —
    # 它展示的是原始检索管线输出，用于直观确认交叉编码器是否活跃且正确重排序。
    # -----------------------------------------------------------------
    print("\n=== 重排序管线（直接 search 调用） ===")
    try:
        from research_agent.mcp_servers import knowledge_server as ks

        search_result = await ks.search(
            query="2030 carbon neutrality net-zero scope emissions",
            collection=collection,
            top_k=5,
        )
        if "error" in search_result:
            print(f"  搜索错误: {search_result['error']}")
        else:
            reranker_active = any(
                r.get("rerank_score") is not None
                for r in search_result.get("results", [])
            )
            print(f"  质量           : {search_result['quality']}")
            print(f"  最高分（向量） : {search_result['top_score']}")
            print(f"  平均分         : {search_result['mean_score']}")
            print(f"  重排序器活跃   : {reranker_active}")
            print(f"  返回命中数     : {search_result['top_k_returned']}")
            print()
            print(f"  {'#':<3} {'rerank':>8} {'vector':>8} {'bm25':>8} "
                  f"{'rrf':>10}  {'page':>4}  内容（前 60 字符）")
            print(f"  {'─'*3} {'─'*8} {'─'*8} {'─'*8} {'─'*10}  {'─'*4}  {'─'*40}")
            for i, hit in enumerate(search_result.get("results", []), 1):
                rs = hit.get("rerank_score")
                rs_str = f"{rs:8.4f}" if rs is not None else "    n/a "
                vs = f"{hit.get('vector_score', 0):8.4f}"
                bs = f"{hit.get('bm25_score', 0):8.4f}"
                rrf = f"{hit.get('rrf_score', 0):10.6f}"
                page = f"{hit.get('page', '?'):>4}"
                text = (hit.get("content", "") or "")[:60].replace("\n", " ")
                print(f"  {i:<3} {rs_str} {vs} {bs} {rrf}  {page}  {text}")
            print()
    except Exception as exc:  # noqa: BLE001
        print(f"  （重排序可视化已跳过: {exc}）")

    print("\n=== 启发式校验 ===")
    ok_answer = bool(final.strip())
    ok_route = "knowledge_expert" in reached
    ok_topic = (
        "carbon" in final.lower()
        or "neutrality" in final.lower()
        or "碳中和" in final
        or "2030" in final
    )
    ok_citation = (
        "examplecorp_esg_2024" in final.lower()
        or "page" in final.lower()
        or "page=" in final.lower()
        or "p." in final.lower()
        or "页" in final
    )
    print(f"  最终回答非空                 : {ok_answer}")
    print(f"  已路由到 knowledge_expert    : {ok_route}")
    print(f"  提及主题（carbon/2030）      : {ok_topic}")
    print(f"  回答中包含引用提示           : {ok_citation}")

    if ok_answer and ok_route and ok_topic and ok_citation:
        print("\n  [PASS] knowledge_expert 端到端 RAG 流程验证通过。")
        return 0
    print("\n  [WARN] 启发式校验失败 — 请检查上方跟踪信息。")
    return 1


if __name__ == "__main__":
    # 每次运行使用新的带时间戳的 collection（见 ``_make_collection_name``），
    # 因此磁盘目录开销可控；用户可通过应用内的
    # ``knowledge_list_collections`` / ``knowledge_delete_collection`` 清理。
    sys.exit(asyncio.run(main()))
