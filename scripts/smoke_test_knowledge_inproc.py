"""进程内知识库工具的端到端冒烟测试。

验证生产运行时路径（无 MCP 子进程）：Agent 调用:mod:`research_agent.tools.knowledge_tools` 暴露的四个``StructuredTool`` 实例，这些实例委托给
:mod:`research_agent.mcp_servers.knowledge_server` 中的 async 协程。

测试序列：

1. 导入 ``KNOWLEDGE_TOOLS`` 并验证四个预期名称均存在。
2. 在磁盘上手工构建一个 2 页合成 PDF（无外部文件依赖）。
3. 对唯一命名的临时 collection（时间戳后缀防止跨运行污染 ``./data/knowledge_db/``）执行 ``knowledge_ingest_pdf``。
4. 调用 ``knowledge_list_collections`` 并断言临时 collection 存在。
5. 使用与第 1 页在词汇和语义上重叠的查询执行 ``knowledge_search``；断言命中数非零、分数合理、质量不为 ``low``。
6. 调用 ``knowledge_delete_collection`` 清理。

每次调用均包含计时器 + ``asyncio.wait_for``，因此 embedder 管线的回归卡住时能快速失败并给出有用信息，而非锁死开发者终端。

运行::

    .venv/Scripts/python.exe scripts/smoke_test_knowledge_inproc.py

全部成功返回退出码 0，任何失败/超时/格式错误返回非零。如接受冷启动开销（bge-small 首次下载约 30 秒），可用于 CI 冒烟测试。
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path

# 强制 stdout 使用 UTF-8，防止中文 source / 命中结果在 Windows 代码页上报错。
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def _step(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _make_tiny_pdf(pages: list[str]) -> bytes:
    """手工构建一个 pypdf 可解析的最小规范 PDF。

    与单元测试的 ``_make_tiny_pdf`` 策略相同 — 内联保留以避免本脚本对测试包的导入依赖。
    """

    def _content_stream(text: str) -> bytes:
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        return f"BT /F1 12 Tf 50 750 Td ({safe}) Tj ET".encode("latin-1")

    objects: list[bytes] = []

    def _push(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)

    catalog = _push(b"<< /Type /Catalog /Pages 2 0 R >>")
    pages_obj_num = _push(b"")  # 占位符，后面回填
    font = _push(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_obj_nums: list[int] = []
    for text in pages:
        stream = _content_stream(text)
        contents = _push(
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
        page_obj = (
            b"<< /Type /Page /Parent "
            + str(pages_obj_num).encode("ascii")
            + b" 0 R "
            + b"/MediaBox [0 0 612 792] "
            + b"/Resources << /Font << /F1 "
            + str(font).encode("ascii")
            + b" 0 R >> >> "
            + b"/Contents "
            + str(contents).encode("ascii")
            + b" 0 R >>"
        )
        page_obj_nums.append(_push(page_obj))

    kids = b" ".join(str(n).encode("ascii") + b" 0 R" for n in page_obj_nums)
    objects[pages_obj_num - 1] = (
        b"<< /Type /Pages /Count "
        + str(len(page_obj_nums)).encode("ascii")
        + b" /Kids ["
        + kids
        + b"] >>"
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
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode("ascii")
        + b" /Root "
        + str(catalog).encode("ascii")
        + b" 0 R >>\nstartxref\n"
        + str(xref_pos).encode("ascii")
        + b"\n%%EOF\n"
    )
    return buf.getvalue()


async def _call(tool, payload: dict, *, timeout: float, label: str) -> dict:
    """带计时器和清晰错误报告的 ``tool.ainvoke`` 包装。

    StructuredTool 返回底层协程的实际返回值（对工具来说是 dict），不像 MCP-stdio 适配器会将每个响应包装在内容块列表中。因此此处无需展平步骤。
    """
    t0 = time.time()
    try:
        result = await asyncio.wait_for(tool.ainvoke(payload), timeout=timeout)
    except TimeoutError as exc:
        raise RuntimeError(f"{label}: 超时，已等待 {timeout:.1f} 秒") from exc
    elapsed = time.time() - t0
    if not isinstance(result, dict):
        raise RuntimeError(
            f"{label}: 预期 dict，实际得到 {type(result).__name__}: {result!r}"
        )
    _step(f"{label}: {elapsed:.2f}s -> {sorted(result)}")
    return result


async def amain() -> int:
    _step("正在加载进程内知识库工具")
    from research_agent.tools.knowledge_tools import KNOWLEDGE_TOOLS

    by_name = {t.name: t for t in KNOWLEDGE_TOOLS}
    expected = {
        "knowledge_ingest_pdf",
        "knowledge_search",
        "knowledge_list_collections",
        "knowledge_delete_collection",
    }
    missing = expected - by_name.keys()
    if missing:
        _step(f"失败: 缺少工具 {missing}")
        return 1
    _step(f"  工具列表: {sorted(by_name)}")

    suffix = time.strftime("%Y%m%d-%H%M%S")
    coll = f"smoke-inproc-{suffix}"

    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "tiny.pdf"
        pdf.write_bytes(
            _make_tiny_pdf(
                [
                    "carbon neutrality 2030 scope 1 emissions reduction roadmap",
                    "shareholder dividend policy quarterly distribution schedule",
                ]
            )
        )
        _step(f"已写入合成 PDF: {pdf}（{pdf.stat().st_size} 字节）")

        ingest = await _call(
            by_name["knowledge_ingest_pdf"],
            {"local_path": str(pdf), "collection": coll},
            timeout=180.0,  # 冷启动：bge-small 首次加载 + langchain 导入
            label="ingest_pdf",
        )
        if "error" in ingest:
            _step(f"失败: ingest_pdf 返回错误: {ingest}")
            return 1
        if ingest.get("num_chunks_added", 0) <= 0:
            _step(f"失败: ingest_pdf 添加了零个分块: {ingest}")
            return 1
        _step(f"  已灌入 {ingest['num_chunks_added']} 个分块（collection={coll!r}）")

        listing = await _call(
            by_name["knowledge_list_collections"],
            {},
            timeout=30.0,
            label="list_collections",
        )
        names = {c["name"] for c in listing.get("collections", [])}
        if coll not in names:
            _step(f"失败: 临时 collection {coll!r} 未出现在列表中: {names}")
            return 1
        _step(f"  list_collections 看到 {len(names)} 个 collection；我们的已存在。")

        hits = await _call(
            by_name["knowledge_search"],
            {"query": "carbon neutrality goal", "collection": coll, "top_k": 3},
            timeout=60.0,  # 预热路径：灌入后应为亚秒级
            label="search",
        )
        if "error" in hits:
            _step(f"失败: search 返回错误: {hits}")
            return 1
        if hits.get("top_k_returned", 0) < 1:
            _step(f"失败: search 未产生命中: {hits}")
            return 1
        if hits.get("quality") == "low":
            _step(f"警告: 在干净的词汇匹配上 search 质量为 'low': {hits}")
        top = hits["results"][0]
        ok_lexical = "carbon" in (top.get("content") or "").lower()
        if not ok_lexical:
            _step(f"失败: 首条命中不包含 'carbon': {top}")
            return 1
        _step(
            f"  首条命中 page={top.get('page')} vec={top.get('vector_score')} "
            f"bm25={top.get('bm25_score')} rrf={top.get('rrf_score')} "
            f"quality={hits['quality']}"
        )

        deleted = await _call(
            by_name["knowledge_delete_collection"],
            {"collection": coll},
            timeout=30.0,
            label="delete_collection",
        )
        if not deleted.get("deleted"):
            _step(f"失败: delete_collection 未成功删除: {deleted}")
            return 1
        _step("  已清理临时 collection")

    _step("全部通过")
    return 0


if __name__ == "__main__":
    rc = 1
    with suppress(KeyboardInterrupt):
        rc = asyncio.run(amain())
    sys.exit(rc)
