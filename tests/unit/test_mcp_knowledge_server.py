"""单元测试 — ``knowledge_server`` MCP 工具与辅助函数。

两层覆盖范围：

1. 纯辅助函数测试（始终运行）：使用一个微小的合成内存 PDF 测试分词器、BM25 索引、混合融合数学、质量分类器、分块器和 PDF 加载器。无网络、无模型权重 — 运行时间 <1 秒。

2. 端到端摄入 + 搜索（标记为 ``slow``）：在临时目录下构建真实 FAISS 索引，摄入微小 PDF，并运行 ``search()`` 往返。
   首次运行时会拉取 bge-small embedding 模型（约 95 MB 缓存在 ``~/.cache/huggingface`` 下）；后续运行已预热，几秒内完成。

慢速层通过 ``pytest -m slow`` 选择性加入，这样 CI 不用在每次提交时承担 embedding 下载开销。``pytest``（默认）运行快速辅助函数测试；
演示脚本 ``scripts/demo_knowledge_expert.py`` 是真正的端到端冒烟测试。
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest

from research_agent.mcp_servers import knowledge_server
from research_agent.mcp_servers.knowledge_server import (
    QUALITY_HIGH_THRESHOLD,
    QUALITY_MEDIUM_THRESHOLD,
    _BM25Index,
    _chunk_pages,
    _classify_quality,
    _hybrid_fuse,
    _load_pdf_pages,
    _validate_collection_name,
)
from research_agent.rag import faiss_store

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------
# 辅助函数 — 合成内存 PDF
# ---------------------------------------------------------------------


def _make_tiny_pdf(pages: list[str]) -> bytes:
    """使用 ``pypdf`` 将 ``pages`` 渲染为真实（微小）PDF。

    避免在仓库中附带 fixture PDF，因为快速层必须在全新克隆中工作。
    ``pypdf`` 无法从零创建，所以使用 ``reportlab``（如果可用）— 否则退回到一个硬编码的 2 页 PDF，
    其中页面文本是在内容流的 BT/ET 标记之间逐字捕获的 ASCII。后者是实际使用的方式，以保持依赖最少（``pypdf`` 已是必需项）。

    结果是一个结构最小但符合规范的 PDF，``pypdf.PdfReader`` 可以正常解析。
    """
    # 手工构建的最小 PDF 骨架。我们只需要文本提取往返能工作；并不试图进行任何可视化渲染。pypdf 的文本提取器读取 ``BT/ET`` 内的 ``Tj`` 操作符。

    def _content_stream(text: str) -> bytes:
        # 转义括号；PDF 字符串以 () 分隔。
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        body = f"BT /F1 12 Tf 50 750 Td ({safe}) Tj ET".encode("latin-1")
        return body

    objects: list[bytes] = []

    def _push(obj_bytes: bytes) -> int:
        objects.append(obj_bytes)
        return len(objects)

    # 1. 目录对象
    catalog_obj_num = _push(b"<< /Type /Catalog /Pages 2 0 R >>")
    # 2. 页面树（前向引用占位符；页面对象创建后再填充子节点列表）。
    pages_obj_num = _push(b"")
    # 字体（所有页面共享）
    font_obj_num = _push(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_obj_nums: list[int] = []
    for text in pages:
        stream = _content_stream(text)
        contents_obj_num = _push(
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
            + str(font_obj_num).encode("ascii")
            + b" 0 R >> >> "
            + b"/Contents "
            + str(contents_obj_num).encode("ascii")
            + b" 0 R >>"
        )
        page_obj_nums.append(_push(page_obj))

    # 用实际子节点引用填充页面树。
    kids = b" ".join(str(n).encode("ascii") + b" 0 R" for n in page_obj_nums)
    objects[pages_obj_num - 1] = (
        b"<< /Type /Pages /Count "
        + str(len(page_obj_nums)).encode("ascii")
        + b" /Kids ["
        + kids
        + b"] >>"
    )

    # 组装 PDF：头部、主体对象、交叉引用表、尾部。
    buf = io.BytesIO()
    buf.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = [0]  # 条目 0 是标准的"空闲"记录
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
        + str(catalog_obj_num).encode("ascii")
        + b" 0 R >>\nstartxref\n"
        + str(xref_pos).encode("ascii")
        + b"\n%%EOF\n"
    )
    return buf.getvalue()


@pytest.fixture
def tiny_pdf_path(tmp_path: Path) -> Path:
    """具有可预测可提取文本的两页合成 PDF。"""
    pdf_bytes = _make_tiny_pdf(
        [
            "carbon neutrality 2030 goal scope 1 emissions",
            "shareholder dividend policy quarterly distribution schedule",
        ]
    )
    out = tmp_path / "tiny.pdf"
    out.write_bytes(pdf_bytes)
    return out


# ---------------------------------------------------------------------
# 纯辅助函数测试
# ---------------------------------------------------------------------


class TestValidateCollectionName:
    def test_accepts_simple_alphanum(self) -> None:
        _validate_collection_name("esg2024")
        _validate_collection_name("user-library")
        _validate_collection_name("a_b.c")

    def test_rejects_too_short(self) -> None:
        with pytest.raises(ValueError, match="length"):
            _validate_collection_name("ab")

    def test_rejects_too_long(self) -> None:
        with pytest.raises(ValueError, match="length"):
            _validate_collection_name("x" * 64)

    def test_rejects_leading_punct(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            _validate_collection_name("-leading")
        with pytest.raises(ValueError, match="must match"):
            _validate_collection_name(".dotted")

    def test_rejects_double_dot(self) -> None:
        with pytest.raises(ValueError, match=r"\.\."):
            _validate_collection_name("foo..bar")

    def test_rejects_invalid_chars(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            _validate_collection_name("with space")
        with pytest.raises(ValueError, match="must match"):
            _validate_collection_name("中文集合")


class TestQualityClassifier:
    def test_high_when_top_score_clears_threshold(self) -> None:
        assert _classify_quality(QUALITY_HIGH_THRESHOLD + 0.05, 0.5, unique_sources=2) == "high"

    def test_medium_band(self) -> None:
        # top_score ≥ medium 但 < high；mean 仍然足够高。
        score = (QUALITY_HIGH_THRESHOLD + QUALITY_MEDIUM_THRESHOLD) / 2
        assert _classify_quality(score, score, unique_sources=2) == "medium"

    def test_low_when_top_score_below_medium(self) -> None:
        assert _classify_quality(QUALITY_MEDIUM_THRESHOLD - 0.05, 0.1, unique_sources=1) == "low"

    def test_low_when_no_unique_sources(self) -> None:
        # 我们将 ``unique_sources=0`` 视为"无可用证据" — 即使部分分数
        # 漂移上升，分类器也不得报告 'high'。
        assert _classify_quality(QUALITY_HIGH_THRESHOLD + 0.1, 0.5, unique_sources=0) in {
            "medium",
            "low",
        }


class TestBM25Index:
    def test_tokenizer_lowercases_and_splits(self) -> None:
        toks = _BM25Index._tokenize("Carbon Neutrality 2030 Goal!")
        assert toks == ["carbon", "neutrality", "2030", "goal"]

    def test_search_returns_relevant_doc_first(self) -> None:
        docs = [
            {"content": "carbon neutrality scope emissions", "metadata": {}},
            {"content": "shareholder dividend policy", "metadata": {}},
            {"content": "supply chain logistics warehouse", "metadata": {}},
        ]
        idx = _BM25Index(docs)
        ranked = idx.search("carbon emissions", top_k=3)
        assert ranked, "expected at least one BM25 hit"
        # 第一个文档共享两个查询词 — 必须排名第 1。
        first_idx, first_score = ranked[0]
        assert first_idx == 0, ranked
        assert first_score > 0

    def test_search_handles_empty_query(self) -> None:
        idx = _BM25Index([{"content": "hello world", "metadata": {}}])
        assert idx.search("   ", top_k=5) == []

    def test_handles_empty_corpus(self) -> None:
        # 空语料库是正常的冷启动场景（集合存在但尚未摄入文档）
        # — 不应抛出异常。
        idx = _BM25Index([])
        assert idx.search("anything", top_k=5) == []


class TestChunkPages:
    def test_chunks_inherit_page_metadata(self) -> None:
        pages = [
            {"page": 1, "text": "alpha beta " * 200},
            {"page": 2, "text": "gamma delta " * 50},
        ]
        chunks = _chunk_pages(pages, source="x.pdf", chunk_size=300, chunk_overlap=50)
        assert chunks
        assert all("page" in c["metadata"] for c in chunks)
        assert all(c["metadata"]["source"] == "x.pdf" for c in chunks)
        # 分块绝不跨页 — 来源追溯必须保持干净。
        page_to_chunks: dict[int, int] = {}
        for c in chunks:
            page_to_chunks[c["metadata"]["page"]] = page_to_chunks.get(c["metadata"]["page"], 0) + 1
        assert 1 in page_to_chunks and 2 in page_to_chunks


class TestHybridFuse:
    def test_doc_only_in_vector_list_still_included(self) -> None:
        vec_doc = {"content": "vector-only-doc", "metadata": {"source": "a", "page": 1}}
        bm_doc = {"content": "bm25-only-doc", "metadata": {"source": "b", "page": 1}}
        fused = _hybrid_fuse(
            vector_hits=[(vec_doc, 0.9)],
            bm25_hits=[(0, 5.0, bm_doc)],
        )
        contents = {r["content"] for r in fused}
        assert "vector-only-doc" in contents
        assert "bm25-only-doc" in contents

    def test_doc_in_both_lists_gets_summed_rrf(self) -> None:
        doc = {"content": "shared", "metadata": {"source": "x", "page": 1}}
        fused_only_vec = _hybrid_fuse(vector_hits=[(doc, 0.8)], bm25_hits=[])
        fused_both = _hybrid_fuse(vector_hits=[(doc, 0.8)], bm25_hits=[(0, 5.0, doc)])
        # 同一文档出现在两个列表中时，其 RRF 分数应高于仅出现在单个列表中的版本。
        assert fused_both[0]["rrf_score"] > fused_only_vec[0]["rrf_score"]

    def test_results_sorted_descending_by_rrf(self) -> None:
        docs = [{"content": f"doc{i}", "metadata": {"source": "s", "page": i}} for i in range(5)]
        # doc0 向量排名最高，doc4 最低。BM25 同理。
        vec = [(docs[i], 1.0 - i * 0.1) for i in range(5)]
        bm = [(i, 5.0 - i, docs[i]) for i in range(5)]
        fused = _hybrid_fuse(vec, bm)
        scores = [r["rrf_score"] for r in fused]
        assert scores == sorted(scores, reverse=True), fused


# ---------------------------------------------------------------------
# PDF 加载器（仍属纯辅助函数层，但使用真实 pypdf）
# ---------------------------------------------------------------------


class TestLoadPdfPages:
    def test_extracts_text_from_synthetic_pdf(self, tiny_pdf_path: Path) -> None:
        pages = _load_pdf_pages(tiny_pdf_path)
        assert len(pages) == 2
        assert all(isinstance(p["page"], int) for p in pages)
        # 第一页文本包含碳中和相关短语。
        joined = " ".join(p["text"] for p in pages)
        assert "carbon" in joined.lower()
        assert "dividend" in joined.lower()


# ---------------------------------------------------------------------
# 端到端摄入 + 搜索（慢速层）
# ---------------------------------------------------------------------


@pytest.mark.slow
class TestIngestAndSearch:
    """真实 FAISS + bge-small embedder 端到端测试。

    标记为 ``slow``，因为首次运行会下载约 95 MB 的模型权重并
    预热 sentence-transformer；后续运行在预热缓存上不到一秒。
    使用 ``pytest -m slow`` 运行。
    """

    @pytest.mark.asyncio
    async def test_ingest_then_search_returns_relevant_chunk(
        self, tiny_pdf_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 将持久化 + 缓存的单例重定向到干净的临时目录， 使此测试永远不会与用户的真实数据库冲突。
        # 注意：fastmcp 中的 ``@mcp.tool()`` 返回原始未包装的异步函数 — 我们可以直接 ``await`` 它而无需通过 stdio。
        monkeypatch.setattr(knowledge_server, "DEFAULT_DB_DIR", tmp_path / "kb")
        monkeypatch.setattr(faiss_store, "_FAISS_STORES", {})
        monkeypatch.setattr(knowledge_server, "_BM25_CACHE", {})

        ingest_result = await knowledge_server.ingest_pdf(
            local_path=str(tiny_pdf_path),
            collection="test-coll",
        )
        assert "error" not in ingest_result, ingest_result
        assert ingest_result["num_chunks_added"] >= 2

        # 碳中和查询必须将第 1 页的分块浮到顶部附近，而非第 2 页的股息分块。
        hits = await knowledge_server.search(
            query="carbon neutrality goal",
            collection="test-coll",
            top_k=3,
        )
        assert "error" not in hits, hits
        assert hits["top_k_returned"] >= 1
        assert hits["unique_sources"] == 1
        assert hits["results"], hits
        top_content = hits["results"][0]["content"].lower()
        assert "carbon" in top_content or "neutrality" in top_content

        # 在微小语料库中精确短语匹配的质量应至少分类为 medium。
        assert hits["quality"] in {"high", "medium"}, hits

    @pytest.mark.asyncio
    async def test_search_on_empty_collection_returns_low_quality(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(knowledge_server, "DEFAULT_DB_DIR", tmp_path / "kb")
        monkeypatch.setattr(faiss_store, "_FAISS_STORES", {})
        monkeypatch.setattr(knowledge_server, "_BM25_CACHE", {})

        result = await knowledge_server.search(
            query="anything",
            collection="empty-coll",
            top_k=5,
        )
        assert result["top_k_returned"] == 0
        assert result["quality"] == "low"
        assert result["results"] == []
        assert "warning" in result

    @pytest.mark.asyncio
    async def test_list_and_delete_collection(
        self, tiny_pdf_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(knowledge_server, "DEFAULT_DB_DIR", tmp_path / "kb")
        monkeypatch.setattr(faiss_store, "_FAISS_STORES", {})
        monkeypatch.setattr(knowledge_server, "_BM25_CACHE", {})

        await knowledge_server.ingest_pdf(local_path=str(tiny_pdf_path), collection="to-delete")

        listing = await knowledge_server.list_collections()
        names = {c["name"] for c in listing["collections"]}
        assert "to-delete" in names

        deleted = await knowledge_server.delete_collection(collection="to-delete")
        assert deleted["existed"] is True
        assert deleted["deleted"] is True

        # 幂等的第二次调用。
        deleted_again = await knowledge_server.delete_collection(collection="to-delete")
        assert deleted_again["existed"] is False
        assert deleted_again["deleted"] is False
