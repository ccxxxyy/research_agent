"""混合检索原语 — BM25 稀疏索引 + RRF 融合。

从 ``knowledge_server.py`` 中提取，使检索流水线可独立导入、测试。
``knowledge_server`` 仍然使用这些类，但调用方无需启动 MCP 服务器即可运行检索逻辑。

组件
----------
``BM25Index``
    对 ``rank_bm25.BM25Okapi`` 的轻量封装，负责对文档分词并将 BM25 分数映射回原始的 ``(content, metadata)`` 字典。
    BM25 稀疏索引：传统关键词匹配

``hybrid_rrf_fuse``
    稠密（向量）与稀疏（BM25）结果列表的加权 RRF 融合。返回按融合排序分数排列的统一记录列表，并按 ``(source, page, content[:80])`` 去重。"来自同一个文件、同一页、开头 80 个字符一样的，就认为是同一段话，合并处理。"
    RRF 融合：把向量检索结果和 BM25 结果合并成统一排序

FAISS 向量检索返回（vector_hits）
格式：[(文档字典, 余弦相似度分数), ...]
    vector_hits = [
    # 排名1：语义最相关
    (
        {"content": "2024年度，公司实现归属于母公司股东的净利润1.23亿元，同比增长15.2%。",
         "metadata": {"source": "annual_report_2024.pdf", "page": 12}},
        0.82  # 余弦相似度，0-1之间，越高越相关
    ),
    ...
    ]

BM25 关键词检索返回（bm25_raw → 转成 bm25_hits）
格式：[(文档在语料中的索引, BM25分数, 文档字典), ...]
    bm25_hits = [
    # 排名1：包含精确关键词 "归母净利润" + "2024"
    (
        42,   # 这个文档在语料库中的编号
        8.73, # BM25分数（无固定范围，越高越匹配）
        {"content": "2024年度，公司实现归属于母公司股东的净利润1.23亿元，同比增长15.2%。",
         "metadata": {"source": "annual_report_2024.pdf", "page": 12}}
    ),
    ...
    ]

送入 hybrid_rrf_fuse 后发：
annual_report_2024.pdf 第 12 页那段话同时被向量（排名1）和 BM25（排名1）命中了，融合后这一个"双重命中"的文档 RRF 分数会叠加，排在最前面：
    fused_results = [
    {
        "content": "2024年度，公司实现归属于母公司股东的净利润1.23亿元...",
        "metadata": {"source": "annual_report_2024.pdf", "page": 12},
        "vector_score": 0.82,
        "bm25_score": 8.73,
        "rrf_score": 0.6/(60+1) + 0.4/(60+1),  # = 0.01639 (两边都排名1)
        "vector_rank": 1,
        "bm25_rank": 1,
    },
    ...
    ]
"""

from __future__ import annotations

import re
from typing import Any


class BM25Index:
    """基于 ``{content, metadata}`` 字典列表的 BM25Okapi 索引。基于 rank_bm25.BM25Okapi 算法的封装。

    分词策略有意保持简单：转小写 + 按非单词字符拆分。
    CJK 中文字符作为单字符 token 保留，对于与文档共享名词短语的查询，BM25 可以正常处理。

    _SPLIT_RE：正则 r"\\W+"，意思是"一个或多个非单词字符"作为分隔符。例如 "Hello, world!" → ["hello", "world"]。

    __init__：
        接收 [{"content": "...", "metadata": {...}}, ...] 格式的文档列表
        对每个文档的 content 做分词
        如果文档列表为空，放入一个占位空文档（防止 BM25Okapi 初始化时报错）

    _tokenize：把文本转小写，然后按非单词字符切分成 token 列表。

    search：返回 [(文档在语料中的索引, BM25分数)]，按分数降序，最多返回 top_k 个。
    """

    _SPLIT_RE = re.compile(r"\W+", flags=re.UNICODE)

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        from rank_bm25 import BM25Okapi

        self.docs = docs
        self._is_empty: bool = not docs
        tokenized = [self._tokenize(d["content"]) for d in docs]
        if not tokenized:
            tokenized = [[""]]
            self.docs = [{"content": "", "metadata": {}}]
        self._bm25 = BM25Okapi(tokenized)

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        return [t for t in cls._SPLIT_RE.split(text.lower()) if t]

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """返回按分数降序排列的 ``[(corpus_index, bm25_score)]``。"""
        if self._is_empty:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


def _bm25_source_key(doc: dict[str, Any]) -> str:
    """分片键：PDF ``metadata.source``，缺失时归入 ``_unknown``。"""
    meta = doc.get("metadata") or {}
    src = meta.get("source")
    if src is None or str(src).strip() == "":
        return "_unknown"
    return str(src)


class BM25ShardedIndex:
    """按 ``metadata.source`` 管理的 BM25 索引，对外 API 与 :class:`BM25Index` 同形。

    * ``docs`` — 扁平文档列表（全局下标）
    * ``search(query, top_k)`` — 与单库 ``BM25Index`` 相同的全局 IDF 检索
    * ``add_docs`` / ``remove_source`` — 按 PDF source 增量改内存语料后重建 Okapi

    分片的价值在**生命周期**而非检索打分：ingest / 删单文档时只改内存中的
    ``docs`` 并重建 Okapi，无需重新遍历 FAISS docstore。检索始终用全局语料
    统计量，避免「每 PDF 一个小分片 → IDF 退化、分数全 0」的问题。

    当 ``len(docs) >= shard_search_threshold`` 且 source 数 ≥ 2 时，``search``
    改为各 source 分片检索再按分数合并（大库降低单次 ``get_scores`` 峰值）；
    小库仍走全局索引以保证质量。
    """

    _tokenize = BM25Index._tokenize
    # 低于此分块数时强制全局检索（小库 IDF 更稳）
    _SHARD_SEARCH_THRESHOLD = 200

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs: list[dict[str, Any]] = list(docs)
        self._is_empty: bool = not self.docs
        # source -> 该 source 在 docs 中的全局下标（供大库分片检索）
        self._shard_indices: dict[str, list[int]] = {}
        self._shard_indexes: dict[str, BM25Index] = {}
        self._flat: BM25Index = BM25Index([])
        self._reindex()

    def _reindex(self) -> None:
        self._is_empty = not self.docs
        self._flat = BM25Index(self.docs)
        by_source: dict[str, list[int]] = {}
        for i, doc in enumerate(self.docs):
            by_source.setdefault(_bm25_source_key(doc), []).append(i)
        self._shard_indices = by_source
        self._shard_indexes = {
            key: BM25Index([self.docs[i] for i in idxs]) for key, idxs in by_source.items()
        }

    @property
    def _shards(self) -> dict[str, BM25Index]:
        """测试与调试用：source → 分片索引。"""
        return self._shard_indexes

    def add_docs(self, new_docs: list[dict[str, Any]]) -> None:
        """追加文档并重建内存 BM25（不读 FAISS）。"""
        if not new_docs:
            return
        self.docs.extend(new_docs)
        self._reindex()

    def remove_source(self, source: str) -> int:
        """移除某 source 的全部分块并重建。返回移除条数。"""
        if not source:
            return 0
        key = source if source.strip() else "_unknown"
        before = len(self.docs)
        self.docs = [d for d in self.docs if _bm25_source_key(d) != key]
        removed = before - len(self.docs)
        if removed:
            self._reindex()
        else:
            self._is_empty = not self.docs
        return removed

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """返回 ``[(global_index, bm25_score)]``，与 :class:`BM25Index` 同形。"""
        if self._is_empty or top_k < 1:
            return []
        use_sharded = (
            len(self.docs) >= self._SHARD_SEARCH_THRESHOLD and len(self._shard_indexes) >= 2
        )
        if not use_sharded:
            return self._flat.search(query, top_k=top_k)

        merged: list[tuple[int, float]] = []
        for key, shard in self._shard_indexes.items():
            if shard._is_empty:
                continue
            offsets = self._shard_indices[key]
            for local_idx, score in shard.search(query, top_k=top_k):
                if 0 <= local_idx < len(offsets):
                    merged.append((offsets[local_idx], float(score)))
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:top_k]


def hybrid_rrf_fuse(
    vector_hits: list[tuple[dict[str, Any], float]],
    bm25_hits: list[tuple[int, float, dict[str, Any]]],
    *,
    k_rrf: int = 60,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list[dict[str, Any]]:
    """向量 + BM25 结果的加权 RRF（Reciprocal Rank Fusion）融合。

    按 ``(source, page, content[:80])`` 去重后，每个唯一文档返回一条记录，包含以下字段：

    * ``content``、``metadata`` — 原始文档内容和元数据
    * ``vector_score`` — 原始余弦相似度（归一化后 [0, 1]）
    * ``bm25_score`` — 原始 BM25 分数（无上界，与模型相关）
    * ``rrf_score`` — 融合排序分数（排序键）
    * ``vector_rank``、``bm25_rank`` — 在各自列表中的原始的排名（从 1 开始）


    参数：
        vector_hits：向量检索结果，格式 [(文档字典, 余弦相似度分数), ...]
        bm25_hits：BM25 检索结果，格式 [(语料索引, BM25分数, 文档字典), ...]
        k_rrf=60：RRF 公式中的常数 k（经典默认值 60）
        vector_weight=0.6：向量结果的权重（6:4 偏向向量）
        bm25_weight=0.4：BM25 结果的权重RRF 公式：对每个文档，其融合分数 = weight / (k + rank)

    RRF 公式核心：
    单个文档的 rrf_score = Σ (weight / (k + rank))

    举例：如果一个文档在向量结果中排第 2，在 BM25 中排第 5：
    rrf_score = 0.6/(60+2) + 0.4/(60+5) = 0.00968 + 0.00615 = 0.01583
    去重策略（_key 函数）：按 (来源文件名, 页码, content前80字符) 生成唯一键。同一个文档在向量和 BM25 中都命中时，只保留一条记录，分数累加。

    这个函数把"语义相似的结果"和"关键词匹配的结果"合并成一个统一的排名。两边都出现的文档分数会叠加（更可信），只出现在一边的也不会被丢掉。最终按融合分数从高到低排序。
    """
    fused: dict[str, dict[str, Any]] = {}

    def _key(meta: dict[str, Any], content: str) -> str:
        return f"{meta.get('source', '')}|p={meta.get('page', '?')}|{content[:80]}"

    for rank, (doc, score) in enumerate(vector_hits, start=1):
        k = _key(doc["metadata"], doc["content"])
        rec = fused.setdefault(
            k,
            {
                "content": doc["content"],
                "metadata": doc["metadata"],
                "vector_score": score,
                "bm25_score": 0.0,
                "rrf_score": 0.0,
                "vector_rank": rank,
                "bm25_rank": None,
            },
        )
        rec["vector_score"] = max(rec["vector_score"], score)
        rec["rrf_score"] += vector_weight / (k_rrf + rank)

    for rank, (_, score, doc) in enumerate(bm25_hits, start=1):
        k = _key(doc["metadata"], doc["content"])
        rec = fused.setdefault(
            k,
            {
                "content": doc["content"],
                "metadata": doc["metadata"],
                "vector_score": 0.0,
                "bm25_score": score,
                "rrf_score": 0.0,
                "vector_rank": None,
                "bm25_rank": rank,
            },
        )
        rec["bm25_score"] = max(rec["bm25_score"], score)
        rec["bm25_rank"] = rank if rec["bm25_rank"] is None else min(rec["bm25_rank"], rank)
        rec["rrf_score"] += bm25_weight / (k_rrf + rank)

    return sorted(fused.values(), key=lambda r: r["rrf_score"], reverse=True)


__all__ = ["BM25Index", "BM25ShardedIndex", "hybrid_rrf_fuse"]
