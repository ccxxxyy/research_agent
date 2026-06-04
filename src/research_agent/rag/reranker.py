"""cross-encoder 重排序器 — 本地运行、快速、无需 API 调用。

为什么选择 cross-encoder 而非 LLM？
------------------------------------
本模块的第一版使用 ``LLMReranker``，通过 JSON 格式让 LLM 对每个 (query, document) 对打分。虽然可行，但存在三个问题：

* 延迟 — 每个文档一次 LLM 往返。即使使用快速的 LIGHT 级别模型，``top_k=5`` 时每次搜索也需要 3-10 秒，在 corrective-RAG 重试循环中会累积放大。
* 成本 — 每次搜索消耗 5+ 个 LIGHT 级别的 prompt 调用，按次计费。
* 漂移 — 分数 JSON 从聊天补全中解析；模型有时返回散文、``score: "high"``、或带单位的数字字符串，解析器不得不逐一兜底处理。

一个小型 cross-encoder（此处为 ``BAAI/bge-reranker-base``，约 280 MB）可同时解决以上三个问题：本地 CPU 运行约 50 ms / 对，无按次成本，且返回真实值 logit。
用于初始 FAISS 检索的 bi-encoder 嵌入和用于重排序的 cross-encoder 在设计上互补 — bi-encoder 足够快以评分数千候选项，cross-encoder 足够准确以重排顶部少量结果。

调用位置
--------
唯一的生产调用方为``research_agent.mcp_servers.knowledge_server._search()``。
该函数从 RRF 过量检索（通常为 ``top_k * 3``），将候选项送入此处，并将重排序输出裁剪至 ``top_k``。
每个结果上附加 ``rerank_score``，以便下游 Agent（特别是 ``knowledge_expert`` 的 corrective-RAG 循环）可以在 bi-encoder 向量分数旁查看 cross-encoder 的判定。
假设 FAISS + BM25 融合后返回了 3 个候选文档，经过 cross-encoder 重排序后（列表按rerank_score分数从大到小排序）：
[
    {
        "content": "2024年归母净利润为1.23亿元，同比增长15%...",
        "metadata": {"source": "annual_report.pdf", "page": 12},
        "vector_score": 0.72,
        "bm25_score": 3.1,
        "rrf_score": 0.0158,
        "rerank_score": 2.8734    # ← cross-encoder 给的分数（越高越相关）
    },
    ...
]

可选 / 容错
------------
重排序由环境变量 ``KNOWLEDGE_RERANKER_ENABLED``（默认 ``"1"``）控制。
当禁用或模型在主机上加载失败时 — 消费方回退到 RRF 顺序，``search()`` 绝不会因重排序失败而抛出异常；最差情况是"使用 RRF 顺序并在日志中输出一行警告"。
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from loguru import logger

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_MODELSCOPE_CACHE_PATH = os.path.expanduser("~/.cache/modelscope/hub/models/BAAI/bge-reranker-base")
_HF_CACHE_ROOT = os.path.expanduser("~/.cache/huggingface/hub")


def _resolve_default_model() -> str:
    """在导入时选择最佳可用模型路径。

    优先级：
      1. 显式设置的 ``KNOWLEDGE_RERANKER_MODEL`` 环境变量。
      2. ModelScope 本地缓存。
      3. HuggingFace 本地缓存 snapshot。
      4. HuggingFace 模型 ID（需联网）。
    """
    explicit = os.environ.get("KNOWLEDGE_RERANKER_MODEL", "").strip()
    if explicit:
        return explicit
    if os.path.isfile(os.path.join(_MODELSCOPE_CACHE_PATH, "model.safetensors")):
        return _MODELSCOPE_CACHE_PATH
    hf_snap = os.path.join(_HF_CACHE_ROOT, "models--BAAI--bge-reranker-base", "snapshots")
    if os.path.isdir(hf_snap):
        snaps = sorted(
            os.listdir(hf_snap),
            key=lambda d: os.path.getmtime(os.path.join(hf_snap, d)),
            reverse=True,
        )
        if snaps:
            local = os.path.join(hf_snap, snaps[0])
            logger.info("Using local cached reranker: {}", local)
            return local
    return "BAAI/bge-reranker-base"


DEFAULT_RERANKER_MODEL = _resolve_default_model()
"""默认 cross-encoder 模型路径。

在导入时通过 :func:`_resolve_default_model` 解析：

* 若设置了 ``KNOWLEDGE_RERANKER_MODEL``，以其为准。
* 否则，若 ``~/.cache/modelscope/hub/models/BAAI/bge-reranker-base/model.safetensors`` 存在（通过可选的``modelscope`` extra 预下载），使用本地路径 — 零网络 IO。
* 否则回退到 HuggingFace 模型 ID ``BAAI/bge-reranker-base``（需要 HF Hub 访问或 ``~/.cache/huggingface`` 缓存命中）。

``BAAI/bge-reranker-base`` 为中英双语模型，磁盘占用约 1 GB，在 CPU 上每对约 50 ms。

这个常量在 import 时就被确定了（调用上面的 _resolve_default_model()）。是中英双语模型，磁盘约 1GB，CPU 每对约 50ms。
"""

# 模块级缓存：加载 CrossEncoder 会启动分词器并将数百 MB 权重拉入内存。在进程生命周期内保留一个实例，并在所有搜索调用中复用。
_CROSS_ENCODER: Any | None = None


def _get_cross_encoder(model_name: str | None = None) -> Any:
    """返回单例 CrossEncoder，首次调用时构建。

    ``sentence_transformers`` 的导入被延迟，以便从不请求重排序的进程（如不相关辅助功能的单元测试）无需承担约 1 秒的导入开销。
    首次调用还需要加载模型权重 — 缓存热时约 3 秒，冷启动更长。

    全局单例模式。模型权重几百 MB，加载一次后在整个进程生命周期内复用。
    sentence_transformers 库的 import 也是延迟的（lazy import），避免不需要重排序的代码路径承担约 1 秒的导入开销。
    """
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        from sentence_transformers import CrossEncoder

        name = model_name or DEFAULT_RERANKER_MODEL
        logger.info("Loading cross-encoder reranker: {}", name)
        _CROSS_ENCODER = CrossEncoder(name, device="cpu")
    return _CROSS_ENCODER


class CrossEncoderReranker:
    """本地 cross-encoder 重排序器。

    操作 :mod:`research_agent.mcp_servers.knowledge_server` 所使用的字典结构，因此集成是即插即用的：输入一组字典，每个字典需要一个 ``"content"``键（分块文本）；
    其余字段原样保留，并添加一个新的 ``"rerank_score"``字段。输出为同一列表，按 ``rerank_score`` 降序排列。

    输出：同样的字典列表，按 cross-encoder 分数降序排列，每个字典多了一个 "rerank_score" 字段

    Args:
        model_name: HuggingFace 模型 ID。``None``（默认）使用:data:`DEFAULT_RERANKER_MODEL`，其本身遵循``KNOWLEDGE_RERANKER_MODEL`` 环境变量。
        max_pairs: 每次调用发送给 cross-encoder 的 (query, document)对数硬上限。防止意外输入 1000 个候选项导致 CPU 满载。
            超过 ``max_pairs`` 的部分将保持未排序状态置于输出尾部。

    Example::

        reranker = CrossEncoderReranker()
        ranked = await reranker.rerank(
            query="2030 carbon neutrality",
            documents=fused_rrf_hits,  # list[dict] with "content"
        )
        top_5 = ranked[:5]
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        max_pairs: int = 64,
    ) -> None:
        self._model_name = model_name
        # 限制每次调用的开销，防止不小心传入 1000 个候选把 CPU 打满。64 对在 base reranker + CPU 上约 3 秒
        # — 已远超任何合理的 top_k 值。
        self._max_pairs = max(1, int(max_pairs))

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """对 ``documents`` 相对于 ``query`` 重新打分并降序排列。

        简单情况（0 或 1 个文档）会短路 直接返回而不加载模型 — 对离线测试路径和候选列表为空的冷启动场景很有用（适合测试和空结果场景）。

        此函数永不抛出异常：模型层的任何异常都会被捕获，返回输入顺序原始顺序并输出警告日志，以便调用方可以保持正常流程。

        正常流程：取前 max_pairs 个送模型打分，超出的部分保留在尾部不排序
        """
        if not documents:
            return []
        if len(documents) == 1:
            documents[0].setdefault("rerank_score", None)
            return documents

        head = documents[: self._max_pairs]
        tail = documents[self._max_pairs :]

        try:
            scores = await asyncio.to_thread(self._predict, query, head)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CrossEncoderReranker failed ({}); returning input order",
                exc,
            )
            for d in documents:
                d.setdefault("rerank_score", None)
            return documents

        for doc, score in zip(head, scores, strict=False):
            doc["rerank_score"] = round(float(score), 4)

        # 超过 max_pairs 上限的条目从未经过模型。保留它们，但由于分数为 None，它们排在所有已重排序条目之后。，遇到 None 就当作 -∞（负无穷大。
        # 使用哨兵浮点值进行排序，：没打分的文档自动排到最后面，float("-inf") 就是那个"哨兵值"，它不是真实分数，只是为了让排序能正常工作而设置的一个占位符，以避免 None 导致比较崩溃。
        for d in tail:
            d.setdefault("rerank_score", None)

        def _sort_key(d: dict[str, Any]) -> float:
            score = d.get("rerank_score")
            return float(score) if score is not None else float("-inf")

        return sorted(documents, key=_sort_key, reverse=True)

    def _predict(self, query: str, documents: list[dict[str, Any]]) -> list[float]:
        """对 (query, content) 组成配对，送入运行 cross-encoder。

        抽取为同步方法，以便异步的 ``rerank`` 可以通过``asyncio.to_thread`` 调用，而不会将 torch 导入细节泄漏到公共接口。
        """
        model = _get_cross_encoder(self._model_name)
        pairs = [(query, doc.get("content", "") or "") for doc in documents]
        raw = model.predict(pairs)
        # ``CrossEncoder.predict`` 对批量输入返回 numpy ndarray。CrossEncoder.predict(pairs) 是调用模型给每对 (查询, 文档) 打分，返回一个 numpy 数组，包含每对的分数。比如传入 5 对，它返回 array([2.87, 1.20, -0.45, 0.33, 1.55])
        # 转换为普通 float 列表，以免下游调用方被迫导入 numpy。把 array([2.87, 1.20, -0.45]) 转成普通 Python 列表 [2.87, 1.20, -0.45]
        return [float(s) for s in raw]


__all__ = [
    "CrossEncoderReranker",
    "DEFAULT_RERANKER_MODEL",
]
