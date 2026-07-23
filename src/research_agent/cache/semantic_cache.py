"""静态金融知识的分层语义缓存。

缓存什么
--------
固定口径、几乎不随时间变化的知识，避免重复浪费 LLM 调用：

* ``glossary`` — 金融术语（ROE / 市盈率 / 北向资金…）
* ``methodology`` — 计算口径与公式解读
* ``template`` — 固定解读框架 / 报告提纲
* ``faq`` — 常见问题（交易时间、涨跌停规则…）
* ``macro`` — 宏观指标「是什么、怎么读」
* ``historical_event`` — 历史事件定义（不回答「对今天行情的影响」）

不缓存什么
----------
含 A 股代码、或带时效词 + 行情/新闻语义的问题（走工具 TTL 缓存 +supervisor 实时研究）。终答默认不入本缓存。

分层
----
* **L0 精确键** — 规范化问题哈希，完全一致才命中。
* **L1 语义** — FAISS 向量相似 + 元数据维度过滤（``cache_domain`` / ``version`` / ``locale`` / ``prompt_version``）。

持久化
------
``./data/semantic_cache/static_knowledge/``（LangChain FAISS 文件对）。
种子数据：``research_agent/cache/seed/static_knowledge.json``。
首次 ``lookup`` / ``ensure_ready`` 时若索引缺失或版本漂移则重建。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------
CACHE_DOMAINS = frozenset(
    {
        "glossary",
        "methodology",
        "template",
        "faq",
        "macro",
        "historical_event",
    }
)

DEFAULT_DB_DIR = Path("./data/semantic_cache").resolve()
COLLECTION_NAME = "static_knowledge"
SEED_PATH = Path(__file__).resolve().parent / "seed" / "static_knowledge.json"

DEFAULT_SIMILARITY_THRESHOLD = 0.82
"""归一化向量余弦相似度阈值（bge-small-zh + normalize_embeddings）。"""

DEFAULT_TOP_K = 4

# A 股 6 位代码 → 一律视为实时/标的研究，跳过语义缓存
_A_SHARE_CODE = re.compile(r"(?<!\d)\d{6}(?!\d)")

# 时效词 + 动态数据语义 → 跳过
_DYNAMIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(最新|今天|今日|现在|此刻|实时|盘中|刚才|刚刚|本周|本月|昨天|上周)"
        r".{0,8}(行情|股价|涨跌|新闻|快讯|资金流|龙虎榜|舆情|成交|净值)"
    ),
    re.compile(r"(帮我|给我|请|麻烦).{0,6}(分析|看看|研究|点评)(?!框架|模板|提纲)"),
    re.compile(r"(值不值得|能不能买|该不该买|要不要买)"),
    re.compile(r"(近\s*\d+\s*[日天周月年]|过去\s*\d+\s*[日天周月年]).{0,4}(涨|跌|行情|表现|收益)"),
)


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def normalize_query(text: str) -> str:
    """L0 键规范化：去空白与常见标点、小写。"""
    q = text.strip().lower()
    q = re.sub(r"\s+", "", q)
    q = re.sub(r"[？?！!。．\.，,、；;：:\s\"'“”‘’（）()【】\[\]]+", "", q)
    return q


def is_cacheable_query(query: str) -> bool:
    """是否允许进入语义缓存（不含标的 / 时效动态语义）。"""
    q = query.strip()
    if len(q) < 2 or len(q) > 500:
        return False
    if _A_SHARE_CODE.search(q):
        return False
    return all(not pat.search(q) for pat in _DYNAMIC_PATTERNS)


@dataclass(frozen=True)
class SemanticHit:
    """一次语义/精确命中。"""

    answer: str
    cache_domain: str
    score: float
    matched_question: str
    version: str
    locale: str
    exact: bool
    prompt_version: str = "v1"


@dataclass
class _SeedEntry:
    cache_domain: str
    question: str
    answer: str
    aliases: list[str] = field(default_factory=list)
    version: str = "1"
    locale: str = "zh-CN"
    prompt_version: str = "v1"


class SemanticKnowledgeCache:
    """静态知识 FAISS 语义缓存。"""

    def __init__(
        self,
        *,
        db_dir: Path | None = None,
        seed_path: Path | None = None,
        similarity_threshold: float | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.db_dir = (db_dir or DEFAULT_DB_DIR).resolve()
        self.seed_path = seed_path or SEED_PATH
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else _env_float("SEMANTIC_CACHE_THRESHOLD", DEFAULT_SIMILARITY_THRESHOLD)
        )
        self.enabled = (
            enabled if enabled is not None else _env_flag("SEMANTIC_CACHE_ENABLED", default=True)
        )
        self._lock = threading.RLock()
        self._ready = False
        self._exact_index: dict[str, SemanticHit] = {}
        self._store: Any | None = None
        self._seed_meta: dict[str, str] = {}
        self.hits = 0
        self.misses = 0
        self.skips = 0

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def ensure_ready(self) -> None:
        """确保索引已加载；缺失或版本漂移时从种子重建。"""
        if not self.enabled:
            return
        with self._lock:
            if self._ready:
                return
            self._load_or_rebuild()
            self._ready = True

    def lookup(self, query: str) -> SemanticHit | None:
        """对用户问题做 L0→L1 查找；不可缓存或未命中返回 ``None``。"""
        if not self.enabled:
            self.skips += 1
            return None
        if not is_cacheable_query(query):
            self.skips += 1
            return None

        self.ensure_ready()

        # L0 精确
        key = normalize_query(query)
        hit = self._exact_index.get(key)
        if hit is not None:
            self.hits += 1
            logger.info(
                "semantic_cache L0 命中 domain={} q={!r}",
                hit.cache_domain,
                hit.matched_question,
            )
            return hit

        # L1 语义
        hit = self._semantic_search(query)
        if hit is not None:
            self.hits += 1
            logger.info(
                "semantic_cache L1 命中 domain={} score={:.3f} matched={!r}",
                hit.cache_domain,
                hit.score,
                hit.matched_question,
            )
            return hit

        self.misses += 1
        return None

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ready": self._ready,
            "hits": self.hits,
            "misses": self.misses,
            "skips": self.skips,
            "exact_keys": len(self._exact_index),
            "threshold": self.similarity_threshold,
            "seed_version": self._seed_meta.get("version", ""),
        }

    def reset_for_tests(self) -> None:
        """测试用：清空内存状态（不删磁盘）。"""
        with self._lock:
            self._ready = False
            self._exact_index.clear()
            self._store = None
            self.hits = self.misses = self.skips = 0

    # ------------------------------------------------------------------
    # 内部：加载 / 重建
    # ------------------------------------------------------------------
    def _marker_path(self) -> Path:
        return self.db_dir / COLLECTION_NAME / "seed_meta.json"

    def _load_seed(self) -> list[_SeedEntry]:
        raw = json.loads(self.seed_path.read_text(encoding="utf-8"))
        self._seed_meta = {
            "version": str(raw.get("version", "1")),
            "locale": str(raw.get("locale", "zh-CN")),
            "prompt_version": str(raw.get("prompt_version", "v1")),
        }
        entries: list[_SeedEntry] = []
        for item in raw.get("entries", []):
            domain = str(item["cache_domain"])
            if domain not in CACHE_DOMAINS:
                logger.warning("semantic_cache 跳过未知 domain={}", domain)
                continue
            entries.append(
                _SeedEntry(
                    cache_domain=domain,
                    question=str(item["question"]).strip(),
                    answer=str(item["answer"]).strip(),
                    aliases=[str(a).strip() for a in item.get("aliases", []) if str(a).strip()],
                    version=str(item.get("version", self._seed_meta["version"])),
                    locale=str(item.get("locale", self._seed_meta["locale"])),
                    prompt_version=str(
                        item.get("prompt_version", self._seed_meta["prompt_version"])
                    ),
                )
            )
        return entries

    def _seed_fingerprint(self, entries: list[_SeedEntry]) -> str:
        payload = {
            "meta": self._seed_meta,
            "entries": [
                {
                    "d": e.cache_domain,
                    "q": e.question,
                    "a": e.answer,
                    "aliases": e.aliases,
                    "v": e.version,
                    "loc": e.locale,
                    "pv": e.prompt_version,
                }
                for e in entries
            ],
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def _needs_rebuild(self, fingerprint: str) -> bool:
        from research_agent.rag import faiss_store as _faiss

        if not _faiss.index_exists(COLLECTION_NAME, db_dir=self.db_dir):
            return True
        marker = self._marker_path()
        if not marker.is_file():
            return True
        try:
            meta = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True
        return meta.get("fingerprint") != fingerprint

    def _build_exact_index(self, entries: list[_SeedEntry]) -> None:
        index: dict[str, SemanticHit] = {}
        for e in entries:
            hit = SemanticHit(
                answer=e.answer,
                cache_domain=e.cache_domain,
                score=1.0,
                matched_question=e.question,
                version=e.version,
                locale=e.locale,
                exact=True,
                prompt_version=e.prompt_version,
            )
            for text in [e.question, *e.aliases]:
                key = normalize_query(text)
                if key:
                    index[key] = hit
        self._exact_index = index

    def _rebuild(self, entries: list[_SeedEntry], fingerprint: str) -> None:
        from research_agent.rag import faiss_store as _faiss

        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for e in entries:
            for text in [e.question, *e.aliases]:
                texts.append(text)
                metadatas.append(
                    {
                        "cache_domain": e.cache_domain,
                        "answer": e.answer,
                        "canonical_question": e.question,
                        "version": e.version,
                        "locale": e.locale,
                        "prompt_version": e.prompt_version,
                    }
                )

        logger.info(
            "semantic_cache 重建索引 entries={} vectors={} dir={}",
            len(entries),
            len(texts),
            self.db_dir / COLLECTION_NAME,
        )
        _faiss.invalidate_cache(COLLECTION_NAME)
        self._store = _faiss.create_from_texts(
            COLLECTION_NAME,
            texts,
            metadatas,
            db_dir=self.db_dir,
        )
        marker = self._marker_path()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "version": self._seed_meta.get("version", "1"),
                    "locale": self._seed_meta.get("locale", "zh-CN"),
                    "prompt_version": self._seed_meta.get("prompt_version", "v1"),
                    "entry_count": len(entries),
                    "vector_count": len(texts),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load_or_rebuild(self) -> None:
        entries = self._load_seed()
        if not entries:
            logger.warning("semantic_cache 种子为空，缓存不可用")
            self._exact_index = {}
            self._store = None
            return

        fingerprint = self._seed_fingerprint(entries)
        self._build_exact_index(entries)

        if self._needs_rebuild(fingerprint):
            self._rebuild(entries, fingerprint)
        else:
            from research_agent.rag import faiss_store as _faiss

            self._store = _faiss.load_store(COLLECTION_NAME, db_dir=self.db_dir)
            logger.info(
                "semantic_cache 已加载磁盘索引 exact_keys={} threshold={}",
                len(self._exact_index),
                self.similarity_threshold,
            )

    def _semantic_search(self, query: str) -> SemanticHit | None:
        if self._store is None:
            return None

        expected_version = self._seed_meta.get("version", "1")
        expected_locale = self._seed_meta.get("locale", "zh-CN")
        expected_pv = self._seed_meta.get("prompt_version", "v1")

        try:
            # LangChain FAISS：score 越小越相似（L2）；归一化后可用
            # similarity_search_with_relevance_scores → 越高越好（0~1）
            pairs = self._store.similarity_search_with_relevance_scores(
                query,
                k=DEFAULT_TOP_K,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("semantic_cache 向量检索失败: {}", exc)
            return None

        best: SemanticHit | None = None
        for doc, score in pairs:
            meta = doc.metadata or {}
            domain = str(meta.get("cache_domain", ""))
            if domain not in CACHE_DOMAINS:
                continue
            if str(meta.get("version", "")) != expected_version:
                continue
            if str(meta.get("locale", "")) != expected_locale:
                continue
            if str(meta.get("prompt_version", "")) != expected_pv:
                continue
            if float(score) < self.similarity_threshold:
                continue
            candidate = SemanticHit(
                answer=str(meta.get("answer", "")),
                cache_domain=domain,
                score=float(score),
                matched_question=str(meta.get("canonical_question", doc.page_content)),
                version=str(meta.get("version", expected_version)),
                locale=str(meta.get("locale", expected_locale)),
                exact=False,
                prompt_version=str(meta.get("prompt_version", expected_pv)),
            )
            if not candidate.answer:
                continue
            if best is None or candidate.score > best.score:
                best = candidate
        return best


_cache: SemanticKnowledgeCache | None = None
_cache_lock = threading.Lock()


def get_semantic_cache() -> SemanticKnowledgeCache:
    """进程内惰性单例。"""
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = SemanticKnowledgeCache()
    return _cache


def reset_semantic_cache_for_tests() -> None:
    """丢弃单例（测试用）。"""
    global _cache
    with _cache_lock:
        if _cache is not None:
            _cache.reset_for_tests()
        _cache = None
