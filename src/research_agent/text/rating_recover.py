"""从工具消息回收研报评级，修正终稿「研报评级未取得」误称。"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

_RATING_MISSING_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\|?\s*研报评级\s*\|?\s*未取得\s*\|?", re.I),
    re.compile(r"研报评级\s*[：:|\t]\s*未取得", re.I),
    re.compile(r"研报评级\s*未取得", re.I),
    re.compile(r"机构评级\s*[：:|\t]\s*未取得", re.I),
    re.compile(r"(?<![买卖增持减持])评级\s*[：:|\t]\s*未取得", re.I),
)


def _walk_for_ratings(obj: Any, *, depth: int = 0) -> dict[str, Any] | None:
    if depth > 6 or obj is None:
        return None
    if isinstance(obj, dict):
        summary = obj.get("analyst_summary")
        if (
            isinstance(summary, dict)
            and summary.get("used")
            and (summary.get("ratings_sample") or summary.get("institutions_sample"))
        ):
            return summary
        samples = obj.get("ratings_sample")
        if isinstance(samples, list) and samples:
            return {
                "used": True,
                "ratings_sample": samples[:8],
                "institutions_sample": (obj.get("institutions_sample") or [])[:6],
                "count": obj.get("count") or len(samples),
            }
        aux = obj.get("aux_signals")
        if isinstance(aux, dict):
            an = aux.get("analyst")
            if isinstance(an, dict) and an.get("used") and an.get("ratings_sample"):
                return {
                    "used": True,
                    "ratings_sample": list(an.get("ratings_sample") or [])[:8],
                    "institutions_sample": [
                        {
                            "institution": r.get("institution"),
                            "rating": r.get("rating"),
                            "date": str(r.get("date") or "")[:10],
                        }
                        for r in (an.get("reports") or [])[:6]
                        if isinstance(r, dict)
                    ],
                    "count": an.get("count"),
                    "source_url": an.get("source_url"),
                }
        for v in obj.values():
            hit = _walk_for_ratings(v, depth=depth + 1)
            if hit:
                return hit
    elif isinstance(obj, list):
        for item in obj[:20]:
            hit = _walk_for_ratings(item, depth=depth + 1)
            if hit:
                return hit
    elif isinstance(obj, str):
        text = obj.strip()
        if not text or ("ratings_sample" not in text and "analyst_summary" not in text):
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # 工具结果偶发前后夹杂说明文字
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                return None
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return _walk_for_ratings(parsed, depth=depth + 1)
    return None


_DIGEST_RATING_RE = re.compile(
    r"研报评级[：:]\s*(?!本旁路未取得|未取得|本轮旁路未取得)([^\n。；;]+)",
    re.I,
)


def _ratings_from_prose(text: str) -> dict[str, Any] | None:
    """从专家/主管正文里的「研报评级：东吴证券买入…」回收（last_message 模式无 ToolMessage）。"""
    if not text or "研报评级" not in text:
        return None
    m = _DIGEST_RATING_RE.search(text)
    if not m:
        return None
    blob = m.group(1).strip()
    if not blob or "未取得" in blob:
        return None
    parts = [p.strip() for p in re.split(r"[、,，/]", blob) if p.strip()]
    institutions: list[dict[str, str]] = []
    samples: list[str] = []
    for p in parts[:6]:
        rm = re.search(r"(买入|增持|持有|中性|减持|卖出)$", p)
        if rm:
            rating = rm.group(1)
            inst = p[: rm.start()].strip()
            samples.append(rating)
            institutions.append({"institution": inst, "rating": rating})
        else:
            samples.append(p)
    if not samples and not institutions:
        return None
    return {
        "used": True,
        "ratings_sample": samples,
        "institutions_sample": institutions,
        "count": len(samples) or len(institutions),
    }


def extract_ratings_from_messages(messages: list[BaseMessage] | None) -> dict[str, Any] | None:
    """从 ToolMessage / 专家 AIMessage 中提取可用评级摘要。"""
    if not messages:
        return None
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            hit = _walk_for_ratings(msg.content)
            if hit:
                return hit
            art = getattr(msg, "artifact", None)
            if art is not None:
                hit = _walk_for_ratings(art)
                if hit:
                    return hit
        elif isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            content = str(msg.content or "")
            hit = _walk_for_ratings(content)
            if hit:
                return hit
            hit = _ratings_from_prose(content)
            if hit:
                return hit
    return None


def format_ratings_line(summary: dict[str, Any]) -> str:
    """把评级摘要收成一行中文。"""
    institutions = summary.get("institutions_sample") or []
    bits: list[str] = []
    for row in institutions[:4]:
        if not isinstance(row, dict):
            continue
        inst = str(row.get("institution") or "").strip()
        rating = str(row.get("rating") or "").strip()
        if inst and rating:
            bits.append(f"{inst}{rating}")
        elif rating:
            bits.append(rating)
    samples = [str(x) for x in (summary.get("ratings_sample") or []) if x][:6]
    if bits:
        return "、".join(bits)
    if samples:
        return "、".join(samples)
    return "有评级样本"


def text_claims_ratings_missing(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _RATING_MISSING_RES)


def recover_ratings_in_text(text: str, summary: dict[str, Any] | None) -> str:
    """若正文声称研报评级未取得，但工具里有评级，则替换为真实样本。"""
    if not text or not summary or not summary.get("used"):
        return text
    if not text_claims_ratings_missing(text) and "aux_signals.analyst" not in text:
        return text
    line = format_ratings_line(summary)
    if not line:
        return text
    out = text
    for pat in _RATING_MISSING_RES:
        out = pat.sub(f"研报评级：{line}", out)
    return out
