"""从工具返回载荷中提取真实 ``source`` / ``source_url``（供 SSE 与 UI 标签使用）。"""

from __future__ import annotations

import json
from typing import Any


def coerce_tool_payload(content: Any) -> dict[str, Any] | None:
    """把 ToolMessage.content 尽量解析成顶层 dict。"""
    if content is None:
        return None
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        # MCP / LangChain 内容块：优先取 text JSON
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if isinstance(text, str) and text.strip():
                    parsed = coerce_tool_payload(text)
                    if parsed is not None:
                        return parsed
                if "source" in part or "source_url" in part:
                    return part
            elif isinstance(part, str) and part.strip():
                parsed = coerce_tool_payload(part)
                if parsed is not None:
                    return parsed
        return None
    if not isinstance(content, str):
        content = str(content)
    text = content.strip()
    if not text:
        return None
    # 少数适配器会包一层 markdown code fence
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        return data
    return None


def extract_tool_result_source(content: Any) -> tuple[str | None, str | None]:
    """从工具结果提取 ``(source, source_url)``；缺失则为 ``(None, None)``。"""
    payload = coerce_tool_payload(content)
    if not payload:
        return None, None
    if payload.get("error") and not payload.get("source"):
        return None, None
    source = payload.get("source")
    source_url = payload.get("source_url")
    source_s = str(source).strip() if source not in (None, "") else None
    url_s = str(source_url).strip() if source_url not in (None, "") else None
    return source_s, url_s


__all__ = ["coerce_tool_payload", "extract_tool_result_source"]
