"""将 ``ConfidenceValidator`` 接到 supervisor 终稿消息上。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from research_agent.agents.confidence import (
    ConfidenceValidator,
    Recommendation,
)
from research_agent.text.gap_sanitize import apply_confidence_footer, sanitize_data_gaps
from research_agent.text.rating_recover import (
    extract_ratings_from_messages,
    recover_ratings_in_text,
)

_validator = ConfidenceValidator()


def _last_synthesis_ai(messages: list[BaseMessage]) -> AIMessage | None:
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if getattr(msg, "tool_calls", None):
            continue
        content = msg.content
        if isinstance(content, str) and content.strip():
            return msg
    return None


def _original_query(messages: list[BaseMessage]) -> str:
    for msg in messages:
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            # 跳过 reviewer feedback
            if msg.content.startswith("[REVIEWER FEEDBACK]"):
                continue
            return msg.content
    return ""


def apply_confidence_gate_to_messages(
    messages: list[BaseMessage],
    *,
    validator: ConfidenceValidator | None = None,
) -> list[BaseMessage]:
    """对主管综合稿做规则置信度校验，写回 ``additional_kwargs`` 并追加脚注。

    - ``accept``：仅写入 metadata
    - ``downweight`` / 有 flags：追加「可信度提示」
    - ``reject``：追加强提示；极短文本替换为不可用说明

    不删除专家 ToolMessage，避免破坏审计轨迹。
    """
    if not messages:
        return messages
    v = validator or _validator
    target = _last_synthesis_ai(messages)
    if target is None:
        return messages

    text = str(target.content or "")
    ratings = extract_ratings_from_messages(messages)
    text = recover_ratings_in_text(text, ratings)
    text = sanitize_data_gaps(text)
    query = _original_query(messages)
    verdict = v.validate(
        text,
        expert_name="supervisor_synthesis",
        original_query=query,
    )
    new_text = apply_confidence_footer(text, verdict)
    if verdict.recommendation == Recommendation.REJECT and len(text.strip()) < 40:
        new_text = (
            "本轮综合置信度不足，未形成可发布结论。请重试或缩小问题范围，并核对工具原始返回。"
        )

    meta: dict[str, Any] = dict(getattr(target, "additional_kwargs", None) or {})
    meta["confidence"] = {
        "score": verdict.score,
        "level": str(verdict.level),
        "recommendation": str(verdict.recommendation),
        "flags": list(verdict.flags),
    }
    new_msg = AIMessage(
        content=new_text,
        additional_kwargs=meta,
        name=getattr(target, "name", None),
        id=getattr(target, "id", None),
    )
    # 替换最后一条综合 AIMessage
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i] is target:
            out[i] = new_msg
            break
    return out
