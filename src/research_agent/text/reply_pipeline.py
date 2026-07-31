"""研究终稿管道：缺口清洗 + 置信度脚注 +（可选）输出 Guard。"""

from __future__ import annotations

from research_agent.agents.confidence import ConfidenceValidator, Recommendation
from research_agent.security.prompt_guard import PromptGuard, ThreatLevel
from research_agent.text.disclaimer import with_financial_disclaimer
from research_agent.text.gap_sanitize import apply_confidence_footer, sanitize_data_gaps

_OUTPUT_BLOCKED_PLACEHOLDER = "[输出已过滤：检测到敏感信息泄漏风险]"
_default_guard = PromptGuard()
_default_confidence = ConfidenceValidator()


def guard_output_text(
    text: str,
    *,
    guard: PromptGuard | None = None,
    system_prompt: str | None = None,
) -> tuple[str, bool]:
    """对文本做输出 Guard。返回 ``(可能替换后的文本, 是否被拦截)``。"""
    g = guard or _default_guard
    verdict = g.check_output(text or "", system_prompt=system_prompt)
    if verdict.level == ThreatLevel.BLOCKED:
        return _OUTPUT_BLOCKED_PLACEHOLDER, True
    return text, False


def polish_research_reply(
    text: str,
    *,
    apply_disclaimer: bool = True,
    apply_confidence: bool = True,
    apply_gap_sanitize: bool = True,
    guard: PromptGuard | None = None,
    run_output_guard: bool = False,
    system_prompt: str | None = None,
) -> str:
    """终稿统一抛光：缺口清洗 → 置信度脚注 → 可选 Guard → 免责声明。"""
    cleaned = (text or "").strip()
    if apply_gap_sanitize:
        cleaned = sanitize_data_gaps(cleaned)
    if apply_confidence and cleaned:
        verdict = _default_confidence.validate(
            cleaned,
            expert_name="supervisor_synthesis",
            original_query="",
        )
        # REJECT 时仍保留正文，但加脚注；极端空文本由 validate 自身降分
        cleaned = apply_confidence_footer(cleaned, verdict)
        if verdict.recommendation == Recommendation.REJECT and len(cleaned.strip()) < 40:
            cleaned = (
                "本轮综合置信度不足，未形成可发布结论。请重试或缩小问题范围，并核对工具原始返回。"
            )
    if run_output_guard:
        cleaned, _blocked = guard_output_text(cleaned, guard=guard, system_prompt=system_prompt)
    if apply_disclaimer:
        cleaned = with_financial_disclaimer(cleaned)
    return cleaned
