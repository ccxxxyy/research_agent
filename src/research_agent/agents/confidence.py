"""专家输出置信度校验模块。

在每个专家返回结果时，使用 LIGHT 层级模型做快速置信度校验：
1. 数字合理性检查（数值是否在合理范围内）
2. 结论一致性检查（结论是否与引用的原文矛盾）
3. 幻觉指标检测（是否包含编造痕迹）

校验结果包含：
- confidence_score: 0.0-1.0 的置信度评分
- flags: 触发的具体问题标签
- recommendation: 对 supervisor 的建议（accept/downweight/reject）

设计原则
--------
- 快速：使用 LIGHT 模型（最低延迟/成本），单次调用
- 非阻塞：校验失败不会阻止专家输出传递，而是标记降权
- 可选：supervisor 根据 confidence_score 决定是否采纳
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class ConfidenceLevel(StrEnum):
    """置信度等级。"""

    HIGH = "high"  # >= 0.8, 可直接采纳
    MEDIUM = "medium"  # 0.5-0.8, 建议交叉验证
    LOW = "low"  # < 0.5, supervisor 应降权或丢弃


class Recommendation(StrEnum):
    """对 supervisor 的处理建议。"""

    ACCEPT = "accept"  # 高置信度，直接使用
    DOWNWEIGHT = "downweight"  # 中置信度，综合时降低权重
    REJECT = "reject"  # 低置信度，不纳入最终报告


@dataclass(frozen=True)
class ConfidenceVerdict:
    """单次置信度校验结果。"""

    score: float
    level: ConfidenceLevel
    recommendation: Recommendation
    flags: tuple[str, ...] = ()
    details: str = ""

    @property
    def is_reliable(self) -> bool:
        return self.level in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM)


# ---------------------------------------------------------------------------
# Rule-based pre-checks (fast, no LLM call needed) 基于规则的预检查
# ---------------------------------------------------------------------------

_HALLUCINATION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "fabricated_citation",
        re.compile(
            r"(?:根据|来源|引用|参考|出处)[：:]\s*(?:无|暂无|不详|未知)",
            re.IGNORECASE,
        ),
    ),
    (
        "hedging_overload",
        re.compile(
            r"(?:可能|大概|也许|或许|似乎|应该是|据推测){3,}",
        ),
    ),
    (
        "round_number_suspicious",
        re.compile(
            r"(?:约|大约|接近)\s*\d+(?:\.0+)?\s*(?:亿|万|%)",
        ),
    ),
    (
        "self_contradiction",
        re.compile(
            r"(?:但是|然而|不过).{0,50}(?:相反|矛盾|不一致)",
        ),
    ),
    (
        "tool_unavailable_claim",
        re.compile(
            r"(?:工具|功能|接口)\s*(?:不可用|受限|无法访问|暂不支持)",
            re.IGNORECASE,
        ),
    ),
]

_NUMERIC_SANITY: list[tuple[str, re.Pattern[str], float, float]] = [
    ("pe_ratio_extreme", re.compile(r"(?:市盈率|PE|P/E)[：:\s]*([+-]?\d+\.?\d*)"), -100, 10000),
    ("roe_extreme", re.compile(r"(?:ROE|净资产收益率)[：:\s]*([+-]?\d+\.?\d*)%?"), -200, 200),
    (
        "revenue_negative",
        re.compile(r"(?:营收|营业收入|总收入)[：:\s]*([+-]?\d+\.?\d*)"),
        0,
        float("inf"),
    ),
    ("stock_price_extreme", re.compile(r"(?:股价|收盘价|现价)[：:\s]*([+-]?\d+\.?\d*)"), 0, 100000),
]


def _run_pattern_checks(text: str) -> list[str]:
    """执行基于正则的快速检查，返回触发的标签列表。"""
    flags: list[str] = []
    for name, pattern in _HALLUCINATION_PATTERNS:
        if pattern.search(text):
            flags.append(name)
    return flags


def _run_numeric_checks(text: str) -> list[str]:
    """检查数值是否在合理范围内。"""
    flags: list[str] = []
    for name, pattern, lo, hi in _NUMERIC_SANITY:
        for match in pattern.finditer(text):
            try:
                value = float(match.group(1))
                if value < lo or value > hi:
                    flags.append(f"{name}:{value}")
            except (ValueError, IndexError):
                continue
    return flags


def _compute_base_score(text: str, flags: list[str]) -> float:
    """根据规则检查结果计算基础置信度分数。"""
    score = 1.0

    penalty_per_flag = 0.15
    score -= len(flags) * penalty_per_flag

    if not text.strip():
        score = 0.0
    elif len(text.strip()) < 20:
        score -= 0.3

    error_indicators = ["error", "失败", "超时", "异常", "unavailable"]
    for indicator in error_indicators:
        if indicator.lower() in text.lower():
            score -= 0.2
            break

    return max(0.0, min(1.0, score))


class ConfidenceValidator:
    """专家输出置信度校验器。

    Usage::

        validator = ConfidenceValidator()
        verdict = validator.validate(
            expert_output="宁德时代2023年营收4009亿元...",
            expert_name="data_expert",
            original_query="分析宁德时代2023年业绩",
        )
        if verdict.recommendation == Recommendation.REJECT:
            # supervisor 丢弃此结果或要求专家重试
            ...

    对于需要更高精度的场景，可通过 ``validate_with_llm`` 使用 LIGHT 模型做深度语义校验（额外一次 LLM 调用）。
    """

    def validate(
        self,
        expert_output: str,
        expert_name: str = "",
        original_query: str = "",
        context_snippets: list[str] | None = None,
    ) -> ConfidenceVerdict:
        """纯规则校验（零延迟，无 LLM 调用）， 适合在每次专家返回时立即执行。"""
        all_flags: list[str] = []

        all_flags.extend(_run_pattern_checks(expert_output))
        all_flags.extend(_run_numeric_checks(expert_output))

        if context_snippets:
            contradiction_flags = _check_source_consistency(expert_output, context_snippets)
            all_flags.extend(contradiction_flags)

        score = _compute_base_score(expert_output, all_flags)

        level = _score_to_level(score)
        recommendation = _level_to_recommendation(level)

        return ConfidenceVerdict(
            score=score,
            level=level,
            recommendation=recommendation,
            flags=tuple(all_flags),
            details=f"Expert: {expert_name}, Query: {original_query[:100]}",
        )

    def build_llm_validation_prompt(
        self,
        expert_output: str,
        expert_name: str,
        original_query: str,
    ) -> str:
        """构建用于 LLM 深度校验的 prompt。

        调用方获取此 prompt 后自行调用 LIGHT 模型，解析返回的 JSON 得到更精确的置信度评分。
        """
        return f"""\
你是一个输出质量校验器。请评估以下专家回答的可靠性。

【用户原始问题】
{original_query}

【专家身份】
{expert_name}

【专家回答】
{expert_output[:3000]}

请从以下维度打分（每项 0-1）：
1. factual_consistency: 数字和事实是否自洽，无内部矛盾
2. source_grounding: 结论是否有数据/引用支撑（vs 凭空断言）
3. completeness: 是否回答了问题的核心部分
4. hallucination_risk: 是否有编造痕迹（越低越好，0=无编造风险）

返回严格 JSON（无其他文本）：
{{"factual_consistency": 0.X, "source_grounding": 0.X, "completeness": 0.X, "hallucination_risk": 0.X, "overall": 0.X, "flags": ["flag1", ...]}}
"""


def _check_source_consistency(output: str, sources: list[str]) -> list[str]:
    """粗略检查专家输出是否与提供的源文本矛盾。

    使用简单的数字提取对比：如果专家提到的数字在源中完全不存在，标记。
    """
    flags: list[str] = []

    output_numbers = set(re.findall(r"\d+\.?\d*", output))
    if not output_numbers or not sources:
        return flags

    source_text = " ".join(sources)
    source_numbers = set(re.findall(r"\d+\.?\d*", source_text))

    large_numbers_in_output = {n for n in output_numbers if float(n) > 100}

    if large_numbers_in_output:
        ungrounded = large_numbers_in_output - source_numbers
        ratio = len(ungrounded) / len(large_numbers_in_output) if large_numbers_in_output else 0
        if ratio > 0.5:
            flags.append(f"ungrounded_numbers:{len(ungrounded)}/{len(large_numbers_in_output)}")

    return flags


def _score_to_level(score: float) -> ConfidenceLevel:
    if score >= 0.8:
        return ConfidenceLevel.HIGH
    elif score >= 0.5:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


def _level_to_recommendation(level: ConfidenceLevel) -> Recommendation:
    return {
        ConfidenceLevel.HIGH: Recommendation.ACCEPT,
        ConfidenceLevel.MEDIUM: Recommendation.DOWNWEIGHT,
        ConfidenceLevel.LOW: Recommendation.REJECT,
    }[level]
