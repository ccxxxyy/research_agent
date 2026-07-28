"""跨市场 MIXED 编排计划（P5）。

不引入新的混合专家、不合并 MCP 工具：仅根据 ``MarketResolution`` 生成可注入 supervisor 的**分侧子任务清单**，约束平行隔离路由。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from research_agent.market.types import Market, MarketResolution, SymbolRef

_COMPARE_RE = re.compile(r"(对比|比较|对照|vs\.?|versus|横比|孰优|谁更)", re.I)
_FILING_RE = re.compile(r"(年报|季报|公告|披露|10-?\s*k|10-?\s*q|8-?\s*k|edgar|巨潮)", re.I)
_PRIVATE_FUND_RE = re.compile(
    r"(私募|中基协|协会备案|amac|form\s*d|form\s*adv|adv\s*披露|private\s*equity|private\s*fund)",
    re.I,
)
_NEWS_RE = re.compile(r"(新闻|快讯|资讯|headline|news)", re.I)
_SENTIMENT_RE = re.compile(r"(舆情|情绪|情感|sentiment)", re.I)
_PRICE_RE = re.compile(r"(股价|报价|行情|走势|涨跌|市值|price|quote|chart)", re.I)


@dataclass(frozen=True)
class MixedSubTask:
    """一条单侧子任务。"""

    side: Market  # CN_A 或 US
    focus: str
    intent: str
    preferred_experts: tuple[str, ...]
    instruction: str


@dataclass(frozen=True)
class MixedOrchestrationPlan:
    """MIXED 问句的编排计划。"""

    subtasks: tuple[MixedSubTask, ...]
    synthesis_hint: str
    is_comparison: bool = False

    def format_for_prompt(self) -> str:
        if not self.subtasks:
            return (
                "[MixedOrchestration]\n"
                "无明确双边标的；请先把问句拆成 A 股侧与美股侧子问题，"
                "再分别移交对应市场专家，最后做综合对比。"
            )
        lines = ["[MixedOrchestration]"]
        if self.is_comparison:
            lines.append("mode=comparison")
        for i, t in enumerate(self.subtasks, start=1):
            experts = ", ".join(t.preferred_experts)
            lines.append(
                f"{i}. side={t.side.value} focus={t.focus} intent={t.intent} "
                f"experts=[{experts}] → {t.instruction}"
            )
        lines.append("synthesis=" + self.synthesis_hint)
        lines.append(
            "规则：每侧只用本侧专家；禁止跨市场工具；最终回答分侧陈述再综合，"
            "标明币种/时区/会计口径差异。"
        )
        lines.append(
            "强制：清单中每一侧至少完成一次对应 ``transfer_to_<专家>`` 后再写最终回答；"
            "严禁只写「需转交/请交给 sentiment_expert」而不发起移交。"
            "美股侧可一次移交批量查多个 US ticker；A 股侧对 CN 标的另一次移交。"
        )
        return "\n".join(lines)


def _infer_intent(query: str) -> str:
    if _PRIVATE_FUND_RE.search(query):
        return "private_fund"
    if _FILING_RE.search(query):
        return "filing"
    if _SENTIMENT_RE.search(query):
        return "sentiment"
    if _NEWS_RE.search(query):
        return "news"
    if _PRICE_RE.search(query):
        return "quote"
    return "overview"


def _experts_for(side: Market, intent: str) -> tuple[str, ...]:
    if side == Market.US:
        mapping = {
            "private_fund": ("us_filing_expert",),
            "filing": ("us_filing_expert",),
            "news": ("us_news_expert",),
            "sentiment": ("us_sentiment_expert", "us_news_expert"),
            "quote": ("us_data_expert",),
            "overview": ("us_data_expert",),
        }
    else:
        mapping = {
            "private_fund": ("fund_expert",),
            "filing": ("report_expert",),
            "news": ("news_expert",),
            "sentiment": ("sentiment_expert", "news_expert"),
            "quote": ("data_expert",),
            "overview": ("data_expert",),
        }
    return mapping.get(intent, mapping["overview"])


def _instruction_for(side: Market, focus: str, intent: str) -> str:
    side_name = "美国市场" if side == Market.US else "中国市场"
    if intent == "private_fund":
        if side == Market.US:
            return (
                f"仅用 {side_name} us_filing_expert 查 {focus} 的 EDGAR 概况 / Form D·ADV；"
                f"禁止编造私募净值，勿用 us_data 共同基金接口"
            )
        return (
            f"仅用 {side_name} fund_expert 的 AMAC 私募备案工具查 {focus}；"
            f"禁止编造净值，勿用公募 nav 冒充"
        )
    if intent == "filing":
        return f"仅用{side_name}披露专家查 {focus} 的披露/年报，勿跨市场"
    if intent == "news":
        return f"仅用{side_name}新闻专家查 {focus} 相关新闻"
    if intent == "sentiment":
        return f"仅用{side_name}舆情专家分析 {focus} 情绪"
    if intent == "quote":
        return f"仅用{side_name}行情专家查 {focus} 报价/走势"
    return f"仅用{side_name}侧专家回答与 {focus} 相关的子问题"


def _focus_label(sym: SymbolRef) -> str:
    name = sym.display_name or sym.raw
    if sym.ticker:
        return f"{name}({sym.ticker})"
    return name


def build_mixed_orchestration_plan(
    resolution: MarketResolution,
    query: str = "",
) -> MixedOrchestrationPlan | None:
    """仅当 ``market=MIXED`` 时生成编排计划；否则返回 ``None``。"""
    if resolution.market != Market.MIXED:
        return None

    intent = _infer_intent(query)
    is_cmp = bool(_COMPARE_RE.search(query))
    cn_syms = [s for s in resolution.symbols if s.market == Market.CN_A]
    us_syms = [s for s in resolution.symbols if s.market == Market.US]

    subtasks: list[MixedSubTask] = []

    if cn_syms:
        for sym in cn_syms[:3]:
            focus = _focus_label(sym)
            subtasks.append(
                MixedSubTask(
                    side=Market.CN_A,
                    focus=focus,
                    intent=intent,
                    preferred_experts=_experts_for(Market.CN_A, intent),
                    instruction=_instruction_for(Market.CN_A, focus, intent),
                )
            )
    else:
        cn_focus = "中国市场侧主题"
        cn_instr = (
            "拆出问句中的中国市场子问题（含 AMAC 私募备案时走 fund_expert），仅用中国市场侧专家作答"
            if intent == "private_fund"
            else "拆出问句中的中国市场子问题，仅用中国市场侧专家作答"
        )
        subtasks.append(
            MixedSubTask(
                side=Market.CN_A,
                focus=cn_focus,
                intent=intent,
                preferred_experts=_experts_for(Market.CN_A, intent),
                instruction=cn_instr,
            )
        )

    if us_syms:
        for sym in us_syms[:3]:
            focus = _focus_label(sym)
            subtasks.append(
                MixedSubTask(
                    side=Market.US,
                    focus=focus,
                    intent=intent,
                    preferred_experts=_experts_for(Market.US, intent),
                    instruction=_instruction_for(Market.US, focus, intent),
                )
            )
    else:
        us_focus = "美国市场侧主题"
        us_instr = (
            "拆出问句中的美国市场子问题（ADV/Form D 走 us_filing_expert），仅用美国市场侧专家作答"
            if intent == "private_fund"
            else "拆出问句中的美国市场子问题，仅用美国市场侧专家作答"
        )
        subtasks.append(
            MixedSubTask(
                side=Market.US,
                focus=us_focus,
                intent=intent,
                preferred_experts=_experts_for(Market.US, intent),
                instruction=us_instr,
            )
        )

    if is_cmp:
        synthesis = (
            "分侧汇总关键数字后做对比；明确标注币种、交易时区与会计口径，禁止混用工具链数据。"
        )
    else:
        synthesis = "按侧别组织回答；若某一侧专家未挂载，明确写明能力缺口后再综合。"

    return MixedOrchestrationPlan(
        subtasks=tuple(subtasks),
        synthesis_hint=synthesis,
        is_comparison=is_cmp,
    )


__all__ = [
    "MixedOrchestrationPlan",
    "MixedSubTask",
    "build_mixed_orchestration_plan",
]
