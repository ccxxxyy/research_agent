"""研究回答「数据缺口」后处理：删除假缺口 / 模板凑数条目。

Prompt 层已有硬规则，但模型仍可能编造「巨潮 PDF 未提取」「北向个股流向」等。
本模块在 FINAL 定稿时确定性清洗，不依赖模型自觉。
"""

from __future__ import annotations

import re
from typing import Any

# 命中即删除的缺口行（用户未点名、或本系统无对应能力 / 已有回退仍抱怨）
_FAKE_GAP_LINE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"巨潮.*PDF\s*未提取", re.I),
    re.compile(r"通过巨潮\s*PDF\s*提取", re.I),
    re.compile(r"PDF\s*(未提取|提取).*(年报|中报|科目|营收|现金流|应收|存货)", re.I),
    re.compile(r"中报完整科目", re.I),
    re.compile(r"(年报|中报).{0,20}(正式披露|披露后).{0,30}(巨潮|PDF|科目)", re.I),
    re.compile(r"需等.{0,16}(中报|年报).{0,40}(巨潮|PDF)", re.I),
    re.compile(r"(应收账款|应收/应付|存货).*(明细|巨潮|年报|中报|变动)", re.I),
    re.compile(r"北向资金\s*个股\s*流向", re.I),
    re.compile(r"个股\s*北向.*(接口|流向|空|无法)", re.I),
    # 市场级北向/南向：字段映射修复前模型常把 null 净买额写成「近 N 日为空」
    re.compile(
        r"(北向|南向).{0,12}(资金)?.{0,20}近\s*\d+\s*日.*(接口|数据).*(为空|返回为空)", re.I
    ),
    re.compile(r"(北向|南向).{0,20}(接口|数据).*(为空|返回为空).{0,30}(外资|态度|操作)", re.I),
    re.compile(r"无法判断外资.{0,16}(态度|操作|方向)", re.I),
    re.compile(r"外资对.{0,20}当日操作", re.I),
    re.compile(r"未调用\s*sentiment_expert", re.I),
    re.compile(r"未调(用)?舆情", re.I),
    re.compile(r"新闻情绪面待补充", re.I),
    re.compile(r"机构目标价", re.I),
    re.compile(r"目标价", re.I),  # 本系统研报旁路无目标价字段
    re.compile(r"缺(少|失)?.{0,8}目标价", re.I),
    re.compile(r"评级分布", re.I),
    re.compile(r"万得|同花顺|Wind|Bloomberg|付费终端", re.I),
    re.compile(r"财联社.{0,40}(超时|接口超时)", re.I),
    re.compile(r"已通过东财.{0,12}回退", re.I),
    re.compile(r"本轮未拉取.{0,10}(龙虎榜|北向)", re.I),
    re.compile(r"未拉取.{0,8}(龙虎榜|研报全文)", re.I),
    # sentiment 失败时仍把「无目标价/评级机构」当缺口凑数
    re.compile(
        r"sentiment.{0,40}(超时|失败|空).{0,40}(目标价|评级机构|评级分布|评级明细)",
        re.I,
    ),
    re.compile(r"(研报评级|sentiment_get_stock_sentiment_report).{0,80}(目标价|评级明细)", re.I),
)

# 可出现在「资金面/舆情」等非「数据缺口」小节的假叙事（行级删除）
_GLOBAL_FAKE_NARRATIVE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"巨潮.*PDF\s*未提取", re.I),
    re.compile(r"通过巨潮\s*PDF\s*提取", re.I),
    re.compile(r"中报完整科目", re.I),
    re.compile(r"(年报|中报).{0,20}(正式披露|披露后).{0,30}(巨潮|PDF|科目)", re.I),
    re.compile(r"需等.{0,16}(中报|年报).{0,40}(巨潮|PDF)", re.I),
    re.compile(r"(应收账款|应收/应付|存货).*(明细|巨潮|年报|中报|变动)", re.I),
    re.compile(r"北向资金\s*个股\s*流向", re.I),
    re.compile(r"个股\s*北向.*(接口|流向|空|无法)", re.I),
    re.compile(
        r"(北向|南向).{0,12}(资金)?.{0,20}近\s*\d+\s*日.*(接口|数据).*(为空|返回为空)", re.I
    ),
    re.compile(r"(北向|南向).{0,20}(接口|数据).*(为空|返回为空).{0,30}(外资|态度|操作)", re.I),
    re.compile(r"无法判断外资.{0,16}(态度|操作|方向)", re.I),
    re.compile(r"财联社.{0,40}(超时|接口超时)", re.I),
    re.compile(r"已通过东财.{0,12}回退", re.I),
    re.compile(
        r"sentiment.{0,40}(超时|失败|空).{0,40}(目标价|评级机构|评级分布|评级明细)",
        re.I,
    ),
    re.compile(r"未能获取具体评级机构", re.I),
    # 「未返回 aux_signals.analyst / 评级标缺口」混在结论长句里：只做子句剥离，勿整行删除
)

# A 股研报旁路无目标价字段；模型常把新闻里的价写进「操作相关表述」——做子句级剥离，保留同句其余观察
_TARGET_PRICE_SPAN_RE = re.compile(
    r"[、,，/]?\s*"
    r"(?:机构(?:一致)?(?:给出)?)?目标价"
    r"(?:区间)?"
    r"[^。；;\n元]{0,40}"
    r"[\d.]+(?:\s*[-~～—至到]\s*[\d.]+)?"
    r"\s*元"
    r"(?:\s*[（(][^）)\n]{0,48}[）)])?",
    re.I,
)
_TARGET_PRICE_BARE_RE = re.compile(
    r"[、,，/]?\s*(?:机构)?目标价\s*(?:区间\s*)?[\d.]+(?:\s*[-~～—至到]\s*[\d.]+)?",
    re.I,
)

# 结论里误称「未返回 aux_signals.analyst / 评级标缺口」（工具常有顶层 ratings_sample）
_AUX_ANALYST_MISS_SENT_RE = re.compile(
    r"[^。\n；;]*?(?:未返回|没有|缺失|未获取)[^。\n；;]*?aux_signals\.analyst[^。\n；;]*[。．.!？?；;]?",
    re.I,
)
_RATING_MARKED_GAP_SENT_RE = re.compile(
    r"[^。\n；;]*?(?:评级部分标注为数据缺口|研报评级方面[^。\n]{0,80}数据缺口)[^。\n；;]*[。．.!？?；;]?",
    re.I,
)
_RATING_MISSING_CELL_RE = re.compile(
    r"(研报评级|机构评级)\s*[：:|\t]\s*未取得",
    re.I,
)

_GAP_HEADING_RE = re.compile(
    r"^#{1,6}\s*数据缺口\s*$",
    re.MULTILINE,
)
_ANY_HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_BULLET_RE = re.compile(r"^(\s*[-*•]\s+|\s*\d+[.)、]\s+)")


def _line_body(line: str) -> str:
    text = line.strip()
    if not text:
        return ""
    return _BULLET_RE.sub("", text).strip()


def _is_fake_gap_line(line: str) -> bool:
    body = _line_body(line)
    if not body:
        return False
    return any(p.search(body) for p in _FAKE_GAP_LINE_RES)


def _is_global_fake_narrative(line: str) -> bool:
    body = _line_body(line)
    if not body:
        return False
    return any(p.search(body) for p in _GLOBAL_FAKE_NARRATIVE_RES)


def _filter_gap_section_body(body: str) -> str:
    kept: list[str] = []
    for raw in body.splitlines():
        if not raw.strip():
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if _is_fake_gap_line(raw):
            continue
        kept.append(raw)
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def _strip_unsupported_target_prices(text: str) -> str:
    """从全文剥离「目标价 + 价格」子句，避免操作节把无字段数据写成依据。"""
    if not text or "目标价" not in text:
        return text
    out = _TARGET_PRICE_SPAN_RE.sub("", text)
    out = _TARGET_PRICE_BARE_RE.sub("", out)
    # 残留标点：评级、，基本面 → 评级，基本面
    out = re.sub(r"([、,，])\s*([、,，])+", r"\1", out)
    out = re.sub(r"([“\"'])\s*([、,，])\s*", r"\1", out)
    out = re.sub(r"([、,，])\s*(?=[。．.!？?\n]|$)", "", out)
    return out


def _strip_aux_analyst_miss_claims(text: str) -> str:
    """剥离「未返回 aux_signals.analyst / 评级标为数据缺口」误称。"""
    if not text:
        return text
    if "aux_signals" not in text and "评级部分标注为数据缺口" not in text:
        return text
    out = _AUX_ANALYST_MISS_SENT_RE.sub("", text)
    out = _RATING_MARKED_GAP_SENT_RE.sub("", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


def _finalize_sanitize(text: str) -> str:
    text = _strip_unsupported_target_prices(text)
    return _strip_aux_analyst_miss_claims(text)


def sanitize_data_gaps(
    text: str,
    *,
    attempted_tools: list[str] | None = None,
) -> str:
    """清洗 ``## 数据缺口``：删除假缺口行；小节删空则整节移除。

    另对全文做一轮「假叙事」清洗，并剥离无依据的目标价子句。
    """
    del attempted_tools  # 预留：按已调用工具白名单过滤
    if not text:
        return text
    if "数据缺口" not in text and "aux_signals" not in text:
        result = text
        if any(p.search(text) for p in _GLOBAL_FAKE_NARRATIVE_RES):
            result = _scrub_global_narratives(text)
        return _finalize_sanitize(result)

    matches = list(_GAP_HEADING_RE.finditer(text))
    if not matches:
        lines = [ln for ln in text.splitlines() if not _is_fake_gap_line(ln)]
        result = _scrub_global_narratives("\n".join(lines))
        result = re.sub(r"\n{3,}", "\n\n", result)
        return _finalize_sanitize(result)

    pieces: list[str] = []
    cursor = 0
    for m in matches:
        pieces.append(text[cursor : m.start()])
        heading_end = m.end()
        # 吃掉标题后的单个换行
        if heading_end < len(text) and text[heading_end] == "\n":
            heading_end += 1
        rest = text[heading_end:]
        next_h = _ANY_HEADING_RE.search(rest)
        if next_h:
            body = rest[: next_h.start()]
            section_end = heading_end + next_h.start()
        else:
            body = rest
            section_end = len(text)
        filtered = _filter_gap_section_body(body)
        if filtered.strip():
            pieces.append(m.group(0).rstrip() + "\n")
            pieces.append(filtered)
            if section_end < len(text) and not filtered.endswith("\n"):
                pieces.append("\n")
        cursor = section_end
    pieces.append(text[cursor:])
    result = "".join(pieces)
    result = _scrub_global_narratives(result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return _finalize_sanitize(result)


def _scrub_global_narratives(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not _is_global_fake_narrative(ln)]
    return "\n".join(lines)


def apply_confidence_footer(text: str, verdict: Any) -> str:
    """按置信度建议追加简短脚注（不覆盖正文）。"""
    from research_agent.agents.confidence import Recommendation

    if verdict is None:
        return text
    if "## 可信度提示" in text:
        return text
    rec = getattr(verdict, "recommendation", None)
    flags = tuple(getattr(verdict, "flags", ()) or ())
    if rec == Recommendation.ACCEPT and not flags:
        return text
    if rec == Recommendation.REJECT:
        note = (
            "\n\n## 可信度提示\n"
            "- 本轮综合文本触发低置信度规则"
            + (f"（{', '.join(flags[:5])}）" if flags else "")
            + "，请优先核对上文工具数字与 ``source_url``，勿将未锚定表述当作事实。\n"
        )
        return text.rstrip() + note
    if rec == Recommendation.DOWNWEIGHT or flags:
        note = (
            "\n\n## 可信度提示\n"
            "- 综合文本存在需交叉验证的信号"
            + (f"：{', '.join(flags[:5])}" if flags else "")
            + "；结论权重已按规则降权，请以工具返回字段为准。\n"
        )
        return text.rstrip() + note
    return text
