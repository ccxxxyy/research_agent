"""假缺口后处理 + 终稿管道 + 置信度门控。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from research_agent.agents.confidence import Recommendation
from research_agent.graph.confidence_gate import apply_confidence_gate_to_messages
from research_agent.security.prompt_guard import PromptGuard, ThreatLevel
from research_agent.text.gap_sanitize import sanitize_data_gaps
from research_agent.text.reply_pipeline import guard_output_text, polish_research_reply


def test_sanitize_removes_fake_gap_bullets() -> None:
    text = """## 结论
宏发看好。

## 数据缺口
- 巨潮公告 PDF 未提取（营收明细可在年报核实）
- 北向资金个股流向今日接口空，无法判断外资当日操作
- 财联社快讯：本轮调用返回「接口超时」，已通过东财快讯回退获取宏观线索
- 东财研报评级旁路超时，未能获取买入、目标价等明细
- 东财新闻接口超时（真实失败可保留）

## 操作相关表述
无。
"""
    out = sanitize_data_gaps(text)
    assert "巨潮公告 PDF 未提取" not in out
    assert "北向资金个股流向" not in out
    assert "财联社" not in out
    assert "目标价" not in out
    assert "东财新闻接口超时" in out
    assert "## 数据缺口" in out


def test_sanitize_q2_funding_and_cls_narratives() -> None:
    """问题二：资金面/缺口里的北向空接口与财联社超时回退应被剥掉。"""
    text = """## 资金面
北向/南向资金：近 10 日数据接口返回为空，无法判断外资当日态度。数据缺口
北向/南向资金：近 10 日接口返回为空，无法判断外资在极端分化日的操作方向，建议后续补查。
财联社快讯：本轮调用返回「接口超时」，已通过东财快讯回退获取宏观线索，但部分盘中即时消息可能缺失。

## 多空对照
多。
"""
    out = sanitize_data_gaps(text)
    assert "近 10 日" not in out
    assert "财联社" not in out
    assert "无法判断外资" not in out
    assert "## 多空对照" in out


def test_sanitize_qe_sentiment_target_price_gap() -> None:
    text = """## 五、数据缺口
东财研报评级旁路（sentiment_get_stock_sentiment_report）：连续两次调用超时，未能获取机构具体评级明细（如哪家机构给出买入、目标价等）。目前评级信息来自证券时报/界面新闻的新闻报道，非直接研报接口。建议数据源恢复后补查。

## 可信度提示
综合文本存在需交叉验证的信号：round_number_suspicious；结论权重已按规则降权，请以工具返回字段为准。
"""
    out = sanitize_data_gaps(text)
    assert "目标价" not in out
    assert "sentiment_get_stock_sentiment_report" not in out
    assert "## 可信度提示" in out
    assert "round_number_suspicious" in out


def test_sanitize_sentiment_timeout_and_juchao_midreport_gap() -> None:
    """舆情节：sentiment 超时凑目标价 + 巨潮中报科目假缺口应剔除。"""
    text = """## 舆情与叙事
### 研报评级（有限数据）
证券时报 7 月 30 日报道《16 股今日获机构买入评级》，宏发股份在列，但摘要被截断
本轮 sentiment 工具三次超时，未能获取具体评级机构、目标价、评级分布等数据；数据缺口
中报完整科目：经营现金流恶化的具体原因（应收/应付/存货变动）需等中报正式披露后通过巨潮 PDF 提取

## 可信度提示
综合文本存在需交叉验证的信号：round_number_suspicious
"""
    out = sanitize_data_gaps(text)
    assert "三次超时" not in out
    assert "目标价" not in out
    assert "评级分布" not in out
    assert "巨潮" not in out
    assert "中报完整科目" not in out
    assert "证券时报" in out
    assert "round_number_suspicious" in out


def test_sanitize_strips_aux_analyst_miss_from_conclusion() -> None:
    text = (
        "## 结论\n"
        "宏发股份基本面扎实。研报评级方面，本轮舆情工具未返回 aux_signals.analyst "
        "结构化数据，以下分析聚焦财务与资金面，评级部分标注为数据缺口。\n"
        "不构成买卖指令。\n"
    )
    out = sanitize_data_gaps(text)
    assert "aux_signals.analyst" not in out
    assert "评级部分标注为数据缺口" not in out
    assert "基本面扎实" in out


def test_sanitize_strips_target_price_from_action_section() -> None:
    """问题三：操作节混入目标价区间时应子句剥离，保留其余风险/走势观察。"""
    text = """## 数据缺口
本轮已覆盖财务摘要/指标、股价走势、研报评级、新闻舆情。无实质性数据缺口。
个股资金流向接口本轮超时，暂缺主力资金流向明细，但对整体判断影响有限。

## 操作相关表述
机构一致给出"买入"评级、目标价区间 41-44 元（较当前价 +19%~+28%），基本面营收增速 +32% 验证了高增长逻辑。但净利率下滑与经营现金流恶化是两个需要持续跟踪的核心风险点，建议重点关注 Q3 财报中这两个指标的修复情况。当前价位处于 60 日区间中位偏上，短期趋势偏弱，可等待放量企稳信号后再做决策。
"""
    out = sanitize_data_gaps(text)
    assert "目标价" not in out
    assert "41-44" not in out
    assert "+19%" not in out
    assert "买入" in out
    assert "净利率下滑" in out
    assert "个股资金流向接口本轮超时" in out
    assert "无实质性数据缺口" in out


def test_sanitize_drops_empty_gap_section() -> None:
    text = """## 结论
ok

## 数据缺口
- 未调用 sentiment_expert
- 建议使用万得查看

## 多空对照
多。
"""
    out = sanitize_data_gaps(text)
    assert "## 数据缺口" not in out
    assert "万得" not in out
    assert "## 多空对照" in out


def test_polish_runs_gap_and_disclaimer() -> None:
    text = "## 结论\n测试\n\n## 数据缺口\n- 机构目标价缺失\n"
    out = polish_research_reply(text, run_output_guard=False)
    assert "机构目标价" not in out
    assert "不构成" in out or "免责" in out or "投资" in out


def test_guard_output_blocks_api_key_leak() -> None:
    guard = PromptGuard()
    leak = "The api_key: skabc123defghijklmnop"
    safe, blocked = guard_output_text(leak, guard=guard)
    assert blocked is True
    assert "过滤" in safe
    assert guard.check_output(leak).level == ThreatLevel.BLOCKED


def test_confidence_gate_annotates_message() -> None:
    messages = [
        HumanMessage(content="分析宁德时代"),
        AIMessage(content=("根据来源：无。可能大概也许或许似乎应该是据推测。" * 2)),
    ]
    out = apply_confidence_gate_to_messages(messages)
    last = out[-1]
    assert isinstance(last, AIMessage)
    conf = (last.additional_kwargs or {}).get("confidence") or {}
    assert "score" in conf
    assert conf["recommendation"] in {
        str(Recommendation.ACCEPT),
        str(Recommendation.DOWNWEIGHT),
        str(Recommendation.REJECT),
        "accept",
        "downweight",
        "reject",
    }


def test_guard_sse_helper_blocks_once() -> None:
    from research_agent.api.routes import supervisor as route

    # 人造 blocked：直接测 helper 对已 blocked 的短路
    text, blocked = route._guard_sse_frame_content("hello", blocked=True)
    assert blocked is True
    assert text == ""
