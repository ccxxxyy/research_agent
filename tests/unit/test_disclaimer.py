"""免责声明去重：模型自写 + 系统附加不应重复。"""

from research_agent.api.routes.supervisor import _finalize_reply
from research_agent.security.prompt_guard import FINANCIAL_DISCLAIMER
from research_agent.text.disclaimer import strip_trailing_disclaimers, with_financial_disclaimer


def test_strip_model_disclaimer() -> None:
    raw = "结论：震荡。\n\n**免责声明：** 以上情景推演基于当前市场结构，不构成投资建议。"
    assert "免责声明" not in strip_trailing_disclaimers(raw)
    assert "震荡" in strip_trailing_disclaimers(raw)


def test_with_disclaimer_only_once() -> None:
    raw = "纳指下跌。\n\n---\n**免责声明：** 模型自写的免责声明。"
    out = with_financial_disclaimer(raw)
    assert out.count("免责声明") == 1
    assert "不构成任何投资建议" in out
    assert FINANCIAL_DISCLAIMER.strip() in out


def test_finalize_reply_dedupes() -> None:
    raw = (
        "## 结论\n美股震荡。\n\n"
        "**免责声明：** 情景推演不构成投资建议。\n\n"
        "---\n"
        "**免责声明：** 以上内容由 AI 研究助手自动生成，仅供参考。"
    )
    out = _finalize_reply(raw)
    assert out.count("免责声明") == 1
    assert "美股震荡" in out
