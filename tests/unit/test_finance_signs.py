"""正负号清洗：避免 ``-+0.64%`` 被前端误判为上涨。"""

from research_agent.api.routes.supervisor import _clean_markdown
from research_agent.text.finance_signs import sanitize_signed_percents


def test_sanitize_ascii_double_sign() -> None:
    assert sanitize_signed_percents("Nasdaq 跌 -+0.64%") == "Nasdaq 跌 -0.64%"
    assert sanitize_signed_percents("跌幅 +-1.15%") == "跌幅 -1.15%"


def test_sanitize_spaced_and_unicode() -> None:
    assert sanitize_signed_percents("跌 **- +0.64%**") == "跌 **-0.64%**"
    assert sanitize_signed_percents("跌 −＋1.15%") == "跌 -1.15%"
    assert sanitize_signed_percents("涨 +0.46%") == "涨 +0.46%"


def test_clean_markdown_applies_sanitize() -> None:
    out = _clean_markdown("纳指领跌\n\n\n跌幅 -+0.64%\n")
    assert "-+0.64%" not in out
    assert "-0.64%" in out
    assert "\n\n\n" not in out
