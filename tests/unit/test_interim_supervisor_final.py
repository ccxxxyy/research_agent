"""守护：主管「请稍候」过渡话不得被当成 FINAL。"""

from __future__ import annotations

from research_agent.api.routes.supervisor import (
    _looks_like_interim_supervisor_text,
    _looks_like_real_synthesis,
)


def test_interim_please_wait_detected() -> None:
    text = (
        "数据正在获取中，我已将分析任务交给美股数据专家，"
        "正在查询标普500 (^GSPC) 和纳斯达克综合指数 (^IXIC) 的实时行情及近期走势。"
        "请稍候，结果出来后我会立即为您呈现完整的分析报告。"
    )
    assert _looks_like_interim_supervisor_text(text)
    assert not _looks_like_real_synthesis(text)


def test_real_synthesis_with_pct() -> None:
    text = (
        "## 市场概况\n"
        "标普500 收于 5600 点，涨跌幅 **+1.25%**。\n"
        "数据来源：[Yahoo Finance](https://finance.yahoo.com)"
    )
    assert not _looks_like_interim_supervisor_text(text)
    assert _looks_like_real_synthesis(text)


def test_empty_is_interim() -> None:
    assert _looks_like_interim_supervisor_text("")
    assert not _looks_like_real_synthesis("")
