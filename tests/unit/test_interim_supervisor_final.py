"""守护：主管「请稍候」过渡话不得被当成 FINAL。"""

from __future__ import annotations

from research_agent.api.routes.supervisor import (
    _looks_like_interim_supervisor_text,
    _looks_like_real_synthesis,
    _strip_meta_above_analysis,
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


def test_strip_meta_above_analysis_opening() -> None:
    raw = (
        "上述分析已完整呈现。今日美股跌幅榜的核心风险信号可浓缩为一句判断："
        "**半导体板块无差别杀跌**。"
    )
    out = _strip_meta_above_analysis(raw)
    assert not out.startswith("上述分析")
    assert "半导体板块" in out
    assert _strip_meta_above_analysis("## 结论\n标普上涨") == "## 结论\n标普上涨"
