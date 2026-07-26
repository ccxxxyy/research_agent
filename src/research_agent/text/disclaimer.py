"""研究回答免责声明：去重后只保留系统统一的一条。"""

from __future__ import annotations

import re

from research_agent.security.prompt_guard import FINANCIAL_DISCLAIMER

# 模型常自写「免责声明」；服务端也会再拼一条 → 需先剥再统一附加
_TRAILING_DISCLAIMER_RE = re.compile(
    r"(?:\n\s*-{3,}\s*)*\n+\s*(?:\*\*)?免责声明\s*[：:].*\Z",
    re.DOTALL | re.IGNORECASE,
)


def strip_trailing_disclaimers(text: str) -> str:
    """去掉文末一个或多个免责声明块（含前置 ``---``）。"""
    s = (text or "").rstrip()
    while True:
        new = _TRAILING_DISCLAIMER_RE.sub("", s).rstrip()
        if new == s:
            break
        s = new
    # 残留的文末分隔线
    s = re.sub(r"(?:\n\s*-{3,}\s*)+\Z", "", s).rstrip()
    return s


def with_financial_disclaimer(text: str) -> str:
    """清洗后只附加一条 ``FINANCIAL_DISCLAIMER``。"""
    body = strip_trailing_disclaimers(text)
    if not body:
        return FINANCIAL_DISCLAIMER.strip()
    return body + FINANCIAL_DISCLAIMER
