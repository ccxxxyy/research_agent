"""清洗 LLM 输出中错误的正负号组合（如 ``-+0.64%``）。"""

from __future__ import annotations

import re

# Unicode 减号 / 破折号 → ASCII -
_MINUS_RE = re.compile(r"[\u2212\u2012\u2013\u2014\uFE63\uFF0D]")
# 全角加号 → ASCII +
_PLUS_RE = re.compile(r"[\uFF0B\uFE62]")
# -+ / +-（可夹空白）后跟数字
_DOUBLE_SIGN_RE = re.compile(r"(?:-\s*\+|\+\s*-)\s*(?=\d)")


def sanitize_signed_percents(text: str) -> str:
    """把 ``-+0.64%`` / ``- +0.64%`` / ``−＋1.15%`` 归一成 ``-0.64%``。

    前端旧正则会把 ``-+0.64%`` 里的 ``+0.64%`` 当成上涨并着红；服务端先洗掉可避免此问题。
    """
    if not text:
        return text
    s = _MINUS_RE.sub("-", text)
    s = _PLUS_RE.sub("+", s)
    s = _DOUBLE_SIGN_RE.sub("-", s)
    return s
