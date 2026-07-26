"""数据来源 HTTP(S) URL 清洗。"""

from __future__ import annotations

import re
from urllib.parse import unquote, urlparse

_HTML_TAG_RE = re.compile(r"<[^>]*>")
# 常见污染：URL 内嵌 %3Cspan / <span class=
_DIRTY_URL_RE = re.compile(
    r"(?i)(%3c|%3e|<|>|javascript:|[\s\"']|span\s+class=|class=)",
)


def sanitize_http_url(url: str | None, *, max_len: int = 500) -> str:
    """返回可安全展示的 http(s) URL；脏链/非 http 返回空串。"""
    if not url:
        return ""
    text = str(url).strip()
    if not text:
        return ""
    # 一旦混入 HTML / 尖括号编码，整段不可信（勿 strip 后假装可点）
    try:
        decoded0 = unquote(text)
    except Exception:  # noqa: BLE001
        decoded0 = text
    if (
        "<" in text
        or ">" in text
        or "<" in decoded0
        or ">" in decoded0
        or "%3c" in text.lower()
        or "%3e" in text.lower()
        or _HTML_TAG_RE.search(text)
        or _DIRTY_URL_RE.search(decoded0)
    ):
        return ""
    # 截断到第一个空白或引号（有时整段杂质粘在后面）
    for sep in (" ", "\n", "\t", '"', "'"):
        if sep in text:
            text = text.split(sep, 1)[0]
    text = text.strip().rstrip(".,);]}>\"'")
    if len(text) > max_len:
        text = text[:max_len]
    if not text.lower().startswith(("http://", "https://")):
        return ""
    try:
        parsed = urlparse(text)
    except Exception:  # noqa: BLE001
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return text


def sanitize_markdown_links(text: str) -> str:
    """清洗正文 markdown 链接：脏 URL 改为纯文本标签。"""
    if not text or "](" not in text:
        return text or ""

    def _repl(m: re.Match[str]) -> str:
        label, url = m.group(1), m.group(2)
        clean = sanitize_http_url(url)
        if clean:
            return f"[{label}]({clean})"
        return label

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _repl, text)


__all__ = ["sanitize_http_url", "sanitize_markdown_links"]
