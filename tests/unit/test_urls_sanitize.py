"""URL 清洗：剔除 HTML 泄漏的脏链。"""

from __future__ import annotations

from research_agent.text.urls import sanitize_http_url, sanitize_markdown_links


def test_reject_html_leaked_yahoo_url() -> None:
    dirty = "https://finance.yahoo.com/m/4276758e-a015-39b7-999b%3Cspan%20class="
    assert sanitize_http_url(dirty) == ""


def test_reject_raw_span_in_url() -> None:
    dirty = "https://finance.yahoo.com/news/foo<span class=x>bar"
    assert sanitize_http_url(dirty) == ""


def test_keep_clean_yahoo_url() -> None:
    clean = "https://finance.yahoo.com/quote/AAPL/news"
    assert sanitize_http_url(clean) == clean


def test_sanitize_markdown_drops_dirty_href() -> None:
    text = (
        "数据来源：[坏链](https://finance.yahoo.com/m/abc%3Cspan%20class=)、"
        "[好链](https://finance.yahoo.com/quote/AMD/news)"
    )
    out = sanitize_markdown_links(text)
    assert "%3Cspan" not in out
    assert "坏链" in out
    assert "](https://finance.yahoo.com/quote/AMD/news)" in out
