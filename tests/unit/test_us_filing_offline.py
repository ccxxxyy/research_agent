"""``us_filing_server`` 离线单元测试 — mock SEC HTTP，无网络。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from research_agent.cache import reset_tool_cache_for_tests


@pytest.fixture(autouse=True)
def _reset_caches():
    from research_agent.mcp_servers import us_filing_server as mod

    reset_tool_cache_for_tests()
    mod.reset_ticker_cache_for_tests()
    yield
    reset_tool_cache_for_tests()
    mod.reset_ticker_cache_for_tests()


def test_pad_cik():
    from research_agent.mcp_servers.us_filing_server import _pad_cik

    assert _pad_cik("320193") == "0000320193"
    assert _pad_cik("0000320193") == "0000320193"


def test_document_url():
    from research_agent.mcp_servers.us_filing_server import _document_url

    url = _document_url(
        cik10="0000320193",
        accession="0000320193-23-000106",
        primary_document="aapl-20230930.htm",
    )
    assert url.endswith("/320193/000032019323000106/aapl-20230930.htm")
    assert url.startswith("https://www.sec.gov/Archives/edgar/data/")


def test_form_matches():
    from research_agent.mcp_servers.us_filing_server import _form_matches

    wanted = {"10-K", "10-Q"}
    assert _form_matches("10-K", wanted)
    assert _form_matches("10-K/A", wanted)
    assert not _form_matches("8-K", wanted)


def test_html_to_text():
    from research_agent.mcp_servers.us_filing_server import _html_to_text

    text = _html_to_text("<html><body><p>Risk Factors</p><script>x</script></body></html>")
    assert "Risk Factors" in text
    assert "script" not in text.lower() or "x" not in text


@pytest.mark.asyncio
async def test_resolve_cik_from_ticker_mocked():
    from research_agent.mcp_servers import us_filing_server as mod

    fake_map = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    }
    with patch.object(mod, "_http_get_json", new=AsyncMock(return_value=fake_map)):
        result = await mod.resolve_cik("aapl")

    assert "error" not in result
    assert result["cik10"] == "0000320193"
    assert result["ticker"] == "AAPL"


@pytest.mark.asyncio
async def test_resolve_cik_from_digits():
    from research_agent.mcp_servers import us_filing_server as mod

    result = await mod.resolve_cik("320193")
    assert result["cik10"] == "0000320193"
    assert result["source"] == "input_cik"


@pytest.mark.asyncio
async def test_search_filings_mocked():
    from research_agent.mcp_servers import us_filing_server as mod

    submissions = {
        "name": "Apple Inc.",
        "tickers": ["AAPL"],
        "filings": {
            "recent": {
                "accessionNumber": ["0000320193-23-000106", "0000320193-23-000077"],
                "filingDate": ["2023-11-03", "2023-08-04"],
                "form": ["10-K", "10-Q"],
                "primaryDocument": ["aapl-20230930.htm", "aapl-20230701.htm"],
                "primaryDocDescription": ["10-K", "10-Q"],
                "reportDate": ["2023-09-30", "2023-07-01"],
            }
        },
    }

    async def fake_json(url: str):
        if "company_tickers" in url:
            return {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
        if "submissions" in url:
            return submissions
        raise AssertionError(url)

    with patch.object(mod, "_http_get_json", new=AsyncMock(side_effect=fake_json)):
        result = await mod.search_filings("AAPL", forms="10-K", limit=5)

    assert "error" not in result
    assert result["count"] == 1
    assert result["filings"][0]["form"] == "10-K"
    assert "document_url" in result["filings"][0]


@pytest.mark.asyncio
async def test_download_and_parse_html(tmp_path: Path):
    from research_agent.mcp_servers import us_filing_server as mod

    html_bytes = b"<html><body><h1>Item 1A</h1><p>Risk one.</p></body></html>"
    url = "https://www.sec.gov/Archives/edgar/data/320193/000/a.htm"

    with (
        patch.object(mod, "DEFAULT_CACHE_DIR", tmp_path),
        patch.object(mod, "_http_get_bytes", new=AsyncMock(return_value=html_bytes)),
    ):
        dl = await mod.download_filing(url)
        assert "error" not in dl
        assert dl["from_cache"] is False
        assert Path(dl["local_path"]).exists()

        meta = await mod.extract_filing_metadata(dl["local_path"])
        assert meta["kind"] == "html"
        assert meta["char_count"] > 0

        parsed = await mod.parse_filing_text(dl["local_path"], start_char=0, max_chars=200)
        assert "Risk" in parsed["text"]


@pytest.mark.asyncio
async def test_download_rejects_non_sec_url():
    from research_agent.mcp_servers import us_filing_server as mod

    result = await mod.download_filing("https://example.com/x.htm")
    assert "error" in result
