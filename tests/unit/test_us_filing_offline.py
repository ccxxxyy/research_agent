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
    from research_agent.mcp_servers.us_filing_server import _expand_wanted_forms, _form_matches

    wanted = {"10-K", "10-Q"}
    assert _form_matches("10-K", wanted)
    assert _form_matches("10-K/A", wanted)
    assert not _form_matches("8-K", wanted)

    etf_wanted = _expand_wanted_forms({"N-PORT", "N-CSR", "485BPOS"})
    assert _form_matches("NPORT-P", etf_wanted)
    assert _form_matches("NPORT-P/A", etf_wanted)
    assert _form_matches("N-CSRS", etf_wanted)
    assert _form_matches("485APOS", etf_wanted)
    assert not _form_matches("10-K", etf_wanted)


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
async def test_search_filings_etf_defaults_include_nport():
    """默认 forms 须能命中 ETF 的 NPORT-P / N-CSR（不再只靠 10-K 过滤）。"""
    from research_agent.mcp_servers import us_filing_server as mod

    submissions = {
        "name": "Invesco QQQ Trust, Series 1",
        "tickers": ["QQQ"],
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0001411573-24-000111",
                    "0001411573-24-000100",
                    "0001411573-24-000090",
                ],
                "filingDate": ["2024-06-28", "2024-05-30", "2024-03-01"],
                "form": ["NPORT-P", "N-CSR", "485BPOS"],
                "primaryDocument": ["primary_doc.xml", "ncsr.htm", "485bpos.htm"],
                "primaryDocDescription": ["NPORT-P", "N-CSR", "485BPOS"],
                "reportDate": ["2024-05-31", "2024-04-30", "2024-02-28"],
            }
        },
    }

    async def fake_json(url: str):
        if "company_tickers" in url:
            return {"0": {"cik_str": 1067839, "ticker": "QQQ", "title": "Invesco QQQ Trust"}}
        if "submissions" in url:
            return submissions
        raise AssertionError(url)

    with patch.object(mod, "_http_get_json", new=AsyncMock(side_effect=fake_json)):
        # 不传 forms → 使用 DEFAULT_FORMS（含 ETF 表单）
        result = await mod.search_filings("QQQ", limit=10)
        # 口语别名 N-PORT 也应命中 EDGAR 的 NPORT-P
        alias = await mod.search_filings("QQQ", forms="N-PORT", limit=5)

    assert "error" not in result
    assert result["count"] == 3
    assert {f["form"] for f in result["filings"]} == {"NPORT-P", "N-CSR", "485BPOS"}
    assert alias["count"] == 1
    assert alias["filings"][0]["form"] == "NPORT-P"


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

        sought = await mod.seek_filing_text(
            dl["local_path"], query="Item 1A", max_chars=200, num_windows=1
        )
        assert "error" not in sought or sought.get("text")
        assert sought.get("match")
        assert "Risk" in (sought.get("text") or "")


@pytest.mark.asyncio
async def test_seek_filing_text_item_and_keyword_windows(tmp_path: Path):
    from research_agent.mcp_servers import us_filing_server as mod

    # TOC 在前、正文 Item 1A 在后；中间塞足够字符模拟目录
    toc = "TABLE OF CONTENTS\nItem 1A Risk Factors .......... 12\n"
    pad = "x" * 9000
    body = (
        "\nItem 1A. Risk Factors\nWe face China export controls and competition risks. "
        + ("More risk detail. " * 400)
        + "\nItem 7. Management's Discussion and Analysis\nRevenue grew in gaming GPUs. "
        + ("MD&A paragraph. " * 200)
    )
    html = f"<html><body><pre>{toc}{pad}{body}</pre></body></html>".encode()
    path = tmp_path / "nvda.htm"
    path.write_bytes(html)

    risk = await mod.seek_filing_text(str(path), query="1A", max_chars=500, num_windows=2)
    assert "error" not in risk
    assert risk["item_key"] == "1a"
    assert risk["match"]["offset"] > 1000  # 跳过文首 TOC
    assert "China export" in risk["text"]
    assert risk["num_windows"] == 2
    assert "window 1/2" in risk["text"]
    assert risk["next_start_char"] > risk["match"]["offset"]

    mda = await mod.seek_filing_text(str(path), query="MD&A", max_chars=400, num_windows=1)
    assert "error" not in mda
    assert "Revenue grew" in mda["text"] or "gaming" in mda["text"].lower()

    kw = await mod.seek_filing_text(str(path), query="China", max_chars=300, num_windows=1)
    assert "error" not in kw
    assert "China" in kw["text"]

    miss = await mod.seek_filing_text(str(path), query="zzzz-not-found-zzzz", max_chars=100)
    assert miss.get("error")
    assert miss.get("match") is None


def test_seek_pattern_helpers():
    from research_agent.mcp_servers import us_filing_server as mod

    assert mod._item_key_from_query("Item 1A") == "1a"
    assert mod._item_key_from_query("risk factors") == "1a"
    assert mod._item_key_from_query("MD&A") == "7"
    assert mod._item_key_from_query("China") is None
    pats = mod._compile_seek_patterns("1A")
    assert pats
    text = "toc Item 1A\n" + ("." * 13000) + "\nItem 1A. Risk Factors\nHello"
    matches = mod._collect_seek_matches(text, pats)
    preferred = mod._prefer_seek_matches(matches, item_query=True)
    assert preferred[0]["offset"] > 1000


@pytest.mark.asyncio
async def test_download_rejects_non_sec_url():
    from research_agent.mcp_servers import us_filing_server as mod

    result = await mod.download_filing("https://example.com/x.htm")
    assert "error" in result


def test_form_matches_adv_and_d():
    from research_agent.mcp_servers.us_filing_server import _expand_wanted_forms, _form_matches

    wanted = _expand_wanted_forms({"ADV", "D"})
    assert _form_matches("ADV", wanted)
    assert _form_matches("ADV-E", wanted)
    assert _form_matches("D", wanted)
    assert _form_matches("D/A", wanted)


@pytest.mark.asyncio
async def test_get_entity_overview_mocked():
    from research_agent.mcp_servers import us_filing_server as mod

    submissions = {
        "name": "Apple Inc.",
        "tickers": ["AAPL"],
        "exchanges": ["Nasdaq"],
        "sic": "3571",
        "sicDescription": "ELECTRONIC COMPUTERS",
        "entityType": "operating",
        "fiscalYearEnd": "0930",
        "stateOfIncorporation": "CA",
        "addresses": {"business": {"city": "Cupertino"}},
        "formerNames": [],
        "filings": {"recent": {}},
    }

    async def fake_json(url: str):
        if "company_tickers" in url:
            return {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
        if "submissions" in url:
            return submissions
        raise AssertionError(url)

    with patch.object(mod, "_http_get_json", new=AsyncMock(side_effect=fake_json)):
        result = await mod.get_entity_overview("AAPL")

    assert "error" not in result
    assert result["overview"]["name"] == "Apple Inc."
    assert result["source"] == "data.sec.gov/submissions"
    assert "NAV" in result["note"]


@pytest.mark.asyncio
async def test_search_entity_by_name_mocked():
    from research_agent.mcp_servers import us_filing_server as mod

    fake_map = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
    }
    with patch.object(mod, "_http_get_json", new=AsyncMock(return_value=fake_map)):
        result = await mod.search_entity_by_name("Apple", limit=5)

    assert result["count"] == 1
    assert result["matches"][0]["ticker"] == "AAPL"
    assert result["source"] == "company_tickers.json"


@pytest.mark.asyncio
async def test_search_filings_form_d():
    from research_agent.mcp_servers import us_filing_server as mod

    submissions = {
        "name": "Some Issuer",
        "tickers": [],
        "filings": {
            "recent": {
                "accessionNumber": ["0001234567-24-000001"],
                "filingDate": ["2024-01-15"],
                "form": ["D"],
                "primaryDocument": ["xslFormDX01/primary_doc.xml"],
                "primaryDocDescription": ["D"],
                "reportDate": [""],
            }
        },
    }

    async def fake_json(url: str):
        if "company_tickers" in url:
            return {"0": {"cik_str": 1234567, "ticker": "TEST", "title": "Some Issuer"}}
        if "submissions" in url:
            return submissions
        raise AssertionError(url)

    with patch.object(mod, "_http_get_json", new=AsyncMock(side_effect=fake_json)):
        result = await mod.search_filings("TEST", forms="D,ADV", limit=5)

    assert result["count"] == 1
    assert result["filings"][0]["form"] == "D"
    assert "IAPD" in result["note"]


@pytest.mark.asyncio
async def test_search_investment_adviser_mocked():
    from research_agent.mcp_servers import us_filing_server as mod

    payload = {
        "hits": {
            "total": 1,
            "hits": [
                {
                    "_source": {
                        "firm_source_id": "157373",
                        "firm_name": "SEQUOIA CAPITAL OPERATIONS, LLC",
                        "firm_other_names": ["SEQUOIA CAPITAL OPERATIONS, LLC"],
                        "firm_ia_full_sec_number": "801-122957",
                        "firm_ia_scope": "ACTIVE",
                        "firm_ia_disclosure_fl": "N",
                        "firm_branches_count": 1,
                        "firm_ia_address_details": (
                            '{"officeAddress": {"city": "MENLO PARK", "state": "CA"}}'
                        ),
                    }
                }
            ],
        }
    }

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, params=None):
            assert "adviserinfo.sec.gov" in url
            assert params["query"] == "Sequoia Capital"
            return _Resp()

    with patch.object(mod.httpx, "AsyncClient", return_value=_Client()):
        result = await mod.search_investment_adviser("Sequoia Capital", limit=5)

    assert result["source"] == "iapd"
    assert result["count"] == 1
    assert result["matches"][0]["firm_id"] == "157373"
    assert "SEQUOIA" in result["matches"][0]["firm_name"]
    assert "IAPD" in result["note"]


@pytest.mark.asyncio
async def test_get_investment_adviser_overview_mocked():
    import json

    from research_agent.mcp_servers import us_filing_server as mod

    iacontent = {
        "basicInformation": {
            "firmId": 157373,
            "firmName": "SEQUOIA CAPITAL OPERATIONS, LLC",
            "otherNames": ["SEQUOIA CAPITAL OPERATIONS, LLC"],
            "iaScope": "ACTIVE",
            "advFilingDate": "07/17/2026",
            "hasPdf": "Y",
            "iaSECNumber": "122957",
            "iaSECNumberType": "801",
        },
        "registrationStatus": [{"secJurisdiction": "SEC", "status": "Approved"}],
        "brochures": {
            "brochuredetails": [
                {
                    "brochureVersionID": 1,
                    "brochureName": "FORM ADV PART 2.A",
                    "dateSubmitted": "3/31/2026",
                }
            ]
        },
    }
    payload = {
        "hits": {
            "total": 1,
            "hits": [{"_source": {"iacontent": json.dumps(iacontent)}}],
        }
    }

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, params=None):
            assert url.endswith("/157373")
            return _Resp()

    with patch.object(mod.httpx, "AsyncClient", return_value=_Client()):
        result = await mod.get_investment_adviser_overview("157373")

    assert result["found"] is True
    assert result["source"] == "iapd"
    assert result["overview"]["firm_name"] == "SEQUOIA CAPITAL OPERATIONS, LLC"
    assert result["overview"]["sec_number"] == "801-122957"
    assert result["overview"]["brochures"][0]["name"] == "FORM ADV PART 2.A"


@pytest.mark.asyncio
async def test_search_investment_adviser_empty_keyword():
    from research_agent.mcp_servers import us_filing_server as mod

    result = await mod.search_investment_adviser("  ")
    assert "error" in result
