"""MCP Server — A-share announcement / research-report PDFs (巨潮资讯).

This is the **document plane** of the Phase-4 financial agent. It
replaces the Phase-0 placeholder ``document_server`` and is the
upstream data source for Phase-4.5's RAG ingestion pipeline.

Why 巨潮资讯
------------
巨潮资讯 (``cninfo.com.cn``) is the official disclosure portal
designated by the CSRC for Shenzhen- and Shanghai-listed companies.
Every 10-K-equivalent annual report, 10-Q quarterly, emergency
announcement, and underwriter prospectus lives there with a stable
URL pattern:

    http://static.cninfo.com.cn/finalpage/<YYYY-MM-DD>/<announcementId>.PDF

That predictability is what makes a tool-based workflow viable —
we don't have to scrape a search UI; cninfo exposes a structured
``/new/hisAnnouncement/query`` JSON endpoint and we just derive the
PDF URL from the results.

Tools exposed
-------------
1. ``search_announcements`` — list announcements for a ticker, with
   the derived PDF URL already attached to each row.
2. ``download_pdf`` — download one PDF into a content-hashed cache
   directory. Re-downloading the same URL is a no-op.
3. ``parse_pdf_pages`` — extract text from a bounded page range so a
   200-page prospectus does not explode the LLM context window.
4. ``extract_pdf_metadata`` — page count, title, author, file size.

Design notes
------------
- All four tools return dict payloads. Errors are wrapped as
  ``{"error": "...", "context": "..."}`` — a raising MCP tool would
  kill the stdio subprocess.
- Downloads are cached by SHA-1(url) under ``./data/pdf_cache/``.
  This is intentional: LLMs routinely re-issue identical tool calls
  while reasoning, and cninfo will throttle at ~10 req/min/ip; the
  cache turns pathological loops into a free re-read.
- ``parse_pdf_pages`` caps ``end_page - start_page`` at 20 pages per
  call. The LLM must slice long documents itself — this is the same
  pattern as ``get_stock_price_history`` capping ``days`` at 365.
- ``search_announcements`` talks to cninfo's JSON endpoint **directly
  via httpx**. We originally prototyped this tool on top of
  ``akshare.stock_zh_a_disclosure_report_cninfo``, but that wrapper
  deadlocks inside a fastmcp stdio subprocess on Windows (something
  in its ``requests``/``tqdm`` stack holds the asyncio write loop
  long enough for MCP clients to time out, even when the HTTP round-
  trip is well under a second). Calling the same endpoint ourselves
  keeps the tool fully async, free of ``asyncio.to_thread``, and
  predictable.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pypdf
from fastmcp import FastMCP

mcp = FastMCP("PDFReportServer")

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DEFAULT_CACHE_DIR = Path("./data/pdf_cache").resolve()
"""Default on-disk cache for downloaded PDFs.

Kept as a module-level constant so tests can monkey-patch it, and so
subprocess runs inherit the same path whether invoked from the repo
root or from a ``uv run`` wrapper in a different CWD.
"""

CNINFO_FINALPAGE_FMT = "http://static.cninfo.com.cn/finalpage/{date}/{aid}.PDF"
"""Canonical PDF URL pattern for 巨潮资讯 disclosures.

Verified against a 宁德时代 2023 annual report summary (page-probe ok,
content-type application/pdf, %PDF magic bytes present).
"""

MAX_PAGE_WINDOW = 20
"""Hard upper bound on ``end_page - start_page + 1`` for a single call."""

DOWNLOAD_TIMEOUT_SECONDS = 60
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # 50 MB — prospectuses exist at this size


def _fmt_error(exc: Exception, *, context: str) -> dict[str, Any]:
    return {"error": f"{type(exc).__name__}: {exc}", "context": context}


def _cache_path(url: str, *, cache_dir: Path | None = None) -> Path:
    """SHA-1-keyed cache location for a PDF URL.

    Uses a short hex digest + ``.pdf`` suffix so paths stay readable
    when eyeballing the cache, but collisions are still astronomically
    unlikely for the volumes we deal with (<10^5 docs).
    """
    base = cache_dir or DEFAULT_CACHE_DIR
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return base / f"{digest}.pdf"


def _derive_pdf_url_from_detail(detail_url: str) -> str | None:
    """Convert a cninfo ``/new/disclosure/detail?...`` URL into the
    direct PDF URL.

    Returns ``None`` if the URL doesn't carry the required parameters,
    which happens for the handful of non-PDF announcement types
    (interactive Q&A transcripts, shareholder-meeting live streams).
    """
    try:
        parsed = urlparse(detail_url)
        qs = parse_qs(parsed.query)
        announcement_id = qs.get("announcementId", [None])[0]
        announcement_time = qs.get("announcementTime", [None])[0]
        if not (announcement_id and announcement_time):
            return None
        # Normalize "20240316" → "2024-03-16" if akshare ever returns unpunctuated form
        if re.fullmatch(r"\d{8}", announcement_time):
            announcement_time = (
                f"{announcement_time[0:4]}-{announcement_time[4:6]}-{announcement_time[6:8]}"
            )
        return CNINFO_FINALPAGE_FMT.format(date=announcement_time, aid=announcement_id)
    except Exception:
        return None


# ---------------------------------------------------------------------
# Tool 1: search announcements — cninfo JSON endpoint, direct via httpx
# ---------------------------------------------------------------------
CNINFO_QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
"""POST endpoint for structured announcement search."""

CNINFO_STOCK_INDEX_URLS: dict[str, str] = {
    "沪深京": "http://www.cninfo.com.cn/new/data/szse_stock.json",
    "港股": "http://www.cninfo.com.cn/new/data/hke_stock.json",
    "三板": "http://www.cninfo.com.cn/new/data/gfzr_stock.json",
    "基金": "http://www.cninfo.com.cn/new/data/fund_stock.json",
    "债券": "http://www.cninfo.com.cn/new/data/bond_stock.json",
}
"""``market → stock-index URL`` map.

The query endpoint needs ``stock=<code>,<orgId>``; ``orgId`` is only
available from these JSON dumps, one per market segment. We fetch
lazily and cache in ``_ORGID_CACHE``.
"""

CNINFO_MARKET_COLUMN: dict[str, str] = {
    "沪深京": "szse",
    "港股": "hke",
    "三板": "third",
    "基金": "fund",
    "债券": "bond",
    "预披露": "pre_disclosure",
}
"""``market → column`` payload value (cninfo's own segmentation)."""

CNINFO_CATEGORY_CODES: dict[str, str] = {
    "年报": "category_ndbg_szsh",
    "半年报": "category_bndbg_szsh",
    "一季报": "category_yjdbg_szsh",
    "三季报": "category_sjdbg_szsh",
    "业绩预告": "category_yjygjxz_szsh",
    "权益分派": "category_qyfpxzcs_szsh",
    "董事会": "category_dshgg_szsh",
    "监事会": "category_jshgg_szsh",
    "股东大会": "category_gddh_szsh",
    "日常经营": "category_rcjy_szsh",
    "公司治理": "category_gszl_szsh",
    "中介报告": "category_zj_szsh",
    "首发": "category_sf_szsh",
    "增发": "category_zf_szsh",
    "股权激励": "category_gqjl_szsh",
    "配股": "category_pg_szsh",
    "解禁": "category_jj_szsh",
    "公司债": "category_gszq_szsh",
    "可转债": "category_kzzq_szsh",
    "其他融资": "category_qtrz_szsh",
    "股权变动": "category_gqbd_szsh",
    "补充更正": "category_bcgz_szsh",
    "澄清致歉": "category_cqdq_szsh",
    "风险提示": "category_fxts_szsh",
    "特别处理和退市": "category_tbclts_szsh",
    "退市整理期": "category_tszlq_szsh",
}
"""Category human-label → cninfo category-code. ``"全部"`` maps to an
empty category filter, which we handle at the call site.
"""

_CNINFO_VALID_CATEGORIES: tuple[str, ...] = ("全部", *CNINFO_CATEGORY_CODES.keys())

_ORGID_CACHE: dict[str, dict[str, str]] = {}
"""Module-level cache: ``market → {symbol: orgId}``.

Subsequent calls within the same subprocess lifetime reuse the map
instead of re-fetching the ~600 KB JSON. Populated lazily on first
use per market segment.
"""

_CNINFO_HTTP_TIMEOUT = 20.0

# A realistic User-Agent is important; cninfo rejects python-httpx's
# default UA with a plain HTML error page, which then fails JSON-decode.
_CNINFO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Origin": "http://www.cninfo.com.cn",
    "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
}


async def _load_orgid_map(client: httpx.AsyncClient, market: str) -> dict[str, str]:
    """Return ``{symbol: orgId}`` for one market segment, cached."""
    if market in _ORGID_CACHE:
        return _ORGID_CACHE[market]
    url = CNINFO_STOCK_INDEX_URLS.get(market)
    if url is None:
        raise ValueError(
            f"no stock-index URL for market {market!r}; "
            f"known markets: {list(CNINFO_STOCK_INDEX_URLS)}"
        )
    r = await client.get(url, headers=_CNINFO_HEADERS, timeout=_CNINFO_HTTP_TIMEOUT)
    r.raise_for_status()
    payload = r.json()
    mapping = {item["code"]: item["orgId"] for item in payload.get("stockList", [])}
    _ORGID_CACHE[market] = mapping
    return mapping


_SHANGHAI_TZ = timezone(timedelta(hours=8))


def _format_publish_time(ms_value: Any) -> str:
    """Convert cninfo's millisecond UTC epoch to an Asia/Shanghai date.

    Must use the Shanghai-local date because cninfo organizes PDF
    files under ``finalpage/<YYYY-MM-DD>/...`` using the publisher's
    local date. Returning a UTC date would produce an off-by-one PDF
    URL for any announcement posted before 08:00 Shanghai time (i.e.
    most morning disclosures).
    """
    try:
        return (
            datetime.fromtimestamp(int(ms_value) / 1000, tz=timezone.utc)
            .astimezone(_SHANGHAI_TZ)
            .strftime("%Y-%m-%d")
        )
    except (TypeError, ValueError, OSError):
        return ""


@mcp.tool()
async def search_announcements(
    symbol: str,
    start_date: str,
    end_date: str,
    category: str = "全部",
    market: str = "沪深京",
    limit: int = 20,
) -> dict:
    """Search A-share announcements / research reports on 巨潮资讯.

    Each returned record already carries ``pdf_url`` — the direct URL
    the LLM can feed into ``download_pdf`` without any derivation
    logic of its own.

    Args:
        symbol: 6-digit ticker, e.g. ``"300750"``.
        start_date: ``YYYYMMDD`` inclusive lower bound, e.g.
            ``"20240101"``.
        end_date: ``YYYYMMDD`` inclusive upper bound.
        category: One of the cninfo category labels (``"全部"``,
            ``"年报"``, ``"半年报"``, ``"一季报"``, ``"三季报"``,
            ``"业绩预告"``, ``"风险提示"``, ...). Defaults to
            ``"全部"``. See ``CNINFO_CATEGORY_CODES`` for the full
            list.
        market: Market segment filter. Defaults to ``"沪深京"``
            (mainland A-shares + 北交所).
        limit: Max records to return (default 20, max 100).

    Returns:
        Dictionary with ``symbol``, ``count``, and ``announcements``:
        each element is ``{code, name, title, publish_date,
        detail_url, pdf_url}``. ``pdf_url`` is ``None`` if the
        announcement is non-PDF (rare — interactive Q&A, etc.).
    """
    if category not in _CNINFO_VALID_CATEGORIES:
        return _fmt_error(
            ValueError(
                f"category must be one of {_CNINFO_VALID_CATEGORIES!r}, got {category!r}"
            ),
            context=f"search_announcements(symbol={symbol!r}, category={category!r})",
        )
    if market not in CNINFO_MARKET_COLUMN:
        return _fmt_error(
            ValueError(
                f"market must be one of {tuple(CNINFO_MARKET_COLUMN)!r}, got {market!r}"
            ),
            context=f"search_announcements(symbol={symbol!r}, market={market!r})",
        )
    limit = max(1, min(limit, 100))

    try:
        async with httpx.AsyncClient(timeout=_CNINFO_HTTP_TIMEOUT) as client:
            # Some markets ("预披露") don't need an orgId lookup — the
            # query endpoint accepts stock="" and filters by category.
            stock_item = ""
            if market in CNINFO_STOCK_INDEX_URLS:
                orgid_map = await _load_orgid_map(client, market)
                org_id = orgid_map.get(symbol)
                if org_id is None:
                    return {
                        "symbol": symbol,
                        "count": 0,
                        "announcements": [],
                        "note": f"symbol {symbol!r} not found in {market!r} index",
                    }
                stock_item = f"{symbol},{org_id}"

            se_date = (
                f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}~"
                f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
            )
            category_code = "" if category == "全部" else CNINFO_CATEGORY_CODES[category]

            # ``pageSize=30`` is cninfo's own default; we page until we
            # have ``limit`` records or run out, whichever comes first.
            records: list[dict[str, Any]] = []
            page_num = 1
            page_size = 30
            while len(records) < limit:
                form = {
                    "pageNum": str(page_num),
                    "pageSize": str(page_size),
                    "column": CNINFO_MARKET_COLUMN[market],
                    "tabName": "fulltext",
                    "plate": "",
                    "stock": stock_item,
                    "searchkey": "",
                    "secid": "",
                    "category": category_code,
                    "trade": "",
                    "seDate": se_date,
                    "sortName": "",
                    "sortType": "",
                    "isHLtitle": "true",
                }
                r = await client.post(
                    CNINFO_QUERY_URL, data=form, headers=_CNINFO_HEADERS
                )
                r.raise_for_status()
                payload = r.json()
                anns = payload.get("announcements") or []
                if not anns:
                    break
                for a in anns:
                    code = str(a.get("secCode", symbol))
                    name = str(a.get("secName", ""))
                    title = str(a.get("announcementTitle", ""))
                    ts_ms = a.get("announcementTime", "")
                    publish_date = _format_publish_time(ts_ms)
                    ann_id = a.get("announcementId", "")
                    org = a.get("orgId", "")
                    detail_url = (
                        f"http://www.cninfo.com.cn/new/disclosure/detail?stockCode={code}"
                        f"&announcementId={ann_id}&orgId={org}&announcementTime={publish_date}"
                    )
                    pdf_url = (
                        CNINFO_FINALPAGE_FMT.format(date=publish_date, aid=ann_id)
                        if (publish_date and ann_id)
                        else None
                    )
                    records.append(
                        {
                            "code": code,
                            "name": name,
                            "title": title,
                            "publish_date": publish_date,
                            "detail_url": detail_url,
                            "pdf_url": pdf_url,
                        }
                    )
                    if len(records) >= limit:
                        break
                total = int(payload.get("totalAnnouncement", 0) or 0)
                if page_num * page_size >= total:
                    break
                page_num += 1

            return {"symbol": symbol, "count": len(records), "announcements": records}
    except Exception as e:
        return _fmt_error(
            e,
            context=(
                f"search_announcements(symbol={symbol!r}, "
                f"start_date={start_date!r}, end_date={end_date!r})"
            ),
        )


# ---------------------------------------------------------------------
# Tool 2: download a PDF into the on-disk cache
# ---------------------------------------------------------------------
async def _download_bytes(url: str) -> bytes:
    async with httpx.AsyncClient(
        timeout=DOWNLOAD_TIMEOUT_SECONDS,
        follow_redirects=True,
    ) as cli:
        async with cli.stream("GET", url) as r:
            r.raise_for_status()
            total = 0
            chunks: list[bytes] = []
            async for chunk in r.aiter_bytes(chunk_size=64 * 1024):
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise ValueError(
                        f"PDF exceeds {MAX_DOWNLOAD_BYTES} byte hard limit "
                        f"(got >{total}); aborting download of {url!r}"
                    )
                chunks.append(chunk)
            return b"".join(chunks)


@mcp.tool()
async def download_pdf(pdf_url: str) -> dict:
    """Download a PDF into the on-disk cache; return local path.

    Cache hits are free — the same URL resolves to the same path
    regardless of how many times the LLM re-calls this tool while
    reasoning. A successful download is verified by the ``%PDF``
    magic bytes before being written, so we never leave a truncated
    or HTML-error-page file on disk claiming to be a PDF.

    Args:
        pdf_url: Absolute URL to a PDF (typically a
            ``static.cninfo.com.cn/finalpage/...`` link returned from
            ``search_announcements``).

    Returns:
        Dictionary with ``pdf_url``, ``local_path`` (absolute),
        ``size_bytes``, and ``from_cache`` (bool indicating whether
        we reused an existing file). On failure returns
        ``{"error": ..., "context": ...}``.
    """
    if not pdf_url or not pdf_url.lower().startswith(("http://", "https://")):
        return _fmt_error(
            ValueError(f"pdf_url must be an absolute http(s) URL, got {pdf_url!r}"),
            context="download_pdf()",
        )

    cache_path = _cache_path(pdf_url)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return _fmt_error(e, context=f"download_pdf({pdf_url!r}): mkdir cache")

    if cache_path.exists() and cache_path.stat().st_size > 0:
        return {
            "pdf_url": pdf_url,
            "local_path": str(cache_path),
            "size_bytes": cache_path.stat().st_size,
            "from_cache": True,
        }

    try:
        data = await _download_bytes(pdf_url)
    except Exception as e:
        return _fmt_error(e, context=f"download_pdf({pdf_url!r})")

    if not data.startswith(b"%PDF"):
        return _fmt_error(
            ValueError(
                f"response did not start with %PDF magic bytes "
                f"(got {data[:16]!r}); upstream may have returned an HTML error page"
            ),
            context=f"download_pdf({pdf_url!r}): magic-byte check",
        )

    try:
        cache_path.write_bytes(data)
    except Exception as e:
        return _fmt_error(e, context=f"download_pdf({pdf_url!r}): write cache")

    return {
        "pdf_url": pdf_url,
        "local_path": str(cache_path),
        "size_bytes": len(data),
        "from_cache": False,
    }


# ---------------------------------------------------------------------
# Tool 3: parse a bounded page range
# ---------------------------------------------------------------------
@mcp.tool()
async def parse_pdf_pages(
    local_path: str,
    start_page: int = 1,
    end_page: int = 5,
) -> dict:
    """Extract text from ``[start_page, end_page]`` (inclusive, 1-indexed).

    The LLM should **never** request the whole document in one call —
    a 200-page prospectus easily exceeds any model's context window.
    For long reports, the intended pattern is:

    1. ``extract_pdf_metadata`` → learn ``num_pages``
    2. ``parse_pdf_pages(start, end)`` in multiple calls, 20 pages
       each, scanning for the section the user asked about.

    Args:
        local_path: Absolute path previously returned by
            ``download_pdf``.
        start_page: First page, 1-indexed (default 1).
        end_page: Last page, inclusive, 1-indexed (default 5).

    Returns:
        Dictionary with ``local_path``, ``requested_range``
        (``{start, end}``), ``total_pages``, and ``pages`` —
        each entry is ``{page_number, char_count, text}``. Pages
        outside the document's actual range are silently skipped.
    """
    if start_page < 1 or end_page < start_page:
        return _fmt_error(
            ValueError(
                f"invalid range: start_page={start_page}, end_page={end_page}"
            ),
            context="parse_pdf_pages()",
        )
    if (end_page - start_page + 1) > MAX_PAGE_WINDOW:
        return _fmt_error(
            ValueError(
                f"page window (end - start + 1 = {end_page - start_page + 1}) "
                f"exceeds {MAX_PAGE_WINDOW}; please call the tool multiple times "
                f"for long documents"
            ),
            context="parse_pdf_pages()",
        )

    path = Path(local_path)
    if not path.exists():
        return _fmt_error(
            FileNotFoundError(f"no such file: {path}"),
            context=f"parse_pdf_pages({local_path!r})",
        )

    def _call() -> dict[str, Any]:
        with path.open("rb") as fh:
            reader = pypdf.PdfReader(fh)
            total = len(reader.pages)
            pages_out: list[dict[str, Any]] = []
            for p in range(start_page, min(end_page, total) + 1):
                # pypdf pages are 0-indexed internally
                page_text = reader.pages[p - 1].extract_text() or ""
                pages_out.append(
                    {
                        "page_number": p,
                        "char_count": len(page_text),
                        "text": page_text,
                    }
                )
        return {
            "local_path": str(path),
            "requested_range": {"start": start_page, "end": end_page},
            "total_pages": total,
            "pages": pages_out,
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(
            e,
            context=(
                f"parse_pdf_pages(local_path={local_path!r}, "
                f"start_page={start_page}, end_page={end_page})"
            ),
        )


# ---------------------------------------------------------------------
# Tool 4: document-level metadata
# ---------------------------------------------------------------------
@mcp.tool()
async def extract_pdf_metadata(local_path: str) -> dict:
    """Return page count, title, author, creator, and file size.

    Cheap — does not decode any page content. Useful as the first
    call on an unfamiliar document to decide how to slice it.

    Args:
        local_path: Absolute path from ``download_pdf``.

    Returns:
        Dictionary with ``local_path``, ``num_pages``, ``size_bytes``,
        and ``metadata`` (a dict of any of
        ``{title, author, subject, creator, producer, creation_date,
        mod_date}`` that the PDF happened to embed). Missing fields
        are simply absent rather than ``None``.
    """
    path = Path(local_path)
    if not path.exists():
        return _fmt_error(
            FileNotFoundError(f"no such file: {path}"),
            context=f"extract_pdf_metadata({local_path!r})",
        )

    def _call() -> dict[str, Any]:
        with path.open("rb") as fh:
            reader = pypdf.PdfReader(fh)
            num_pages = len(reader.pages)
            raw_meta = dict(reader.metadata) if reader.metadata else {}
        # Strip the PDF's leading-slash key convention for readability.
        key_map = {
            "/Title": "title",
            "/Author": "author",
            "/Subject": "subject",
            "/Creator": "creator",
            "/Producer": "producer",
            "/CreationDate": "creation_date",
            "/ModDate": "mod_date",
        }
        nice_meta: dict[str, Any] = {}
        for k, v in raw_meta.items():
            nice_key = key_map.get(k, k.lstrip("/").lower() if isinstance(k, str) else str(k))
            nice_meta[nice_key] = str(v) if not isinstance(v, (datetime, int, float, bool)) else v
        return {
            "local_path": str(path),
            "num_pages": num_pages,
            "size_bytes": path.stat().st_size,
            "metadata": nice_meta,
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        return _fmt_error(e, context=f"extract_pdf_metadata({local_path!r})")


if __name__ == "__main__":
    mcp.run(transport="stdio")
