"""MCP Server — 美股 SEC EDGAR 披露（与巨潮 ``pdf_report_server`` 平行隔离）。

默认表单
--------
- 普通股 / ADR：``10-K`` / ``10-Q`` / ``8-K`` / ``DEF 14A``（修订件如 ``10-K/A`` 也会匹配）
- ETF / 注册投资公司：``NPORT-P``（月度持仓）、``N-CSR`` / ``N-CSRS``（股东报告）、
  ``485BPOS``（招股书更新；``485APOS`` 作别名匹配）

工具
----
1. ``resolve_cik`` — ticker / CIK → 规范化 CIK
2. ``search_filings`` — 按 ticker/CIK + 表单类型列出近期披露
3. ``download_filing`` — 下载主文档到 ``./data/edgar_cache/``（按 URL 哈希缓存）
4. ``extract_filing_metadata`` — 本地文件元数据（类型 / 大小 / PDF 页数）
5. ``parse_filing_text`` — 有界正文提取（PDF 按页；HTML/TXT 按字符窗口）

设计说明
--------
- 数据来源：``company_tickers.json`` + ``data.sec.gov/submissions`` + Archives 文档。
- SEC 要求每个请求带可识别的 ``User-Agent``（可用环境变量 ``SEC_USER_AGENT`` 覆盖）。
- 错误一律 ``{"error": "...", "context": "..."}``，不抛异常以免弄死 stdio。
- 搜索结果走 ``cached_tool`` TTL（namespace=``us_filing``）。
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import logging
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx
from fastmcp import FastMCP

from research_agent.cache import TTL_DAILY, TTL_LONG, cached_tool

logger = logging.getLogger("us_filing_server")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

mcp = FastMCP("UsFilingServer")

DEFAULT_CACHE_DIR = Path("./data/edgar_cache").resolve()
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

# 普通股 + ETF/基金专属表单；查 AAPL 时 ETF 表单通常无命中，查 QQQ 时则不再「稀疏」。
DEFAULT_FORMS = (
    "10-K",
    "10-Q",
    "8-K",
    "DEF 14A",
    "NPORT-P",
    "N-CSR",
    "N-CSRS",
    "485BPOS",
)
DEFAULT_FORMS_CSV = ",".join(DEFAULT_FORMS)

# 用户口头名 / EDGAR 实际 form 码互认（展开进 wanted 集合后再精确/修订匹配）
_FORM_EQUIV_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"N-PORT", "NPORT", "NPORT-P", "NPORT-EX"}),
    frozenset({"N-CSR", "N-CSRS"}),
    frozenset({"485BPOS", "485APOS"}),
)

MAX_PAGE_WINDOW = 20
MAX_CHAR_WINDOW = 12_000
DOWNLOAD_TIMEOUT_SECONDS = 60
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
HTTP_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

_TICKER_CACHE: dict[str, dict[str, str]] | None = None


def _sec_headers() -> dict[str, str]:
    ua = os.environ.get(
        "SEC_USER_AGENT",
        "research-agent/0.1 (edgar-poc; contact@example.com)",
    ).strip()
    return {
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json, text/html, application/pdf, */*",
    }


def _fmt_error(exc: Exception, *, context: str) -> dict[str, Any]:
    logger.error("[%s] %s: %s", context, type(exc).__name__, exc)
    return {"error": f"{type(exc).__name__}: {exc}", "context": context}


def _pad_cik(cik: str | int) -> str:
    digits = re.sub(r"\D", "", str(cik))
    if not digits:
        raise ValueError(f"invalid CIK: {cik!r}")
    return digits.zfill(10)[-10:]


def _cik_int_str(cik10: str) -> str:
    return str(int(cik10))


def _normalize_form(form: str) -> str:
    return re.sub(r"\s+", " ", form.strip().upper())


def _expand_wanted_forms(wanted: set[str]) -> set[str]:
    """把别名组并入过滤集，例如 ``N-PORT`` → 同时接受 ``NPORT-P``。"""
    expanded = {_normalize_form(x) for x in wanted}
    for group in _FORM_EQUIV_GROUPS:
        norms = {_normalize_form(x) for x in group}
        if expanded & norms:
            expanded |= norms
    return expanded


def _form_matches(actual: str, wanted: set[str]) -> bool:
    a = _normalize_form(actual)
    if a in wanted:
        return True
    # 允许 10-K/A 命中 10-K；NPORT-P/A 命中 NPORT-P
    base = a.split("/")[0]
    return base in wanted


def _accession_nodash(accession: str) -> str:
    return accession.replace("-", "")


def _document_url(*, cik10: str, accession: str, primary_document: str) -> str:
    return (
        f"{ARCHIVES_BASE}/{_cik_int_str(cik10)}/"
        f"{_accession_nodash(accession)}/{primary_document.lstrip('/')}"
    )


def _cache_path(url: str, *, cache_dir: Path | None = None) -> Path:
    base = cache_dir or DEFAULT_CACHE_DIR
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    # 保留扩展名便于后续解析分支
    suffix = Path(url.split("?")[0]).suffix.lower() or ".bin"
    if len(suffix) > 8:
        suffix = ".bin"
    return base / f"{digest}{suffix}"


async def _http_get_json(url: str) -> Any:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=_sec_headers()) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()


async def _http_get_bytes(url: str) -> bytes:
    async with (
        httpx.AsyncClient(
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
            headers=_sec_headers(),
            follow_redirects=True,
        ) as client,
        client.stream("GET", url) as r,
    ):
        r.raise_for_status()
        total = 0
        chunks: list[bytes] = []
        async for chunk in r.aiter_bytes(chunk_size=64 * 1024):
            total += len(chunk)
            if total > MAX_DOWNLOAD_BYTES:
                raise ValueError(
                    f"file exceeds {MAX_DOWNLOAD_BYTES} byte hard limit "
                    f"(got >{total}); aborting download of {url!r}"
                )
            chunks.append(chunk)
        return b"".join(chunks)


async def _load_ticker_map() -> dict[str, dict[str, str]]:
    """ticker → {cik10, name}。"""
    global _TICKER_CACHE
    if _TICKER_CACHE is not None:
        return _TICKER_CACHE

    data = await _http_get_json(COMPANY_TICKERS_URL)
    mapping: dict[str, dict[str, str]] = {}
    if isinstance(data, dict):
        rows = data.values()
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        cik_raw = row.get("cik_str") if row.get("cik_str") is not None else row.get("cik")
        name = str(row.get("title") or row.get("name") or "").strip()
        if not ticker or cik_raw is None:
            continue
        mapping[ticker] = {"cik10": _pad_cik(cik_raw), "name": name}
    _TICKER_CACHE = mapping
    return mapping


def _parse_identifier(identifier: str) -> tuple[str, str]:
    """返回 (kind, value)：kind ∈ {cik, ticker}。"""
    s = identifier.strip()
    if not s:
        raise ValueError("identifier 不能为空")
    if re.fullmatch(r"\d{1,10}", s):
        return "cik", _pad_cik(s)
    return "ticker", s.upper()


async def _resolve_cik_impl(identifier: str) -> dict[str, Any]:
    kind, value = _parse_identifier(identifier)
    if kind == "cik":
        return {
            "identifier": identifier.strip(),
            "cik10": value,
            "cik": _cik_int_str(value),
            "ticker": None,
            "name": None,
            "source": "input_cik",
            "source_url": COMPANY_TICKERS_URL,
        }
    mapping = await _load_ticker_map()
    hit = mapping.get(value)
    if not hit:
        return {
            "error": f"未找到 ticker {value!r} 对应的 CIK",
            "context": f"resolve_cik({identifier!r})",
        }
    return {
        "identifier": identifier.strip(),
        "cik10": hit["cik10"],
        "cik": _cik_int_str(hit["cik10"]),
        "ticker": value,
        "name": hit["name"],
        "source": "company_tickers.json",
        "source_url": COMPANY_TICKERS_URL,
    }


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip += 1
        if tag in {"p", "br", "div", "tr", "li", "h1", "h2", "h3", "h4"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip > 0:
            self._skip -= 1
        if tag in {"p", "div", "tr", "li", "h1", "h2", "h3", "h4"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = data.strip()
        if text:
            self._chunks.append(text + " ")

    def text(self) -> str:
        raw = "".join(self._chunks)
        raw = html.unescape(raw)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _html_to_text(raw: str) -> str:
    parser = _HTMLTextExtractor()
    try:
        parser.feed(raw)
        parser.close()
    except Exception:  # noqa: BLE001
        return re.sub(r"<[^>]+>", " ", raw)
    return parser.text()


def _detect_kind(data: bytes, *, path: Path | None = None) -> str:
    if data.startswith(b"%PDF"):
        return "pdf"
    suffix = (path.suffix.lower() if path else "") or ""
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".htm", ".html", ".xhtml"}:
        return "html"
    # heuristic
    head = data[:200].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<html" in head:
        return "html"
    return "text"


# ---------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------


@mcp.tool()
@cached_tool(ttl=TTL_LONG, namespace="us_filing")
async def resolve_cik(identifier: str) -> dict:
    """将美股 ticker 或 CIK 解析为 10 位 CIK。

    Args:
        identifier: 如 ``AAPL``、``TSLA``、``320193``、``0000320193``。
    """
    try:
        return await _resolve_cik_impl(identifier)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context=f"resolve_cik({identifier!r})")


@mcp.tool()
@cached_tool(ttl=TTL_DAILY, namespace="us_filing")
async def search_filings(
    identifier: str,
    forms: str = DEFAULT_FORMS_CSV,
    limit: int = 10,
) -> dict:
    """按 ticker/CIK 搜索 SEC EDGAR 近期披露列表。

    Args:
        identifier: ticker（``AAPL``）或 CIK。
        forms: 逗号分隔表单类型。默认含普通股 ``10-K/10-Q/8-K/DEF 14A`` 与 ETF
            ``NPORT-P/N-CSR/N-CSRS/485BPOS``（``N-PORT`` 等别名可匹配）。
        limit: 返回条数（1–40）。
    """
    limit = max(1, min(int(limit), 40))
    raw_wanted = {_normalize_form(x) for x in forms.split(",") if x.strip()} or set(DEFAULT_FORMS)
    wanted = _expand_wanted_forms(raw_wanted)

    try:
        resolved = await _resolve_cik_impl(identifier)
        if "error" in resolved:
            return resolved
        cik10 = str(resolved["cik10"])
        payload = await _http_get_json(SUBMISSIONS_URL.format(cik10=cik10))
        recent = (payload.get("filings") or {}).get("recent") or {}
        accessions = recent.get("accessionNumber") or []
        filing_dates = recent.get("filingDate") or []
        form_list = recent.get("form") or []
        primaries = recent.get("primaryDocument") or []
        descriptions = recent.get("primaryDocDescription") or []
        report_dates = recent.get("reportDate") or []

        n = min(
            len(accessions),
            len(filing_dates),
            len(form_list),
            len(primaries),
        )
        filings: list[dict[str, Any]] = []
        for i in range(n):
            form = str(form_list[i] or "")
            if not _form_matches(form, wanted):
                continue
            accession = str(accessions[i])
            primary = str(primaries[i] or "")
            if not primary:
                continue
            doc_url = _document_url(cik10=cik10, accession=accession, primary_document=primary)
            filings.append(
                {
                    "accession": accession,
                    "form": form,
                    "filing_date": filing_dates[i],
                    "report_date": report_dates[i] if i < len(report_dates) else None,
                    "primary_document": primary,
                    "description": descriptions[i] if i < len(descriptions) else "",
                    "document_url": doc_url,
                    "company": payload.get("name"),
                    "ticker": (payload.get("tickers") or [None])[0],
                    "cik10": cik10,
                }
            )
            if len(filings) >= limit:
                break

        return {
            "identifier": identifier.strip(),
            "cik10": cik10,
            "company": payload.get("name"),
            "tickers": payload.get("tickers") or [],
            "forms_filter": sorted(wanted),
            "filings": filings,
            "count": len(filings),
            "source": "data.sec.gov/submissions",
            "source_url": SUBMISSIONS_URL.format(cik10=cik10),
        }
    except Exception as e:  # noqa: BLE001
        return _fmt_error(
            e,
            context=f"search_filings(identifier={identifier!r}, forms={forms!r})",
        )


@mcp.tool()
async def download_filing(document_url: str) -> dict:
    """下载 EDGAR 主文档到磁盘缓存；返回本地路径。

    Args:
        document_url: ``search_filings`` 返回的 ``document_url``（sec.gov Archives）。
    """
    if not document_url or not document_url.lower().startswith(("http://", "https://")):
        return _fmt_error(
            ValueError(f"document_url must be absolute http(s) URL, got {document_url!r}"),
            context="download_filing()",
        )
    if "sec.gov" not in document_url.lower():
        return _fmt_error(
            ValueError("document_url 必须是 sec.gov 域名（EDGAR Archives）"),
            context="download_filing()",
        )

    cache_path = _cache_path(document_url)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context=f"download_filing({document_url!r}): mkdir cache")

    if cache_path.exists() and cache_path.stat().st_size > 0:
        return {
            "document_url": document_url,
            "local_path": str(cache_path),
            "size_bytes": cache_path.stat().st_size,
            "from_cache": True,
            "kind": _detect_kind(cache_path.read_bytes()[:64], path=cache_path),
        }

    try:
        data = await _http_get_bytes(document_url)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context=f"download_filing({document_url!r})")

    # 常见：SEC 用 HTML 错误页顶替
    if data.lstrip().startswith(b"<") and b"Request Rate Threshold" in data[:2000]:
        return _fmt_error(
            RuntimeError("SEC rate limit / blocked response (HTML)"),
            context=f"download_filing({document_url!r})",
        )

    try:
        cache_path.write_bytes(data)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context=f"download_filing({document_url!r}): write cache")

    return {
        "document_url": document_url,
        "local_path": str(cache_path),
        "size_bytes": len(data),
        "from_cache": False,
        "kind": _detect_kind(data, path=cache_path),
    }


@mcp.tool()
async def extract_filing_metadata(local_path: str) -> dict:
    """返回本地披露文件的类型、大小；PDF 另含页数与嵌入元数据。

    Args:
        local_path: ``download_filing`` 返回的绝对路径。
    """
    path = Path(local_path)
    if not path.exists():
        return _fmt_error(
            FileNotFoundError(f"没有这个文件: {path}"),
            context=f"extract_filing_metadata({local_path!r})",
        )

    def _call() -> dict[str, Any]:
        data = path.read_bytes()
        kind = _detect_kind(data, path=path)
        out: dict[str, Any] = {
            "local_path": str(path),
            "size_bytes": path.stat().st_size,
            "kind": kind,
            "suffix": path.suffix.lower(),
        }
        if kind == "pdf":
            import pypdf

            with path.open("rb") as fh:
                reader = pypdf.PdfReader(fh)
                out["num_pages"] = len(reader.pages)
                meta = reader.metadata or {}
                cleaned = {}
                for key in ("/Title", "/Author", "/Subject", "/Creator", "/Producer"):
                    val = meta.get(key) if hasattr(meta, "get") else None
                    if val:
                        cleaned[key.lstrip("/").lower()] = str(val)
                if cleaned:
                    out["metadata"] = cleaned
        else:
            text = data.decode("utf-8", errors="ignore")
            if kind == "html":
                text = _html_to_text(text)
            out["char_count"] = len(text)
            out["preview"] = text[:400]
        return out

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(e, context=f"extract_filing_metadata({local_path!r})")


@mcp.tool()
async def parse_filing_text(
    local_path: str,
    start_page: int = 1,
    end_page: int = 5,
    start_char: int = 0,
    max_chars: int = 8000,
) -> dict:
    """提取披露正文的有界窗口。

    - PDF：使用 ``start_page``/``end_page``（1-indexed，单次最多 20 页）
    - HTML/TXT：使用 ``start_char``/``max_chars``（单次最多 12000 字符）

    Args:
        local_path: ``download_filing`` 返回的路径。
        start_page: PDF 起始页（默认 1）。
        end_page: PDF 结束页（默认 5）。
        start_char: 文本起始偏移（默认 0）。
        max_chars: 文本窗口长度（默认 8000）。
    """
    path = Path(local_path)
    if not path.exists():
        return _fmt_error(
            FileNotFoundError(f"没有这个文件: {path}"),
            context=f"parse_filing_text({local_path!r})",
        )

    max_chars = max(1, min(int(max_chars), MAX_CHAR_WINDOW))
    start_char = max(0, int(start_char))

    def _call() -> dict[str, Any]:
        data = path.read_bytes()
        kind = _detect_kind(data, path=path)

        if kind == "pdf":
            if start_page < 1 or end_page < start_page:
                raise ValueError(f"invalid range: start_page={start_page}, end_page={end_page}")
            if (end_page - start_page + 1) > MAX_PAGE_WINDOW:
                raise ValueError(
                    f"page window exceeds {MAX_PAGE_WINDOW}; call multiple times for long docs"
                )
            import pypdf

            with path.open("rb") as fh:
                reader = pypdf.PdfReader(fh)
                total = len(reader.pages)
                pages_out: list[dict[str, Any]] = []
                for p in range(start_page, min(end_page, total) + 1):
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
                "kind": "pdf",
                "requested_range": {"start": start_page, "end": end_page},
                "total_pages": total,
                "pages": pages_out,
            }

        raw = data.decode("utf-8", errors="ignore")
        text = _html_to_text(raw) if kind == "html" else raw
        window = text[start_char : start_char + max_chars]
        return {
            "local_path": str(path),
            "kind": kind,
            "total_chars": len(text),
            "requested_range": {
                "start_char": start_char,
                "max_chars": max_chars,
                "end_char": start_char + len(window),
            },
            "text": window,
            "char_count": len(window),
            "truncated": start_char + len(window) < len(text),
        }

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:  # noqa: BLE001
        return _fmt_error(
            e,
            context=(
                f"parse_filing_text(local_path={local_path!r}, "
                f"start_page={start_page}, end_page={end_page}, "
                f"start_char={start_char}, max_chars={max_chars})"
            ),
        )


def reset_ticker_cache_for_tests() -> None:
    """测试辅助：清空 ticker→CIK 内存缓存。"""
    global _TICKER_CACHE
    _TICKER_CACHE = None


if __name__ == "__main__":
    mcp.run(transport="stdio")
