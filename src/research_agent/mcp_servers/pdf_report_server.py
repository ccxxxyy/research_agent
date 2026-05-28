"""MCP Server — A 股公告 / 研报 PDF（巨潮资讯）。

本模块是金融 Agent 的 文档层。
它是 ``knowledge_server`` 数据导入管道的上游数据源（``knowledge_ingest_pdf`` 通常使用本模块 ``download_pdf`` 返回的路径进行调用）。

为什么选择巨潮资讯
------------------
巨潮资讯（``cninfo.com.cn``）是中国证监会指定的深沪上市公司官方信息披露门户。
所有年报（等同于 10-K）、季报（等同于 10-Q）、紧急公告及承销商招股说明书均存放于此，且 URL 模式稳定：

    http://static.cninfo.com.cn/finalpage/<YYYY-MM-DD>/<announcementId>.PDF

这种可预测性使得基于工具的工作流成为可能——无需抓取搜索界面；
巨潮提供了结构化的 ``/new/hisAnnouncement/query`` JSON 端点，只需从查询结果中推导出 PDF URL 即可。

提供的工具
----------
1. ``search_announcements`` — 根据股票代码列出公告，每条结果已附带推导好的 PDF URL。
2. ``download_pdf`` — 将单个 PDF 下载到基于内容哈希的缓存目录中。重复下载同一 URL 不会产生额外操作。
3. ``parse_pdf_pages`` — 从有限的页码范围中提取文本，避免 200 页的招股说明书撑爆 LLM 上下文窗口。
4. ``extract_pdf_metadata`` — 返回页数、标题、作者、文件大小。

设计说明
--------
- 四个工具均返回 dict。错误以 ``{"error": "...", "context": "..."}`` 形式包装——在 MCP 工具中直接抛出异常会终止 stdio 子进程。
- 下载按 SHA-1(url) 缓存于 ``./data/pdf_cache/``。这是有意为之：LLM 在推理过程中经常重复发出相同的工具调用，
  而巨潮会以约 10 次请求/分钟/IP 的速率进行限流；缓存机制将病态循环变为免费的重复读取。
- ``parse_pdf_pages`` 限制每次调用最多 ``end_page - start_page`` 为 20 页。
  LLM 需要自行对长文档进行分片——这与 ``get_stock_price_history`` 将``days`` 限制为 365 的设计模式一致。
- ``search_announcements`` 通过 httpx 直接访问巨潮的 JSON 端点。
  最初在 ``akshare.stock_zh_a_disclosure_report_cninfo`` 之上构建了该工具原型，用 akshare 的 stock_zh_a_disclosure_report_cninfo 封装函数（内部用 requests + tqdm），
  但该封装在 Windows 上的 fastmcp stdio 子进程中会死锁
  （其 ``requests``/``tqdm`` 调用栈中的某些内容会阻塞 asyncio 写循环足够长的时间，导致 MCP 客户端超时，即使 HTTP 往返时间远低于一秒）。
  自行调用相同端点使工具完全异步，不用 akshare 封装，直接用 httpx.AsyncClient 调巨潮的 JSON 端点，无需 ``asyncio.to_thread``，且行为可预测。
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pypdf
from fastmcp import FastMCP

mcp = FastMCP("PDFReportServer")

# ---------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------
DEFAULT_CACHE_DIR = Path("./data/pdf_cache").resolve()
"""已下载 PDF 的默认磁盘缓存目录。

设为模块级常量，以便测试可以猴子补丁（monkey-patch）覆盖，同时子进程无论是从仓库根目录还是从不同 CWD 的 ``uv run`` 包装器启动，都能继承相同路径。
"""

CNINFO_FINALPAGE_FMT = "http://static.cninfo.com.cn/finalpage/{date}/{aid}.PDF"
"""巨潮资讯披露文件的标准 PDF URL 模式。

已通过宁德时代 2023 年报摘要验证（页面探测正常，content-type 为 application/pdf，%PDF 魔数字节存在）。
"""

MAX_PAGE_WINDOW = 20
"""单次调用 ``end_page - start_page + 1`` 的硬性上限。"""

DOWNLOAD_TIMEOUT_SECONDS = 60
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # 50 MB — 招股说明书可达此大小


def _fmt_error(exc: Exception, *, context: str) -> dict[str, Any]:
    return {"error": f"{type(exc).__name__}: {exc}", "context": context}


def _cache_path(url: str, *, cache_dir: Path | None = None) -> Path:
    """根据 PDF URL 的 SHA-1 哈希生成缓存路径。

    使用短十六进制摘要 + ``.pdf`` 后缀，以保持目视检查缓存时的可读性，
    同时在处理的数据量级（< 10^5 份文档）下，碰撞概率极低可忽略。
    """
    base = cache_dir or DEFAULT_CACHE_DIR
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return base / f"{digest}.pdf"


def _derive_pdf_url_from_detail(detail_url: str) -> str | None:
    """将 cninfo ``/new/disclosure/detail?...`` URL 转换为直接 PDF URL。

    当 URL 不携带所需参数时返回 ``None``，这发生在少数非 PDF 公告类型（互动问答记录、股东大会直播等）上。
    """
    try:
        parsed = urlparse(detail_url)
        qs = parse_qs(parsed.query)
        announcement_id = qs.get("announcementId", [None])[0]
        announcement_time = qs.get("announcementTime", [None])[0]
        if not (announcement_id and announcement_time):
            return None
        # 将 "20240316" 规范化为 "2024-03-16"，以防 akshare 返回无标点形式
        if re.fullmatch(r"\d{8}", announcement_time):
            announcement_time = (
                f"{announcement_time[0:4]}-{announcement_time[4:6]}-{announcement_time[6:8]}"
            )
        return CNINFO_FINALPAGE_FMT.format(date=announcement_time, aid=announcement_id)
    except Exception:
        return None


# ---------------------------------------------------------------------
# 工具 1: 搜索公告 — 通过 httpx 直接访问 cninfo JSON 端点
# ---------------------------------------------------------------------
CNINFO_QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
"""结构化公告搜索的 POST 端点。"""

CNINFO_STOCK_INDEX_URLS: dict[str, str] = {
    "沪深京": "http://www.cninfo.com.cn/new/data/szse_stock.json",
    "港股": "http://www.cninfo.com.cn/new/data/hke_stock.json",
    "三板": "http://www.cninfo.com.cn/new/data/gfzr_stock.json",
    "基金": "http://www.cninfo.com.cn/new/data/fund_stock.json",
    "债券": "http://www.cninfo.com.cn/new/data/bond_stock.json",
}
"""``market → 股票索引 URL`` 映射。

查询端点需要 ``stock=<code>,<orgId>``；
``orgId`` 仅可从这些 JSON 转储获取，每个市场分段一个。
延迟获取并缓存到 ``_ORGID_CACHE``。
"""

CNINFO_MARKET_COLUMN: dict[str, str] = {
    "沪深京": "szse",
    "港股": "hke",
    "三板": "third",
    "基金": "fund",
    "债券": "bond",
    "预披露": "pre_disclosure",
}
"""``market → column`` 请求负载值（cninfo 自有的分段方式）。"""

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
"""分类人类可读标签 → cninfo 分类代码。
``"全部"`` 映射为空分类过滤器，在调用处处理。
"""

_CNINFO_VALID_CATEGORIES: tuple[str, ...] = ("全部", *CNINFO_CATEGORY_CODES.keys())

_ORGID_CACHE: dict[str, dict[str, str]] = {}
"""模块级缓存：``market → {symbol: orgId}``。

同一子进程生命周期内的后续调用复用此映射，而非重新获取约 600 KB 的 JSON。
按市场分段在首次使用时延迟填充。
"""

_CNINFO_HTTP_TIMEOUT = 20.0

# 巨潮资讯的服务器会检查 HTTP 请求的 User-Agent 头，即 cninfo 会用纯 HTML 错误页拒绝 python-httpx 的默认 UA，导致 JSON 解码失败。
# 伪装成 Chrome 浏览器，服务器就正常返回 JSON 了。
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
    """返回某个市场分段的 ``{symbol: orgId}``，带缓存。"""
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
    """将 cninfo 的毫秒级 UTC 时间戳转换为亚洲/上海日期。

    必须使用上海本地日期，
    因为 cninfo 以发布者本地日期组织 PDF 文件路径 ``finalpage/<YYYY-MM-DD>/...``。返回 UTC 日期会导致 08:00 上海时间之前发布的公告产生偏差一天的 PDF URL。
    """
    try:
        return (
            datetime.fromtimestamp(int(ms_value) / 1000, tz=UTC)
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
    """在巨潮资讯上搜索 A 股公告 / 研报。

    每条返回记录已携带 ``pdf_url`` — LLM 可直接将其传给 ``download_pdf``，无需自行推导逻辑。

    Args:
        symbol: 6 位代码，如 ``"300750"``。
        start_date: ``YYYYMMDD`` 格式的起始日期（含），如 ``"20240101"``。
        end_date: ``YYYYMMDD`` 格式的结束日期（含）。
        category: cninfo 分类标签之一（``"全部"``、``"年报"``、``"半年报"``、``"一季报"``、``"三季报"``、``"业绩预告"``、``"风险提示"``……）。默认 ``"全部"``。
            完整列表见``CNINFO_CATEGORY_CODES``。
        market: 市场分段过滤器。默认 ``"沪深京"``（大陆 A 股 + 北交所）。
        limit: 最大返回记录数（默认 20，最大 100）。

    Returns:
        包含 ``symbol``、``count`` 和 ``announcements`` 的字典：每个元素为
        ``{code, name, title, publish_date, detail_url, pdf_url}``。
        若公告非 PDF 类型（罕见 — 互动问答等），则 ``pdf_url`` 为 ``None``。
    """
    if category not in _CNINFO_VALID_CATEGORIES:
        return _fmt_error(
            ValueError(f"category must be one of {_CNINFO_VALID_CATEGORIES!r}, got {category!r}"),
            context=f"search_announcements(symbol={symbol!r}, category={category!r})",
        )
    if market not in CNINFO_MARKET_COLUMN:
        return _fmt_error(
            ValueError(f"market must be one of {tuple(CNINFO_MARKET_COLUMN)!r}, got {market!r}"),
            context=f"search_announcements(symbol={symbol!r}, market={market!r})",
        )
    limit = max(1, min(limit, 100))

    try:
        async with httpx.AsyncClient(timeout=_CNINFO_HTTP_TIMEOUT) as client:
            # 某些市场（"预披露"）不需要 orgId 查找 — 查询端点接受 stock="" 并按分类过滤。
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

            # ``pageSize=30`` 是 cninfo 自身的默认值；分页直到获得 ``limit`` 条记录或数据耗尽，以先到者为准。
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
                r = await client.post(CNINFO_QUERY_URL, data=form, headers=_CNINFO_HEADERS)
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
# 工具 2：将 PDF 下载到磁盘缓存
# ---------------------------------------------------------------------
async def _download_bytes(url: str) -> bytes:
    async with (
        httpx.AsyncClient(
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
            follow_redirects=True,
        ) as cli,
        cli.stream("GET", url) as r,
    ):
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
    """将 PDF 下载到磁盘缓存；返回本地路径。

    缓存命中零开销 — 无论 LLM 在推理过程中重复调用此工具多少次，相同 URL 始终解析为相同路径。
    成功下载的文件在写入前会验证 ``%PDF`` 魔数字节，因此磁盘上不会留下截断或 HTML 错误页伪装的 PDF。

    Args:
        pdf_url: PDF 的绝对 URL（通常为 ``search_announcements`` 返回的 ``static.cninfo.com.cn/finalpage/...`` 链接）。

    Returns:
        包含 ``pdf_url``、``local_path``（绝对路径）、``size_bytes`` 和 ``from_cache``（布尔值，表示是否复用了已有文件）的字典。
        失败时返回 ``{"error": ..., "context": ...}``。
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
# 工具 3：解析有限页范围
# ---------------------------------------------------------------------
@mcp.tool()
async def parse_pdf_pages(
    local_path: str,
    start_page: int = 1,
    end_page: int = 5,
) -> dict:
    """提取 ``[start_page, end_page]``（包含，1-indexed）范围的文本。

    LLM 不应在单次调用中请求整个文档 — 200 页的招股说明书轻易超出任何模型的上下文窗口。对于长报告，预期模式为：

    1. ``extract_pdf_metadata`` → 获取 ``num_pages``
    2. 多次调用 ``parse_pdf_pages(start, end)``，每次 20 页，扫描用户询问的章节。

    Args:
        local_path: ``download_pdf`` 先前返回的绝对路径。
        start_page: 起始页，1-indexed（默认 1）。
        end_page: 结束页，包含，1-indexed（默认 5）。

    Returns:
        包含 ``local_path``、``requested_range``（``{start, end}``）、``total_pages`` 和 ``pages`` 的字典 — 每条为``{page_number, char_count, text}``。
        超出文档实际范围的页面被静默跳过。
    """
    if start_page < 1 or end_page < start_page:
        return _fmt_error(
            ValueError(f"invalid range: start_page={start_page}, end_page={end_page}"),
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
            FileNotFoundError(f"没有这个文件: {path}"),
            context=f"parse_pdf_pages({local_path!r})",
        )

    def _call() -> dict[str, Any]:
        with path.open("rb") as fh:
            reader = pypdf.PdfReader(fh)
            total = len(reader.pages)
            pages_out: list[dict[str, Any]] = []
            for p in range(start_page, min(end_page, total) + 1):
                # pypdf 内部页码从 0 开始
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
# 工具 4：文档级元数据
# ---------------------------------------------------------------------
@mcp.tool()
async def extract_pdf_metadata(local_path: str) -> dict:
    """返回页数、标题、作者、创建者和文件大小。

    开销低 — 不解码任何页面内容。适合作为对陌生文档的首次调用，用于决定如何对其分片。

    Args:
        local_path: ``download_pdf`` 返回的绝对路径。

    Returns:
        包含 ``local_path``、``num_pages``、``size_bytes`` 和
         ``metadata``（PDF 中嵌入的 ``{title, author, subject, creator, producer, creation_date, mod_date}`` 之一的字典）的字典。
        缺失字段直接不存在而非 ``None``。
    """
    path = Path(local_path)
    if not path.exists():
        return _fmt_error(
            FileNotFoundError(f"没有这个文件: {path}"),
            context=f"extract_pdf_metadata({local_path!r})",
        )

    def _call() -> dict[str, Any]:
        with path.open("rb") as fh:
            reader = pypdf.PdfReader(fh)
            num_pages = len(reader.pages)
            raw_meta = dict(reader.metadata) if reader.metadata else {}
        # 去除 PDF 元数据键的前导斜杠惯例，提高可读性。
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
