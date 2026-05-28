"""将真实 A 股研报灌入生产知识库。

本脚本的功能
------------
从 巨潮资讯（cninfo）拉取一组代表性 A 股标的最近 1-2 份披露（优先年报/季报而非临时公告），
将各 PDF 下载到``./data/pdf_cache/`` 内容寻址缓存，再灌入持久化 FAISS collection``prod_reports``（所有报告共用一个 collection —
分块元数据携带``source = 本地 PDF 路径``，Agent 仍可按文档引用）。

灌入后的 collection 成为 ``knowledge_expert`` 在运行时搜索的操作语料库（当 supervisor 需要历史研报上下文时）—
见``scripts/demo_full_research.py`` 中对应的端到端演示问题。

为何需要本脚本
--------------
此前项目的 RAG 层虽可演示但从未使用真实数据*驱动：``demo_knowledge_expert.py`` 灌入的是手工构造的 2 页合成 PDF，以验证管道连通性。有了本脚本得到：

  * 一个由真实披露文件填充的固定名称 FAISS collection（可直接引用，无需依赖过期的时间戳或调用中途的网络抖动）；
  * 一条可复现的幂等路径 — 重复运行脚本不会重复灌入已有分块（灌入前会检查 ``knowledge_list_collections`` 和按 source 的分块计数）；
  * 灌入（本脚本，运行一次）与检索增强问答（演示 + 运行时 Agent）之间的清晰分离。

为何以进程内方式调用工具（而非 MCP-stdio）
------------------------------------------
``pdf_report_server`` 和 ``knowledge_server`` 都将工具暴露为``@mcp.tool()`` 装饰的普通 async 函数（装饰器向 FastMCP 注册但不做包装 —
见 ``knowledge_server.py`` 模块文档字符串）。从本脚本直接调用与通过 MCP-stdio 传输功能等价，但省去了每次调用的两次子进程创建 + JSON-RPC 序列化。
对于一个执行约 10 秒 HTTP I/O +约 20 秒灌入的一次性运维脚本，这是值得的捷径。

运行::

    .venv/Scripts/python.exe scripts/seed_real_research_reports.py
    # 可选: --tickers 600519,300750  --collection prod_reports

退出码:
    0 → 至少一份报告已灌入或已存在。
    1 → 所有标的均未能获取报告（网络不通、cninfo 接口变更等）。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any

# 强制 stdout 使用 UTF-8，防止中文字符在 Windows 代码页上报错。
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


# ---------------------------------------------------------------------
# 精选标的列表 — AI / 半导体交叉截面
#
# AI 价值链的三个子板块，使演示问题能在
# "算力 / 互联 / 存储" 之间进行有意义的对比：
#
#     AI 算力芯片     688256 寒武纪      — 训练 GPU/ASIC, 亏损改善
#     CPO / 光模块    300308 中际旭创    — AI 数据中心互联, 业绩爆发
#     存储芯片        603986 兆易创新    — NOR Flash / MCU, 周期复苏
#
# 可自由添加更多标的；灌入按顺序执行，列表越长耗时越长
# （每份年报约 10 MB → 预热后约 20 秒灌入，瓶颈在 embedder）。
# ---------------------------------------------------------------------
DEFAULT_TICKERS: dict[str, str] = {
    "688256": "寒武纪",
    "300308": "中际旭创",
    "603986": "兆易创新",
}

DEFAULT_COLLECTION = "prod_reports"

# 按顺序尝试的披露类别，直到找到可用的。cninfo 的"年报"每年发布一次（3/4 月），因此在刚翻年的时段可能只有季报可用 — 回退链可防止灌入结果为空。
PREFERRED_CATEGORIES = ("年报", "一季报", "三季报", "半年报")

# 回溯窗口：365 天可保证每个标的至少有一份年报，即使其财年在日历年较晚结束。
LOOKBACK_DAYS = 365

# 每个标的的硬性上限。灌入时我们只需一份最近年报 + 一份季报，以保持语料库小而灌入快。embedder 每 100 页报告约耗时 20 秒。
MAX_REPORTS_PER_TICKER = 1


def _step(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def _find_recent_report(
    *,
    search_announcements,
    symbol: str,
    end_date: datetime,
) -> dict[str, Any] | None:
    """按优先级顺序尝试各类别；返回最近的一条记录。

    返回公告 dict（携带 ``pdf_url``）或 ``None``（所有类别均无可用记录）。取 ``announcements`` 的第一条，因为 cninfo 按``publish_date`` 降序排列。
    """
    start_date = (end_date - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    for cat in PREFERRED_CATEGORIES:
        _step(f"  search_announcements(symbol={symbol}, category={cat}, "
              f"window={start_date}..{end_str})")
        try:
            resp = await search_announcements(
                symbol=symbol,
                start_date=start_date,
                end_date=end_str,
                category=cat,
                limit=10,
            )
        except Exception as exc:  # noqa: BLE001
            _step(f"    category={cat} 搜索失败: {exc!r}")
            continue
        if "error" in resp:
            _step(f"    category={cat} 搜索返回错误: {resp['error']}")
            continue
        records = [
            r for r in resp.get("announcements", []) if r.get("pdf_url")
        ]
        if not records:
            _step(f"    category={cat} 无 PDF 公告；尝试下一类别。")
            continue
        chosen = records[0]
        _step(
            f"  已选择: {chosen.get('publish_date')} | "
            f"{chosen.get('title', '')[:60]} | size?"
        )
        return chosen
    return None


async def _ingested_sources_for_collection(
    *,
    list_collections,
    collection: str,
    knowledge_search,
) -> set[str]:
    """返回 ``collection`` 中已存在的 ``source`` 路径集合。

    使用一个泛匹配查询（几乎能匹配任何语料中内容的单个中文字符），并拉取较大的 ``top_k`` 以便去重。FAISS 预热后开销很小。若 collection 不存在则返回空集。
    """
    listing = await list_collections()
    names = {c["name"] for c in listing.get("collections", [])}
    if collection not in names:
        return set()
    # ``top_k`` 受 knowledge_server 的 MAX_TOP_K 限制为 20 — 对于只含少量大 PDF 的灌入语料库，已足够获取不同的 ``source``值用于去重。
    probe = await knowledge_search(
        query="公司",  # 泛匹配，专用于灌入前检查
        collection=collection,
        top_k=20,
    )
    sources: set[str] = set()
    for hit in probe.get("results", []):
        src = hit.get("source")
        if src:
            sources.add(src)
    return sources


async def amain(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="灌入 prod_reports 知识库 collection。"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=",".join(DEFAULT_TICKERS),
        help=(
            "逗号分隔的 6 位 A 股代码。日志中的名称从 DEFAULT_TICKERS "
            "查找，否则直接使用代码作为显示名称。"
        ),
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help="目标 FAISS collection 名称。",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="",
        help=(
            "cninfo 搜索窗口的截止日期，格式 YYYYMMDD。默认为今天。"
            "仅在确定性快照测试时覆盖。"
        ),
    )
    args = parser.parse_args(argv)

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        _step("失败: 未提供标的。")
        return 1

    end_date = (
        datetime.strptime(args.end_date, "%Y%m%d")
        if args.end_date
        else datetime.now()
    )

    # --- 延迟导入，使上面的 argparse 错误能快速返回 ---
    _step("正在加载工具模块（会触发 bge embedder 的导入）")
    from research_agent.mcp_servers.knowledge_server import (
        delete_collection,  # noqa: F401  （暴露用于临时清理）
        ingest_pdf,
        list_collections,
    )
    from research_agent.mcp_servers.knowledge_server import (
        search as knowledge_search,
    )
    from research_agent.mcp_servers.pdf_report_server import (
        download_pdf,
        search_announcements,
    )

    _step(f"目标 collection: {args.collection!r}")
    _step(f"标的列表: {tickers}")

    already = await _ingested_sources_for_collection(
        list_collections=list_collections,
        collection=args.collection,
        knowledge_search=knowledge_search,
    )
    if already:
        _step(f"  collection 已有 {len(already)} 个不同 source；"
              f"已有 PDF 将被跳过（幂等重跑）。")

    seeded = 0
    skipped = 0
    failed: list[str] = []

    for sym in tickers:
        name = DEFAULT_TICKERS.get(sym, sym)
        _step(f"=== {sym} {name} ===")
        record = await _find_recent_report(
            search_announcements=search_announcements,
            symbol=sym,
            end_date=end_date,
        )
        if record is None:
            _step(f"  未找到 {sym} 的近期报告；跳过。")
            failed.append(sym)
            continue

        pdf_url = record.get("pdf_url")
        if not pdf_url:
            _step("  记录缺少 pdf_url；跳过。")
            failed.append(sym)
            continue

        _step(f"  download_pdf({pdf_url[:80]}...)")
        try:
            dl = await download_pdf(pdf_url=pdf_url)
        except Exception as exc:  # noqa: BLE001
            _step(f"  下载失败: {exc!r}")
            failed.append(sym)
            continue
        if "error" in dl:
            _step(f"  下载返回错误: {dl['error']}")
            failed.append(sym)
            continue

        local_path = dl["local_path"]
        size_kb = dl.get("size_bytes", 0) // 1024
        from_cache = dl.get("from_cache", False)
        _step(f"  -> {local_path} ({size_kb} KB, from_cache={from_cache})")

        # 幂等性：如果该 PDF 的路径已在 collection 的 ``source``集合中，则跳过灌入。
        # knowledge_server 已实现基于文件 SHA-256 的内容哈希去重（相同内容的 PDF 返回 skipped=True），
        # 此处路径级检查作为快速前置过滤，避免不必要的函数调用开销。
        if local_path in already:
            _step("  已在此 collection 中灌入过；跳过。")
            skipped += 1
            continue

        _step(f"  knowledge_ingest_pdf(collection={args.collection!r})")
        try:
            ing = await ingest_pdf(
                local_path=local_path,
                collection=args.collection,
            )
        except Exception as exc:  # noqa: BLE001
            _step(f"  灌入崩溃: {exc!r}")
            failed.append(sym)
            continue
        if "error" in ing:
            _step(f"  灌入返回错误: {ing['error']}")
            failed.append(sym)
            continue

        added = ing.get("num_chunks_added", 0)
        total = ing.get("total_chunks_in_collection", 0)
        _step(f"  已灌入: +{added} 个分块（collection 总计: {total}）")
        seeded += 1
        already.add(local_path)

    _step("=== 汇总 ===")
    _step(f"  新灌入:       {seeded}")
    _step(f"  已缓存跳过:   {skipped}")
    _step(f"  失败标的:     {failed if failed else '(无)'}")
    if seeded == 0 and skipped == 0:
        _step("失败: 无任何内容进入索引。")
        return 1
    _step("完成")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(amain(sys.argv[1:])))
