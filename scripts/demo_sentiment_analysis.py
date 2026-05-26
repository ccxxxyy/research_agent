"""演示：多维度个股舆情量化报告（可独立运行，不需要启动 FastAPI）。

运行::

    .venv/Scripts/python.exe scripts/demo_sentiment_analysis.py
    .venv/Scripts/python.exe scripts/demo_sentiment_analysis.py --symbol 600519
    .venv/Scripts/python.exe scripts/demo_sentiment_analysis.py --json  # 审计模式

展示内容
--------
1. 东财新闻逐条打分（标题/分数/标签/关键词/指纹）
2. 聚合统计（正/负/中性比例、均分、总体结论）
3. 高频讨论词 top-15（词频 + 情感权重）
4. 话题聚类（正面/负面/中性 代表性标题各 3 条）
5. 雪球讨论热度（排名、讨论量）
6. 东财热搜关联词 top-10
7. 审计元数据（模型版本、时间戳、文本指纹）

纯本地 NLP — 不依赖大模型、不消耗 API token。
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
from typing import Any

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SEP = "=" * 72
THIN = "-" * 72


def _bar(score: float, width: int = 20) -> str:
    normalized = (score + 1) / 2
    filled = int(normalized * width)
    return "#" * filled + "-" * (width - filled)


def _print_report(report: dict[str, Any]) -> None:
    sym = report.get("symbol", "?")
    print(SEP)
    print(f"  个股舆情量化报告 — {sym}")
    print(f"  数据源: {report.get('source', '?')}")
    print(f"  模型: {report.get('model_version', '?')}")
    print(f"  时间: {report.get('timestamp', '?')}")
    print(SEP)

    # ── 1) 聚合统计 ──
    agg = report.get("aggregate", {})
    sample = agg.get("sample_size", 0)
    print()
    print("  [聚合统计]")
    print(f"    总体结论: {agg.get('overall_label', '?')}  "
          f"(均分: {agg.get('overall_score', 0):+.4f})")
    print(f"    样本量:   {sample} 条")
    print(f"    正面: {agg.get('positive_count', 0):>3} 条 "
          f"({agg.get('positive_ratio', 0):.1%})")
    print(f"    中性: {agg.get('neutral_count', 0):>3} 条 "
          f"({agg.get('neutral_ratio', 0):.1%})")
    print(f"    负面: {agg.get('negative_count', 0):>3} 条 "
          f"({agg.get('negative_ratio', 0):.1%})")

    # ── 2) 高频讨论词 ──
    hot_words = report.get("hot_words", [])
    if hot_words:
        print()
        print("  [高频讨论词] (市场在聊什么)")
        for hw in hot_words:
            word = hw["word"]
            count = hw["count"]
            weight = hw.get("sentiment_weight")
            tag = ""
            if weight is not None:
                if weight > 0:
                    tag = f"  <-- 利好词 (+{weight})"
                elif weight < 0:
                    tag = f"  <-- 利空词 ({weight})"
            print(f"    {word:>6} x{count:<3}{tag}")

    # ── 3) 话题聚类 ──
    clusters = report.get("topic_clusters", {})
    if any(clusters.values()):
        print()
        print("  [话题聚类] (按情感分组的代表性标题)")
        for label, key, marker in [
            ("正面", "positive_headlines", "+"),
            ("负面", "negative_headlines", "-"),
            ("中性", "neutral_headlines", "~"),
        ]:
            titles = clusters.get(key, [])
            if titles:
                print(f"    {label}:")
                for t in titles:
                    print(f"      {marker} {t}")

    # ── 4) 雪球讨论热度 ──
    xq = report.get("xueqiu_heat")
    if xq:
        print()
        print("  [雪球讨论热度]")
        if xq.get("on_list"):
            print(f"    该股在雪球热门榜第 {xq.get('rank', '?')} 名"
                  f" (共 {xq.get('total_ranked', '?')} 支)")
            vol = xq.get("discussion_volume", 0)
            if vol:
                print(f"    讨论量: {vol}")
            name = xq.get("stock_name", "")
            if name:
                print(f"    股票简称: {name}")
            price = xq.get("latest_price", 0)
            if price:
                print(f"    最新价: {price}")
        else:
            print(f"    该股未进入雪球热门榜"
                  f" (当前上榜 {xq.get('total_ranked', '?')} 支)")

    # ── 5) 东财热搜关联词 ──
    em_kws = report.get("eastmoney_trending_keywords", [])
    if em_kws:
        print()
        print("  [东财热搜关联词]")
        for kw in em_kws:
            word = kw.get("keyword", "")
            heat = kw.get("hot_value", "")
            t = kw.get("time", "")
            suffix = f"  ({t})" if t else ""
            print(f"    {word}  热度:{heat}{suffix}")

    # ── 6) 逐条新闻明细 ──
    items = report.get("items", [])
    if items:
        print()
        print("  [逐条新闻明细] (按分数排序)")
        print(f"  {THIN}")
        sorted_items = sorted(items, key=lambda x: x.get("sentiment_score", 0), reverse=True)
        for i, it in enumerate(sorted_items, 1):
            score = it.get("sentiment_score", 0)
            label = it.get("sentiment_label", "?")
            title = it.get("title", "")[:50]
            time_str = it.get("publish_time", "")
            kws = it.get("keywords_matched", [])
            fp = it.get("text_fingerprint", "")[:8]
            bar = _bar(score)
            print(f"  {i:>2}. [{label:>3}] {score:+.4f}  {bar}")
            print(f"      {title}")
            parts = []
            if time_str:
                parts.append(time_str)
            if kws:
                parts.append(f"命中: {', '.join(kws)}")
            parts.append(f"指纹: {fp}...")
            print(f"      {' | '.join(parts)}")
            print()

        # 极值速览
        most_pos = sorted_items[0]
        most_neg = sorted_items[-1]
        print("  [极值速览]")
        print(f"    最正面: {most_pos.get('title', '')[:35]}  "
              f"{most_pos.get('sentiment_score', 0):+.4f}")
        print(f"    最负面: {most_neg.get('title', '')[:35]}  "
              f"{most_neg.get('sentiment_score', 0):+.4f}")
    else:
        print("\n  (无新闻数据)")

    print()
    print(SEP)


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="个股舆情量化分析演示（多维度）")
    parser.add_argument("--symbol", type=str, default="300750",
                        help="6 位 A 股代码，默认 300750（宁德时代）")
    parser.add_argument("--limit", type=int, default=30,
                        help="分析新闻条数上限（默认 30）")
    parser.add_argument("--json", action="store_true",
                        help="输出原始 JSON（用于对账审计）")
    args = parser.parse_args(argv)

    print(f"\n  正在拉取 {args.symbol} 的多维度舆情数据...")
    print(f"  (东财新闻 + 雪球热度 + 热搜词 + 高频词分析 | limit={args.limit})\n")

    from research_agent.mcp_servers.news_sentiment_server import (
        get_stock_sentiment_report,
    )

    report = await get_stock_sentiment_report(symbol=args.symbol, limit=args.limit)

    if "error" in report:
        print(f"  错误: {report['error']}")
        print(f"  上下文: {report.get('context', '')}")
        return 1

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)

    print(f"  可追加 --json 查看原始 JSON 用于审计。\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
