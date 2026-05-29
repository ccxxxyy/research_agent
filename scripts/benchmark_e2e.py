"""端到端性能基准测试套件。

对 Research Agent API 各端点进行延迟 / 吞吐量基准测试，支持并发负载、百分位统计以及 JSON 报告输出。

用法::

    # 完整测试（含 LLM 调用）
    python scripts/benchmark_e2e.py

    # 快速冒烟测试（仅无 LLM 端点）
    python scripts/benchmark_e2e.py --quick

    # 自定义参数
    python scripts/benchmark_e2e.py --base-url http://10.0.0.5:8080 \\
        --concurrency 1,5,10,20 --iterations 50 --warmup 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# 端点定义
# ---------------------------------------------------------------------------

# 各端点的描述：(名称, 方法, 路径, JSON body | None, 是否 SSE 流, 是否需要 LLM)
ENDPOINTS: list[dict[str, Any]] = [
    {
        "name": "GET /health",
        "method": "GET",
        "path": "/health",
        "body": None,
        "stream": False,
        "requires_llm": False,
    },
    {
        "name": "GET /api/usage",
        "method": "GET",
        "path": "/api/usage",
        "body": None,
        "stream": False,
        "requires_llm": False,
    },
    {
        "name": "GET /metrics",
        "method": "GET",
        "path": "/metrics",
        "body": None,
        "stream": False,
        "requires_llm": False,
    },
    {
        "name": "POST /api/supervisor/research",
        "method": "POST",
        "path": "/api/supervisor/research",
        "body": {"query": "查询宁德时代基本面", "user_id": "benchmark"},
        "stream": False,
        "requires_llm": True,
    },
    {
        "name": "POST /api/supervisor/research/stream",
        "method": "POST",
        "path": "/api/supervisor/research/stream",
        "body": {"query": "查询宁德时代基本面", "user_id": "benchmark"},
        "stream": True,
        "requires_llm": True,
    },
]

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class SingleResult:
    """单次请求的测量结果。"""

    latency: float  # 秒
    status_code: int
    error: str | None = None


@dataclass
class EndpointStats:
    """某端点在特定并发级别下的聚合统计。"""

    endpoint: str
    concurrency: int
    iterations: int
    latencies: list[float] = field(default_factory=list)
    errors: int = 0

    @property
    def success_count(self) -> int:
        return len(self.latencies)

    def percentile(self, p: float) -> float:
        """计算第 p 百分位延迟（毫秒）。"""
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        k = (p / 100.0) * (len(sorted_lat) - 1)
        f = int(k)
        c = f + 1
        if c >= len(sorted_lat):
            return sorted_lat[-1] * 1000.0
        return (sorted_lat[f] + (k - f) * (sorted_lat[c] - sorted_lat[f])) * 1000.0

    def summary(self) -> dict[str, Any]:
        if not self.latencies:
            return {
                "endpoint": self.endpoint,
                "concurrency": self.concurrency,
                "iterations": self.iterations,
                "success": 0,
                "errors": self.errors,
                "p50_ms": 0,
                "p95_ms": 0,
                "p99_ms": 0,
                "max_ms": 0,
                "mean_ms": 0,
                "stdev_ms": 0,
                "throughput_rps": 0,
            }
        total_time = sum(self.latencies)
        stdev = statistics.stdev(self.latencies) * 1000.0 if len(self.latencies) > 1 else 0.0
        return {
            "endpoint": self.endpoint,
            "concurrency": self.concurrency,
            "iterations": self.iterations,
            "success": self.success_count,
            "errors": self.errors,
            "p50_ms": round(self.percentile(50), 2),
            "p95_ms": round(self.percentile(95), 2),
            "p99_ms": round(self.percentile(99), 2),
            "max_ms": round(max(self.latencies) * 1000.0, 2),
            "mean_ms": round(statistics.mean(self.latencies) * 1000.0, 2),
            "stdev_ms": round(stdev, 2),
            "throughput_rps": round(self.success_count / total_time, 2) if total_time > 0 else 0,
        }


# ---------------------------------------------------------------------------
# HTTP 请求执行
# ---------------------------------------------------------------------------


async def _do_request(
    client: httpx.AsyncClient,
    endpoint: dict[str, Any],
    base_url: str,
) -> SingleResult:
    """执行单次 HTTP 请求并返回延迟测量。"""
    url = f"{base_url}{endpoint['path']}"
    start = time.perf_counter()
    try:
        if endpoint["stream"]:
            # SSE 流：测量到流结束的完整耗时（TTLT）
            async with client.stream(
                endpoint["method"],
                url,
                json=endpoint["body"],
                timeout=300.0,
            ) as resp:
                async for _ in resp.aiter_lines():
                    pass
                elapsed = time.perf_counter() - start
                return SingleResult(latency=elapsed, status_code=resp.status_code)
        else:
            if endpoint["method"] == "GET":
                resp = await client.get(url, timeout=300.0)
            else:
                resp = await client.post(url, json=endpoint["body"], timeout=300.0)
            elapsed = time.perf_counter() - start
            return SingleResult(latency=elapsed, status_code=resp.status_code)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return SingleResult(latency=elapsed, status_code=0, error=str(exc))


# ---------------------------------------------------------------------------
# 预热与可用性检测
# ---------------------------------------------------------------------------


async def warmup_endpoint(
    client: httpx.AsyncClient,
    endpoint: dict[str, Any],
    base_url: str,
    warmup_rounds: int,
) -> bool:
    """对端点执行预热请求，返回该端点是否可用（全部 2xx）。"""
    for i in range(warmup_rounds):
        result = await _do_request(client, endpoint, base_url)
        if result.error or result.status_code < 200 or result.status_code >= 300:
            log.warning(
                "预热失败 [%s] 第 %d/%d 轮: status=%s error=%s",
                endpoint["name"],
                i + 1,
                warmup_rounds,
                result.status_code,
                result.error,
            )
            return False
    return True


# ---------------------------------------------------------------------------
# 并发负载测试
# ---------------------------------------------------------------------------


async def run_concurrent_load(
    client: httpx.AsyncClient,
    endpoint: dict[str, Any],
    base_url: str,
    concurrency: int,
    iterations: int,
) -> EndpointStats:
    """以指定并发度对端点发起 iterations 次请求。

    使用 asyncio.Semaphore 控制最大并发数，所有 iterations 次请求通过 gather 并行调度。
    """
    stats = EndpointStats(
        endpoint=endpoint["name"],
        concurrency=concurrency,
        iterations=iterations,
    )
    sem = asyncio.Semaphore(concurrency)

    async def _task() -> SingleResult:
        async with sem:
            return await _do_request(client, endpoint, base_url)

    tasks = [asyncio.create_task(_task()) for _ in range(iterations)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        if isinstance(r, Exception) or r.error or r.status_code < 200 or r.status_code >= 300:
            stats.errors += 1
        else:
            stats.latencies.append(r.latency)

    return stats


# ---------------------------------------------------------------------------
# 控制台输出格式化
# ---------------------------------------------------------------------------


def _fmt_float(v: float, width: int = 10) -> str:
    """右对齐浮点数格式化。"""
    return f"{v:>{width}.2f}"


def print_summary_table(all_stats: list[dict[str, Any]]) -> None:
    """以表格形式打印基准测试结果摘要。"""
    if not all_stats:
        log.info("无可用结果。")
        return

    header = (
        f"{'端点':<40s} {'并发':>4s} {'成功':>4s} {'失败':>4s}"
        f" {'P50(ms)':>10s} {'P95(ms)':>10s} {'P99(ms)':>10s}"
        f" {'Max(ms)':>10s} {'Mean(ms)':>10s} {'QPS':>8s}"
    )
    sep = "-" * len(header)

    print()
    print(sep)
    print("  基准测试结果摘要")
    print(sep)
    print(header)
    print(sep)

    for s in all_stats:
        line = (
            f"{s['endpoint']:<40s} {s['concurrency']:>4d} {s['success']:>4d} {s['errors']:>4d}"
            f" {_fmt_float(s['p50_ms'])} {_fmt_float(s['p95_ms'])} {_fmt_float(s['p99_ms'])}"
            f" {_fmt_float(s['max_ms'])} {_fmt_float(s['mean_ms'])}"
            f" {_fmt_float(s['throughput_rps'], 8)}"
        )
        print(line)

    print(sep)
    print()


# ---------------------------------------------------------------------------
# JSON 报告输出
# ---------------------------------------------------------------------------


def save_json_report(
    all_stats: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    """将完整结果保存为带时间戳的 JSON 文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"benchmark_{ts}.json"

    report = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "config": {
            "base_url": args.base_url,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "concurrency_levels": args.concurrency,
            "quick_mode": args.quick,
        },
        "results": all_stats,
    }

    filepath.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("JSON 报告已保存: %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


async def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    """执行完整基准测试流程。"""
    concurrency_levels = [int(c.strip()) for c in args.concurrency.split(",") if c.strip()]
    if not concurrency_levels:
        log.error("并发级别列表为空。")
        return []

    # 根据 --quick 标志筛选端点
    if args.quick:
        endpoints = [ep for ep in ENDPOINTS if not ep["requires_llm"]]
        log.info("快速模式: 仅测试无 LLM 端点 (%d 个)", len(endpoints))
    else:
        endpoints = list(ENDPOINTS)
        log.info("完整模式: 测试全部端点 (%d 个)", len(endpoints))

    log.info(
        "参数: base_url=%s, warmup=%d, iterations=%d, concurrency=%s",
        args.base_url,
        args.warmup,
        args.iterations,
        concurrency_levels,
    )

    all_stats: list[dict[str, Any]] = []

    async with httpx.AsyncClient() as client:
        # --- 连通性检查 ---
        log.info("检查服务可达性...")
        try:
            probe = await client.get(f"{args.base_url}/health", timeout=10.0)
            log.info("服务状态: %d", probe.status_code)
        except Exception as exc:
            log.error("无法连接到 %s: %s", args.base_url, exc)
            log.error("请确认服务已启动。")
            return []

        # --- 预热阶段 ---
        available_endpoints: list[dict[str, Any]] = []
        for ep in endpoints:
            log.info("预热 [%s] (%d 轮)...", ep["name"], args.warmup)
            ok = await warmup_endpoint(client, ep, args.base_url, args.warmup)
            if ok:
                available_endpoints.append(ep)
                log.info("  -> 可用")
            else:
                log.warning("  -> 跳过（预热期间返回非 2xx）")

        if not available_endpoints:
            log.error("所有端点预热均失败，无法继续测试。")
            return []

        log.info("可用端点: %d/%d", len(available_endpoints), len(endpoints))

        # --- 正式测试 ---
        for ep in available_endpoints:
            for conc in concurrency_levels:
                label = f"[{ep['name']}] concurrency={conc}"
                log.info("测试 %s, iterations=%d ...", label, args.iterations)

                stats = await run_concurrent_load(
                    client,
                    ep,
                    args.base_url,
                    concurrency=conc,
                    iterations=args.iterations,
                )
                summary = stats.summary()
                all_stats.append(summary)

                log.info(
                    "  完成: P50=%.1fms P95=%.1fms P99=%.1fms QPS=%.1f errors=%d",
                    summary["p50_ms"],
                    summary["p95_ms"],
                    summary["p99_ms"],
                    summary["throughput_rps"],
                    summary["errors"],
                )

    return all_stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Research Agent API 端到端性能基准测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python scripts/benchmark_e2e.py --quick\n"
            "  python scripts/benchmark_e2e.py --concurrency 1,5,10,20 --iterations 50\n"
        ),
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="API 服务地址 (默认: http://localhost:8080)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="每个端点的预热轮数 (默认: 3)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="每个 (端点, 并发级别) 组合的请求次数 (默认: 20)",
    )
    parser.add_argument(
        "--concurrency",
        default="1,5,10",
        help="逗号分隔的并发级别列表 (默认: 1,5,10)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="JSON 报告输出目录 (默认: benchmark_results)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速模式: 仅测试无 LLM 调用的端点 (health/usage/metrics)，迭代次数减半",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    # 快速模式下自动减少迭代次数（如果用户未显式指定 --iterations）
    if args.quick and "--iterations" not in sys.argv:
        args.iterations = max(args.iterations // 2, 5)
        log.info("快速模式: 迭代次数自动调整为 %d", args.iterations)

    log.info("=" * 60)
    log.info("Research Agent 端到端性能基准测试")
    log.info("=" * 60)

    start = time.perf_counter()
    results = asyncio.run(run_benchmark(args))
    elapsed = time.perf_counter() - start

    if results:
        print_summary_table(results)
        report_path = save_json_report(results, Path(args.output_dir), args)
        log.info("总耗时: %.1f 秒", elapsed)
        log.info("报告文件: %s", report_path)
    else:
        log.warning("未产生任何测试结果。")
        sys.exit(1)


if __name__ == "__main__":
    main()
