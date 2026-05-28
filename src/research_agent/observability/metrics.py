"""轻量级 Prometheus 兼容指标（零外部依赖）。统计请求数、耗时、LLM 费用

暴露三类计数器：

* HTTP 请求计数器 — ``research_agent_http_requests_total``和 ``research_agent_http_request_duration_seconds``（累加挂钟时间的计数器；除以请求数可得平均延迟）。
  标签：``method``、``path``、``status``。
* LLM Token 计数器 — ``research_agent_llm_prompt_tokens_total``、``research_agent_llm_completion_tokens_total``、
  ``research_agent_llm_calls_total``、``research_agent_llm_cost_cny_total``。
  标签：``model``。
* 专家可用性 —``research_agent_specialists_available``（Gauge，启动时设置一次）。

Prometheus 指标名，每个计数器的全局唯一名字：
    research_agent_http_requests_total           → HTTP 请求总数
    research_agent_http_request_duration_seconds  → HTTP 请求累计耗时
    research_agent_llm_prompt_tokens_total        → LLM 输入 token 总数
    research_agent_llm_completion_tokens_total     → LLM 输出 token 总数
    research_agent_llm_calls_total                → LLM 调用总次数
    research_agent_llm_cost_cny_total             → LLM 估算费用（人民币元）
    research_agent_specialists_available          → 当前可用专家数

HTTP 指标（请求数 + 耗时）在 _Counters 类的 _http_requests 和 _http_duration 里。
LLM 指标（token 数 + 费用）不在 _Counters 里——它们在 UsageTracker 里，通过 render(usage_summary=...) 参数传入合并输出。
专家可用性在 _Counters 的 _specialists_available 里。


所有状态保存在模块级单例中；适用于本项目目标的单进程 uvicorn 部署：每个进程有自己的 METRICS 实例，计数器互相独立，/metrics 只能看到当前进程的数据——会不准
``/metrics`` 路由以 Prometheus 期望的文本展示格式渲染。

_Counters 类：存储数据（记录计数）
MetricsMiddleware：自动收集 HTTP 请求数据
METRICS = _Counters()：全局单例
render()：把存储的数据格式化成 Prometheus 文本
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import Request, Response

# ---------------------------------------------------------------------------
# 进程内计数器
# ---------------------------------------------------------------------------


class _Counters:
    """线程安全的指标存储。计数器：记录 HTTP 请求数、耗时、LLM 用量"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._http_requests: dict[tuple[str, str, int], int] = defaultdict(int)
        self._http_duration: dict[tuple[str, str, int], float] = defaultdict(float)
        self._specialists_available: list[str] = []

    def record_http(self, method: str, path: str, status: int, duration: float) -> None:
        key = (method, path, status)
        with self._lock:
            self._http_requests[key] += 1
            self._http_duration[key] += duration

    def set_specialists(self, names: list[str]) -> None:
        with self._lock:
            self._specialists_available = list(names)

    def render(self, usage_summary: dict | None = None) -> str:
        """render() 方法：把计数器格式化成 Prometheus 期望的文本格式

        UsageTracker（ llm/usage_tracker.py ）记录每次 LLM 调用的 token 数，乘以 MODEL_PRICING 里的单价算出费用。
        render() 方法从 UsageTracker.summary() 拿到数据后格式化输出。所以费用是估算值（基于代码里的单价表）。

        分四段拼字符串：
            拼 HTTP 请求指标：遍历 _http_requests 字典，每个 (method, path, status) 组合输出一行
            拼 HTTP 耗时指标：同理
            拼专家可用性：输出当前可用专家数量
            拼 LLM 用量：从传入的 usage_summary 参数里取数据，遍历每个模型输出 token 数、调用数、费用
        """

        lines: list[str] = []

        with self._lock:
            http_req = dict(self._http_requests)
            http_dur = dict(self._http_duration)
            specialists = list(self._specialists_available)

        # -- HTTP 请求 --
        lines.append("# HELP research_agent_http_requests_total Total HTTP requests.")
        lines.append("# TYPE research_agent_http_requests_total counter")
        for (method, path, status), count in sorted(http_req.items()):
            lines.append(
                f'research_agent_http_requests_total{{method="{method}",'
                f'path="{path}",status="{status}"}} {count}'
            )

        lines.append(
            "# HELP research_agent_http_request_duration_seconds_total Cumulative request wall-clock time."
        )
        lines.append("# TYPE research_agent_http_request_duration_seconds_total counter")
        for (method, path, status), dur in sorted(http_dur.items()):
            lines.append(
                f'research_agent_http_request_duration_seconds_total{{method="{method}",'
                f'path="{path}",status="{status}"}} {dur:.6f}'
            )

        # -- 专家可用性 Gauge --
        lines.append("# HELP research_agent_specialists_available Number of active specialists.")
        lines.append("# TYPE research_agent_specialists_available gauge")
        lines.append(f"research_agent_specialists_available {len(specialists)}")

        # -- LLM 用量（来自 UsageTracker.summary()）--
        if usage_summary:
            by_model: dict = usage_summary.get("by_model", {})
            if by_model:
                lines.append(
                    "# HELP research_agent_llm_prompt_tokens_total Total LLM prompt tokens."
                )
                lines.append("# TYPE research_agent_llm_prompt_tokens_total counter")
                for model, rec in sorted(by_model.items()):
                    lines.append(
                        f'research_agent_llm_prompt_tokens_total{{model="{model}"}} '
                        f"{rec.get('prompt_tokens', 0)}"
                    )

                lines.append(
                    "# HELP research_agent_llm_completion_tokens_total Total LLM completion tokens."
                )
                lines.append("# TYPE research_agent_llm_completion_tokens_total counter")
                for model, rec in sorted(by_model.items()):
                    lines.append(
                        f'research_agent_llm_completion_tokens_total{{model="{model}"}} '
                        f"{rec.get('completion_tokens', 0)}"
                    )

                lines.append("# HELP research_agent_llm_calls_total Total LLM calls.")
                lines.append("# TYPE research_agent_llm_calls_total counter")
                for model, rec in sorted(by_model.items()):
                    lines.append(
                        f'research_agent_llm_calls_total{{model="{model}"}} '
                        f"{rec.get('call_count', 0)}"
                    )

                lines.append("# HELP research_agent_llm_cost_cny_total Estimated LLM cost (CNY).")
                lines.append("# TYPE research_agent_llm_cost_cny_total counter")
                for model, rec in sorted(by_model.items()):
                    lines.append(
                        f'research_agent_llm_cost_cny_total{{model="{model}"}} '
                        f"{rec.get('total_cost_cny', 0)}"
                    )

        lines.append("")
        return "\n".join(lines)


METRICS = _Counters()


# ---------------------------------------------------------------------------
# 中间件 — 记录每个请求
# ---------------------------------------------------------------------------


class MetricsMiddleware(BaseHTTPMiddleware):
    """记录每个请求的计数和挂钟耗时。FastAPI 中间件，每个请求进来自动记录一次"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        METRICS.record_http(
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response
