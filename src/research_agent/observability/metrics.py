"""Lightweight Prometheus-compatible metrics (zero external dependencies).

Exposes three families of counters:

* **HTTP request counters** — ``research_agent_http_requests_total``
  and ``research_agent_http_request_duration_seconds`` (a counter
  that sums wall-clock time; divide by request count for average
  latency). Labels: ``method``, ``path``, ``status``.
* **LLM token counters** — ``research_agent_llm_prompt_tokens_total``,
  ``research_agent_llm_completion_tokens_total``,
  ``research_agent_llm_calls_total``,
  ``research_agent_llm_cost_usd_total``.
  Labels: ``model``.
* **Specialist availability** —
  ``research_agent_specialists_available`` (gauge, set once at startup).

All state lives in a module-level singleton, safe for the single-
process uvicorn deployment this project targets. The ``/metrics``
route renders the text exposition format Prometheus expects.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


# ---------------------------------------------------------------------------
# In-process counters
# ---------------------------------------------------------------------------

class _Counters:
    """Thread-safe metric store."""

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
        lines: list[str] = []

        with self._lock:
            http_req = dict(self._http_requests)
            http_dur = dict(self._http_duration)
            specialists = list(self._specialists_available)

        # -- HTTP requests --
        lines.append("# HELP research_agent_http_requests_total Total HTTP requests.")
        lines.append("# TYPE research_agent_http_requests_total counter")
        for (method, path, status), count in sorted(http_req.items()):
            lines.append(
                f'research_agent_http_requests_total{{method="{method}",'
                f'path="{path}",status="{status}"}} {count}'
            )

        lines.append("# HELP research_agent_http_request_duration_seconds_total Cumulative request wall-clock time.")
        lines.append("# TYPE research_agent_http_request_duration_seconds_total counter")
        for (method, path, status), dur in sorted(http_dur.items()):
            lines.append(
                f'research_agent_http_request_duration_seconds_total{{method="{method}",'
                f'path="{path}",status="{status}"}} {dur:.6f}'
            )

        # -- Specialist availability gauge --
        lines.append("# HELP research_agent_specialists_available Number of active specialists.")
        lines.append("# TYPE research_agent_specialists_available gauge")
        lines.append(f"research_agent_specialists_available {len(specialists)}")

        # -- LLM usage (from UsageTracker.summary()) --
        if usage_summary:
            by_model: dict = usage_summary.get("by_model", {})
            if by_model:
                lines.append("# HELP research_agent_llm_prompt_tokens_total Total LLM prompt tokens.")
                lines.append("# TYPE research_agent_llm_prompt_tokens_total counter")
                for model, rec in sorted(by_model.items()):
                    lines.append(
                        f'research_agent_llm_prompt_tokens_total{{model="{model}"}} '
                        f'{rec.get("prompt_tokens", 0)}'
                    )

                lines.append("# HELP research_agent_llm_completion_tokens_total Total LLM completion tokens.")
                lines.append("# TYPE research_agent_llm_completion_tokens_total counter")
                for model, rec in sorted(by_model.items()):
                    lines.append(
                        f'research_agent_llm_completion_tokens_total{{model="{model}"}} '
                        f'{rec.get("completion_tokens", 0)}'
                    )

                lines.append("# HELP research_agent_llm_calls_total Total LLM calls.")
                lines.append("# TYPE research_agent_llm_calls_total counter")
                for model, rec in sorted(by_model.items()):
                    lines.append(
                        f'research_agent_llm_calls_total{{model="{model}"}} '
                        f'{rec.get("call_count", 0)}'
                    )

                lines.append("# HELP research_agent_llm_cost_usd_total Estimated LLM cost (USD).")
                lines.append("# TYPE research_agent_llm_cost_usd_total counter")
                for model, rec in sorted(by_model.items()):
                    lines.append(
                        f'research_agent_llm_cost_usd_total{{model="{model}"}} '
                        f'{rec.get("total_cost_usd", 0)}'
                    )

        lines.append("")
        return "\n".join(lines)


METRICS = _Counters()


# ---------------------------------------------------------------------------
# Middleware — record every request
# ---------------------------------------------------------------------------

class MetricsMiddleware(BaseHTTPMiddleware):
    """Record per-request count and wall-clock duration."""

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
