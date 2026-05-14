"""Aggregate LLM token usage counters (process lifetime).

Two views of the same data:

* ``GET /api/usage``   — JSON (for application clients / dashboards).
* ``GET /metrics``     — Prometheus text exposition format (for
  ``prometheus.yml`` scrape targets or any OTEL-compatible collector).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from research_agent.api.dependencies import ModelRouterDep
from research_agent.observability.metrics import METRICS

router = APIRouter(prefix="/api", tags=["usage"])


@router.get("/usage")
async def get_llm_usage(model_router: ModelRouterDep) -> dict[str, Any]:
    """Return :meth:`~research_agent.llm.usage_tracker.UsageTracker.summary` for this worker."""
    return model_router.usage.summary()


# Mounted at top-level (no /api prefix) so Prometheus can scrape the
# conventional ``/metrics`` path out of the box.
metrics_router = APIRouter(tags=["observability"])


@metrics_router.get("/metrics")
async def prometheus_metrics(model_router: ModelRouterDep) -> PlainTextResponse:
    """Prometheus text exposition endpoint."""
    usage = model_router.usage.summary()
    body = METRICS.render(usage_summary=usage)
    return PlainTextResponse(body, media_type="text/plain; version=0.0.4; charset=utf-8")
