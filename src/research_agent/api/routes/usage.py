"""聚合 LLM Token 用量计数器（进程生命周期）。

同一数据的两种视图：

* ``GET /api/usage``   — JSON（供应用客户端/仪表盘使用）。
* ``GET /metrics``     — Prometheus 文本暴露格式（供``prometheus.yml`` 抓取目标或任何 OTEL 兼容收集器使用）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from research_agent.observability.metrics import METRICS

if TYPE_CHECKING:
    from research_agent.api.dependencies import ModelRouterDep

router = APIRouter(prefix="/api", tags=["usage"])


@router.get("/usage")
async def get_llm_usage(model_router: ModelRouterDep) -> dict[str, Any]:
    """返回当前 worker 的 :meth:`~research_agent.llm.usage_tracker.UsageTracker.summary`。"""
    return model_router.usage.summary()


# 挂载在顶层（无 /api 前缀），以便 Prometheus 可以直接抓取约定的 ``/metrics`` 路径。
metrics_router = APIRouter(tags=["observability"])


@metrics_router.get("/metrics")
async def prometheus_metrics(model_router: ModelRouterDep) -> PlainTextResponse:
    """Prometheus 文本暴露端点。"""
    usage = model_router.usage.summary()
    body = METRICS.render(usage_summary=usage)
    return PlainTextResponse(body, media_type="text/plain; version=0.0.4; charset=utf-8")
