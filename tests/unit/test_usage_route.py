"""GET /api/usage 和 GET /metrics 的单元测试。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from research_agent.api.dependencies import get_model_router
from research_agent.api.routes.usage import metrics_router
from research_agent.api.routes.usage import router as usage_router


@pytest.mark.asyncio
async def test_usage_returns_tracker_summary() -> None:
    sample = {
        "by_agent": {"light": {"prompt_tokens": 1, "call_count": 1}},
        "by_model": {"gpt-test": {}},
        "total_cost_cny": 0.01,
    }
    mr = MagicMock()
    mr.usage.summary.return_value = sample

    app = FastAPI()
    app.include_router(usage_router)
    app.dependency_overrides[get_model_router] = lambda: mr

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            r = await client.get("/api/usage")
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 200
    assert r.json() == sample
    mr.usage.summary.assert_called_once()


@pytest.mark.asyncio
async def test_metrics_returns_prometheus_text() -> None:
    mr = MagicMock()
    mr.usage.summary.return_value = {
        "by_agent": {},
        "by_model": {
            "qwen3-max": {
                "prompt_tokens": 120,
                "completion_tokens": 80,
                "total_tokens": 200,
                "call_count": 3,
                "total_cost_cny": 0.0003,
            },
        },
        "total_cost_cny": 0.0003,
    }

    app = FastAPI()
    app.include_router(metrics_router)
    app.dependency_overrides[get_model_router] = lambda: mr

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            r = await client.get("/metrics")
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    body = r.text
    assert "research_agent_http_requests_total" in body
    assert "research_agent_specialists_available" in body
    assert 'research_agent_llm_prompt_tokens_total{model="qwen3-max"} 120' in body
    assert 'research_agent_llm_completion_tokens_total{model="qwen3-max"} 80' in body
    assert 'research_agent_llm_calls_total{model="qwen3-max"} 3' in body
