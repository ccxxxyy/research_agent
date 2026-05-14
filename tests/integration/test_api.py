"""Integration tests for FastAPI endpoints."""

import pytest

pytestmark = pytest.mark.integration


class TestHealthAPI:
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Verify health probe is reachable without the lifespan running.

        The route should respond with HTTP 200 and a JSON body whose
        ``status`` is one of ``ok`` / ``degraded`` — never raise — even
        when called BEFORE the FastAPI lifespan has populated
        ``app.state``. The test runs without spinning up Postgres or
        Redis, so a ``degraded`` overall status is the expected
        outcome in this environment; what we are actually verifying
        is that the endpoint stays robust to a partially-initialised
        app and reports per-service status honestly.
        """
        from httpx import ASGITransport, AsyncClient

        from research_agent.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] in {"ok", "degraded"}
            assert "services" in data
            # Every probed dependency must surface a status string,
            # so callers can spot which one is down at a glance.
            expected = (
                "postgres",
                "redis",
                "knowledge_db",
                "research_supervisor",
                "checkpointer",
                "memory_store",
            )
            for name in expected:
                assert name in data["services"]
