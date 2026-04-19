"""Integration tests for FastAPI endpoints."""

import pytest

pytestmark = pytest.mark.integration


class TestHealthAPI:
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Verify health check returns 200."""
        from httpx import ASGITransport, AsyncClient

        from research_agent.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
