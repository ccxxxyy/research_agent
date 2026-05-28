"""FastAPI 端点的集成测试。"""

import pytest

pytestmark = pytest.mark.integration


class TestHealthAPI:
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """验证健康探针在 lifespan 未运行时可访问。

        该路由应以 HTTP 200 响应，JSON 体的 ``status`` 为 ``ok`` / ``degraded`` 之一 — 永不抛出异常 —
        即使在 FastAPI lifespan 填充 ``app.state`` 之前被调用。此测试不启动 Postgres 或 Redis，因此 ``degraded`` 整体状态是该环境中的预期结果；
        实际验证的是端点在 Application 部分初始化的情况下保持健壮，并如实报告每个服务的状态。
        """
        from httpx import ASGITransport, AsyncClient

        from research_agent.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] in {"ok", "degraded"}
            assert "services" in data
            # 每个被探测的依赖都必须返回一个状态字符串，以便调用方一眼就能看出哪个服务宕机了。
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
