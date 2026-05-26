"""Postgres 可达性探针的单元测试。

为什么要测试
-------------
如果回归性地重新引入了急切的 ``ConnectionPool``，就会重新引入已解决的症状：开发/测试时缺少 Postgres 会挂起 lifespan和 HTTP 处理器。锁定探针的契约使此类回归易于发现。

这些测试不启动真实 Postgres。而是将探针指向：
  * 显式关闭的回环端口（TCP 拒绝连接）
  * 不可路由的 RFC-5737 地址（超时而非拒绝）
  * 正在监听的端口（成功）

第三种情况使用临时 ``socket.bind((127.0.0.1, 0))``，因此测试不会与主机上的任何其他服务冲突。
"""

from __future__ import annotations

import socket
from contextlib import contextmanager
from typing import Iterator

from research_agent.memory._pg_reachability import (
    _parse_host_port,
    is_postgres_reachable,
)


@contextmanager
def _listening_socket() -> Iterator[int]:
    """yield 一个附带真实监听器的随机空闲 TCP 端口。

    绑定、以 backlog=1 监听并 yield 该端口。socket 在测试期间保持打开，因此探针会看到 SYN-ACK 而非 RST。
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    try:
        yield s.getsockname()[1]
    finally:
        s.close()


class TestParseHostPort:
    def test_libpq_uri(self) -> None:
        assert _parse_host_port("postgresql://u:p@db.example.com:6543/x") == (
            "db.example.com",
            6543,
        )

    def test_sqlalchemy_dialect_suffix_is_ignored(self) -> None:
        assert _parse_host_port("postgresql+asyncpg://u:p@host:5432/x") == (
            "host",
            5432,
        )

    def test_default_port_is_5432(self) -> None:
        assert _parse_host_port("postgresql://u@host/x") == ("host", 5432)

    def test_empty_string_returns_none(self) -> None:
        assert _parse_host_port("") is None

    def test_no_host_returns_none(self) -> None:
        # ``postgresql:///dbname`` 是 libpq 的"使用 Unix socket"
        # 语法 — 无 TCP 主机可探测，因此我们有意跳过。
        assert _parse_host_port("postgresql:///mydb") is None


class TestIsPostgresReachable:
    def test_empty_uri_returns_false(self) -> None:
        assert is_postgres_reachable("") is False

    def test_listening_port_returns_true(self) -> None:
        with _listening_socket() as port:
            uri = f"postgresql://u:p@127.0.0.1:{port}/x"
            assert is_postgres_reachable(uri) is True

    def test_closed_loopback_port_returns_false(self) -> None:
        # 端口 1 是 TCPMUX 知名端口，开发机器上基本不会监听。
        # 使用一个我们*预期*已关闭的端口可获得确定性的 ECONNREFUSED。
        uri = "postgresql://u:p@127.0.0.1:1/x"
        assert is_postgres_reachable(uri, timeout_s=0.5) is False

    def test_unroutable_address_times_out_quickly(self) -> None:
        """RFC-5737 文档块 192.0.2.0/24 保证不会在公网路由。
        探针应在配置的超时时间内返回 False，证明调用者所依赖的快速失败行为。"""
        import time

        uri = "postgresql://u:p@192.0.2.1:5432/x"
        start = time.monotonic()
        result = is_postgres_reachable(uri, timeout_s=0.3)
        elapsed = time.monotonic() - start

        assert result is False
        # 为慢速 CI 留出充裕余量；关键是"远低于 ConnectionPool 会阻塞的 30 秒默认值"。
        assert elapsed < 2.0, f"probe took {elapsed:.2f}s, expected <2s"
