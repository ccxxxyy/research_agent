"""Unit tests for the Postgres reachability probe.

Why test this
-------------
A regression that re-introduces the eager ``ConnectionPool`` would
re-introduce the symptom we solved: a missing Postgres at dev/test
time hangs the lifespan and the HTTP handlers. Locking the probe's
contract makes that regression cheap to catch.

We do NOT spin up a real Postgres in these tests. Instead we point
the probe at:
  * an explicitly closed loopback port (refused TCP)
  * a non-routable RFC-5737 address (timeout, not refused)
  * a port we are actively listening on (success)

The third case uses an ephemeral ``socket.bind((127.0.0.1, 0))`` so
the test cannot collide with anything else on the host.
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
    """Yield a random free TCP port that has a real listener attached.

    We bind, listen with backlog=1, and yield the port. The socket
    stays open for the duration of the test so the probe will see
    a SYN-ACK rather than a RST.
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
        # ``postgresql:///dbname`` is libpq's "use Unix socket"
        # syntax — no TCP host to probe, so we deliberately punt.
        assert _parse_host_port("postgresql:///mydb") is None


class TestIsPostgresReachable:
    def test_empty_uri_returns_false(self) -> None:
        assert is_postgres_reachable("") is False

    def test_listening_port_returns_true(self) -> None:
        with _listening_socket() as port:
            uri = f"postgresql://u:p@127.0.0.1:{port}/x"
            assert is_postgres_reachable(uri) is True

    def test_closed_loopback_port_returns_false(self) -> None:
        # Port 1 is the TCPMUX well-known port and is essentially
        # never listening on a developer machine. Using a port we
        # *expect* to be closed gives us a deterministic ECONNREFUSED.
        uri = "postgresql://u:p@127.0.0.1:1/x"
        assert is_postgres_reachable(uri, timeout_s=0.5) is False

    def test_unroutable_address_times_out_quickly(self) -> None:
        """RFC-5737 documentation block 192.0.2.0/24 is guaranteed
        not to route on the public Internet. The probe should
        return False *within* the configured timeout, proving the
        fast-fail behavior callers depend on."""
        import time

        uri = "postgresql://u:p@192.0.2.1:5432/x"
        start = time.monotonic()
        result = is_postgres_reachable(uri, timeout_s=0.3)
        elapsed = time.monotonic() - start

        assert result is False
        # Allow generous slack for slow CI; the point is "well below
        # the 30-second default that ConnectionPool would block on".
        assert elapsed < 2.0, f"probe took {elapsed:.2f}s, expected <2s"
