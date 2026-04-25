"""TCP-level reachability probe for Postgres URIs.

Why this exists
---------------
``psycopg_pool.ConnectionPool`` is *eager*: as soon as you instantiate
it, a background worker tries to open the minimum number of
connections, and if that fails it keeps **retrying every 5 seconds
for up to 300 seconds** before giving up — and then starts another
300-second cycle. On a developer laptop without Postgres running,
this means:

  1. Lifespan ``setup()`` blocks ~30 s waiting for the first connect
     attempt to time out.
  2. Even after we ``except`` the failure and fall back to
     MemorySaver, the pool object stays alive and its background
     thread keeps logging "connection timeout expired" every few
     minutes, polluting the log.
  3. On Windows asyncio (ProactorEventLoop), the synchronous
     reconnect attempts run on a worker thread but their I/O still
     contends with the main loop badly enough to make HTTP handlers
     hang for tens of seconds at a time. Observed empirically:
     ``GET /health`` blocked for >25 minutes during testing while
     pool-1 / pool-2 reconnection cycles fought for the loop.

A 200-millisecond TCP probe sidesteps all of this. If the port is
not even ACK-ing SYN, we know there's no point creating a pool at
all and we can skip straight to the in-memory fallback.

Notes
  - This only checks reachability, not authentication. A reachable
    but mis-credentialed Postgres will still proceed to ``setup()``
    and surface the credential error there, which is the correct
    place to handle it (you want to know your password is wrong).
  - We intentionally use the synchronous ``socket`` module rather
    than ``asyncio.open_connection``. The probe runs at lifespan
    startup before the event loop is doing any user work, so the
    blocking call is not contended; ``asyncio.open_connection``
    would add complexity (DNS resolution, transport setup) for no
    measurable gain at this call site.
"""

from __future__ import annotations

import socket
from urllib.parse import urlparse

from loguru import logger

# A 0.2-second probe is generous enough to cover loopback latency
# under load yet short enough that a missing-Postgres developer feels
# no perceptible slowdown at startup.
DEFAULT_PROBE_TIMEOUT_S: float = 0.2


def _parse_host_port(uri: str) -> tuple[str, int] | None:
    """Extract ``(host, port)`` from a Postgres URI.

    Accepts both the libpq-style ``postgresql://`` URI and the
    SQLAlchemy-style ``postgresql+driver://`` URI; the dialect
    suffix is irrelevant to the network layer.

    Returns ``None`` if the URI is unparseable or omits a host —
    callers should treat that as "skip the probe and let the pool
    surface the error itself".
    """
    if not uri:
        return None
    try:
        parsed = urlparse(uri)
    except (ValueError, TypeError):
        return None
    if not parsed.hostname:
        return None
    return (parsed.hostname, parsed.port or 5432)


def is_postgres_reachable(
    uri: str, timeout_s: float = DEFAULT_PROBE_TIMEOUT_S
) -> bool:
    """Return True iff a TCP connection to the Postgres host succeeds.

    Returns False (with a debug log) on any of:
      * URI is empty or unparseable
      * DNS resolution fails
      * TCP connect times out within ``timeout_s``
      * connection is actively refused (ECONNREFUSED)

    On a successful probe the socket is closed immediately; we are
    not negotiating the Postgres startup message, only checking that
    *something* is listening.
    """
    target = _parse_host_port(uri)
    if target is None:
        logger.debug("Postgres reachability probe skipped: unparseable URI")
        return False

    host, port = target
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    try:
        sock.connect((host, port))
    except (socket.timeout, ConnectionRefusedError, OSError) as exc:
        logger.debug(
            "Postgres TCP probe failed for {}:{} ({}); skipping pool init",
            host,
            port,
            exc.__class__.__name__,
        )
        return False
    finally:
        try:
            sock.close()
        except OSError:
            pass
    return True


__all__ = ["DEFAULT_PROBE_TIMEOUT_S", "is_postgres_reachable"]
