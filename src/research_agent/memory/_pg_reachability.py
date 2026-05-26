"""Postgres URI 的 TCP 层可达性探测。

为什么需要这个模块
------------------
启动时探测 Postgres 是否可用：在启动时花 0.2 秒试一下 Postgres 数据库能不能连上。能连就用 Postgres，连不上就直接跳过，用 SQLite 或内存代替。
``psycopg_pool.ConnectionPool`` 是 立即执行 的：一旦实例化，后台工作线程就会尝试建立最小连接数。
如果失败，它会 每 5 秒重试一次，持续 300 秒 才放弃，然后再开始下一个 300 秒的循环。在没有 Postgres 时意味：

  1. 生命周期 ``setup()`` 会阻塞约 30 秒，等待第一次连接尝试超时。
  2. 即使 ``except`` 了失败并回退到 MemorySaver，连接池对象仍然存活，其后台线程每隔几分钟就会记录 "connection timeout expired"，污染日志。
  3. 在 Windows asyncio（ProactorEventLoop）上，同步重连尝试运行在工作线程中，但其 I/O 仍然会严重争夺主循环，导致 HTTP 处理程序挂起数十秒。
     实测观察到：在 pool-1 / pool-2 重连循环争夺事件循环期间，``GET /health`` 阻塞了超过 25 分钟。

一个 200 毫秒的 TCP 探测可避免以上所有问题。如果端口甚至没有ACK SYN，就知道没有必要创建连接池，可以直接跳到内存回退方案。

注意事项
  - 本模块只检查可达性，不检查认证。一个可达但凭证错误的 Postgres 仍然会进入 ``setup()`` 并在那里暴露凭证错误——这是处理该问题的正确位置（需要知道密码是否错误）。
  - 故意使用同步的 ``socket`` 模块而非 ``asyncio.open_connection``。
    探测在生命周期启动时运行，此时事件循环还没有处理任何用户工作，因此阻塞调用不会产生争用；``asyncio.open_connection`` 会增加复杂性（DNS 解析、传输层建立），
    但在此调用点没有可衡量的收益。
"""

from __future__ import annotations

import socket
from urllib.parse import urlparse

from loguru import logger

# 0.2 秒的探测可覆盖高负载下的回环延迟，同时又足够短，让无运行 Postgres 在启动时感受不到明显的延迟。
DEFAULT_PROBE_TIMEOUT_S: float = 0.2


def _parse_host_port(uri: str) -> tuple[str, int] | None:
    """从 Postgres URI 中提取 ``(host, port)``。

    同时接受 libpq 风格的 ``postgresql://`` URI 和 SQLAlchemy 风格的 ``postgresql+driver://`` URI；

    如果 URI 无法解析或缺少主机，函数不报错，返回 ``None`` —— 调用方视其为"跳过探测，让ConnectionPool(uri) 去连接，连接池自行暴露更准确的错误"。
    """
    if not uri:
        return None
    try:
        parsed = urlparse(uri)
    except (ValueError, TypeError):
        return None
    if not parsed.hostname:
        return None
    host: str = parsed.hostname
    return host, parsed.port or 5432


def is_postgres_reachable(
    uri: str, timeout_s: float = DEFAULT_PROBE_TIMEOUT_S
) -> bool:
    """当 TCP 连接到 Postgres 主机成功时返回 True。

    以下任一情况返回 False（并记录 debug 日志）：
      * URI 为空或无法解析
      * DNS 解析失败
      * TCP 连接在 ``timeout_s`` 内超时
      * 连接被主动拒绝（ECONNREFUSED）

    探测成功后立即关闭 socket；并不协商 Postgres 启动消息，不连接，只是检查是否有东西在监听。
        从 URI 中提取主机名和端口号
        用最底层的 TCP 连接试探一下那个端口有没有东西在监听
        立即关闭连接，返回 True/False
        不执行 Postgres 协议握手，不验证密码，不创建连接池。连接池的创建在 checkpointer.py 里，只有探测返回 True 时才进行。
    """
    target = _parse_host_port(uri)
    if target is None:
        logger.debug("Postgres reachability probe skipped 可达性探测被跳过: unparseable URI")
        return False

    host, port = target
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    try:
        sock.connect((host, port))
    except (socket.timeout, ConnectionRefusedError, OSError) as exc:
        logger.debug(
            "Postgres TCP probe failed for {}:{} ({}); skipping pool init 跳过池初始化",
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
