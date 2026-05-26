"""基于 loguru 的结构化日志配置。"程序在做什么"的实时记录"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(level: str = "INFO", *, log_file_path: str | None = None) -> None:
    """配置 loguru 结构化日志格式。

    用 loguru 做结构化日志，每条日志包含时间戳、日志级别、模块名、行号和请求 ID（通过中间件 RequestIdMiddleware 注入），同时输出到 stderr 和滚动日志文件。

    日志始终输出到 stderr。当 ``log_file_path`` 为非空字符串时，相同的日志会镜像写入该路径（目录按需自动创建）。

    每行日志包含 ``{extra[request_id]}`` — 默认值为 ``"-"``，
    由 ``RequestIdMiddleware`` 通过 ``logger.contextualize()`` 在每个请求中覆盖。
    """
    logger.remove()
    logger.configure(extra={"request_id": "-"})

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<dim>{extra[request_id]}</dim> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=log_format,
        level=level.upper(),
        colorize=True,
    )

    path_raw = log_file_path or ""
    if path_raw.strip():
        fp = Path(path_raw)
        fp.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(fp),
            format=log_format,
            level=level.upper(),
            rotation="50 MB",
            retention="7 days",
            compression="gz",
        )

    logger.info("Logging initialized at {} level", level)
