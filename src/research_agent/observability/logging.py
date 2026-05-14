"""Structured logging configuration with loguru."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(level: str = "INFO", *, log_file_path: str | None = None) -> None:
    """Configure loguru with structured formatting.

    Logs always go to stderr. When ``log_file_path`` is a non-empty
    string the same messages are mirrored to that path (dirs created
    on demand).

    Every log line includes ``{extra[request_id]}`` — set to ``"-"``
    by default and overridden per-request by ``RequestIdMiddleware``
    via ``logger.contextualize()``.
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
