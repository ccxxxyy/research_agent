"""Structured logging configuration with loguru."""

from __future__ import annotations

import sys

from loguru import logger


def setup_logging(level: str = "INFO") -> None:
    """Configure loguru with structured formatting for production.

    Log format includes timestamp, level, module, and message.
    In production, logs are JSON-formatted for ELK/Loki ingestion.
    """
    logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=log_format,
        level=level.upper(),
        colorize=True,
    )

    logger.add(
        "logs/research_agent.log",
        format=log_format,
        level=level.upper(),
        rotation="50 MB",
        retention="7 days",
        compression="gz",
    )

    logger.info("Logging initialized at {} level", level)
