"""Structured logging with file rotation for the notetaker application."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_LOGGER_NAME = "notetaker"
_initialized = False


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    max_bytes: int = 10_485_760,  # 10 MB
    backup_count: int = 5,
    verbose: bool = False,
) -> logging.Logger:
    """Configure application logging with console + file handlers.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_dir: Directory for log files. If None, file logging is skipped.
        max_bytes: Max size per log file before rotation.
        backup_count: Number of rotated log files to keep.
        verbose: If True, force DEBUG level.

    Returns:
        Configured logger instance.
    """
    global _initialized

    logger = logging.getLogger(_LOGGER_NAME)

    if _initialized:
        return logger

    log_level = logging.DEBUG if verbose else getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    logger.propagate = False

    # Format
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(log_level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler (if log_dir provided)
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path / "notetaker.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    _initialized = True
    return logger


def get_logger(name: str = "") -> logging.Logger:
    """Get a child logger. Call setup_logging() first."""
    base = logging.getLogger(_LOGGER_NAME)
    if name:
        return base.getChild(name)
    return base
