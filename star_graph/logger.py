"""Centralized structured logging for star-graph-memory.

Replaces ad-hoc print() and self.log: list[str] with standard library logging.
Supports configurable levels, structured key=value output, and log file output.

Usage:
    from star_graph.logger import get_logger
    log = get_logger(__name__)
    log.info("Sleep N2_Merge complete", extra={"merged": 5, "duration_ms": 120})
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Optional

_LOG_FORMAT = "%(asctime)s [%(levelname).1s] %(name)s | %(message)s"
_LOG_DATE_FORMAT = "%H:%M:%S"

# Module-level registry
_loggers: dict[str, logging.Logger] = {}
_initialized = False
_log_level = logging.INFO
_log_file: str | None = None


def init_logging(level: int = logging.INFO,
                 log_file: str | None = None,
                 format_str: str | None = None) -> None:
    """Initialize the root star_graph logger. Call once at startup.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path for file output.
        format_str: Optional custom format string.
    """
    global _initialized, _log_level, _log_file
    _log_level = level
    _log_file = log_file

    fmt = format_str or _LOG_FORMAT
    root = logging.getLogger("star_graph")
    root.setLevel(level)

    # Remove existing handlers
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt, datefmt=_LOG_DATE_FORMAT))
    root.addHandler(console)

    # File handler (optional)
    if log_file:
        import os
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
        root.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a module, auto-prefixed with star_graph.

    Args:
        name: Usually __name__ from the calling module.

    Returns:
        A logging.Logger instance.
    """
    full_name = f"star_graph.{name.split('.')[-1]}" if name.startswith("star_graph") \
        else f"star_graph.{name}"
    if full_name not in _loggers:
        _loggers[full_name] = logging.getLogger(full_name)
    return _loggers[full_name]


def shutdown() -> None:
    """Flush and close all handlers."""
    logging.getLogger("star_graph").handlers.clear()
    _loggers.clear()
    global _initialized
    _initialized = False


class StructuredLog:
    """Mixin for classes that want structured log output without print().

    Usage:
        class SleepCycle(StructuredLog):
            def run(self):
                self.log_info("Sleep started", phase="N1_Replay")
    """

    def __init__(self):
        self._logger = logging.getLogger(
            f"star_graph.{self.__class__.__name__}")

    def log_debug(self, msg: str, **extra) -> None:
        self._logger.debug(msg, extra=extra)

    def log_info(self, msg: str, **extra) -> None:
        self._logger.info(msg, extra=extra)

    def log_warning(self, msg: str, **extra) -> None:
        self._logger.warning(msg, extra=extra)

    def log_error(self, msg: str, **extra) -> None:
        self._logger.error(msg, extra=extra)
