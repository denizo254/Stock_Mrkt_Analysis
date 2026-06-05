"""
src/utils.py
============
Cross-cutting helpers: structured logging, timing, and small I/O conveniences.

Keeping these in one place means every module logs with an identical format,
which makes a full pipeline run easy to read top-to-bottom.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Callable, Iterator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


def configure_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """
    Configure root logging exactly once.

    Parameters
    ----------
    level : logging level (e.g. ``logging.INFO``).
    log_file : optional path; if given, logs are mirrored to this file as well
        as stdout — handy for a permanent audit trail of a pipeline run.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    # Windows consoles default to cp1252, which cannot encode the Unicode
    # glyphs used in our log messages (→, ▶, ✔). Force the stdout stream to
    # UTF-8 where the platform allows it, and have the handler degrade
    # gracefully (replace) rather than crash if it still cannot.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover — non-reconfigurable stream
        pass

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    handlers: list[logging.Handler] = [stream_handler]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
    )
    # yfinance / urllib chatter is noisy; keep it to warnings.
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, configuring logging on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
@contextmanager
def timed(label: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Context manager that logs how long a block took."""
    log = logger or get_logger("timer")
    start = time.perf_counter()
    log.info("▶ %s ...", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.info("✔ %s done in %.2fs", label, elapsed)


def timeit(func: Callable) -> Callable:
    """Decorator variant of :func:`timed` for whole functions."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        log = get_logger(func.__module__)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        log.debug("%s() ran in %.3fs", func.__name__, time.perf_counter() - start)
        return result

    return wrapper


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def banner(title: str, char: str = "=", width: int = 78) -> str:
    """Return a centered section banner string for clean console reports."""
    title = f" {title} "
    return f"\n{title.center(width, char)}"
