"""
logger.py
=========
Centralised logger factory for the entire pipeline.

WHY a custom factory instead of `logging.getLogger(...)` everywhere?
--------------------------------------------------------------------
1. Consistent format across all modules — every log line includes
   timestamp, level, module name, line number.
2. Dual output — console (so the developer sees progress in real time)
   AND a rotating file (so we have a permanent audit trail).
3. Idempotent — calling get_logger("foo") twice returns the *same*
   logger and does NOT re-attach handlers (which would cause duplicate
   log lines, the most common Python logging gotcha).
4. One import does it all: `from kd_pipeline.logger import get_logger`.
"""

# `logging` is Python's standard structured-logging library. It's the
# correct tool for any non-trivial script — much better than `print`
# because each log line carries metadata (level, time, source).
import logging

# `logging.handlers` provides higher-level handlers, including the
# rotating-file handler we use to cap log size on disk.
import logging.handlers
from typing import Set

# We import config to discover where to write log files. The import is
# *intra-package*, hence the leading dot.
from . import config


# Module-level cache so we never attach handlers to the same logger twice.
# `_configured_loggers` is a set of logger names we've already set up.
_configured_loggers: Set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """
    Return a fully-configured logger for the given `name`.

    Parameters
    ----------
    name : str
        The dotted name to use for the logger. By convention, callers pass
        `__name__` so the log lines show which module emitted them.

    Returns
    -------
    logging.Logger
        A logger that writes to both stdout and a rotating file.

    Notes
    -----
    Calling this function repeatedly with the same name is safe — the
    handlers are attached only the first time.
    """
    # Ask Python's logging module for a logger with this exact name.
    # Loggers are *singletons* keyed by name, so multiple modules calling
    # `get_logger("kd_pipeline.chunker")` get the same instance.
    logger = logging.getLogger(name)

    # If we've already configured this logger, return it as-is to avoid
    # attaching duplicate handlers (which would print every line twice).
    if name in _configured_loggers:
        return logger

    # The logger's *level* is the threshold below which records are
    # silently dropped. DEBUG = log everything; INFO = drop debug records.
    # We use DEBUG so handlers (which have their own thresholds) can
    # decide what to actually emit.
    logger.setLevel(logging.DEBUG)

    # `propagate=False` stops records from bubbling up to the root logger,
    # which would otherwise duplicate them through the root's handlers.
    logger.propagate = False

    # ── Make sure the log directory exists before opening a file in it ──
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Shared format string for both handlers.
    # Components:
    #   %(asctime)s   — timestamp like "2026-04-08 18:30:01,234"
    #   %(levelname)s — INFO / WARNING / ERROR / DEBUG
    #   %(name)s      — logger name (e.g. "kd_pipeline.chunker")
    #   %(lineno)d    — line number where the log call was made
    #   %(message)s   — the actual log message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler — writes to stdout / terminal ──────────────────
    # `StreamHandler()` defaults to sys.stderr; we leave it as-is so log
    # output doesn't pollute pipeline stdout (useful when piping data).
    console_handler = logging.StreamHandler()
    # INFO+ on the console: avoid spamming the terminal with debug noise.
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── Rotating file handler — keeps a permanent audit trail on disk ──
    # `RotatingFileHandler` rolls the log over once it exceeds maxBytes,
    # keeping `backupCount` old files. Together this caps total disk use.
    file_handler = logging.handlers.RotatingFileHandler(
        filename=config.LOG_DIR / "pipeline.log",
        maxBytes=10 * 1024 * 1024,   # 10 MB per file
        backupCount=5,                # keep 5 old files → ~50 MB max
        encoding="utf-8",             # explicit encoding avoids OS-default surprises
    )
    # DEBUG+ in the file: we want a complete record for postmortem analysis.
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Mark this logger as configured so we skip the setup next time.
    _configured_loggers.add(name)

    return logger
