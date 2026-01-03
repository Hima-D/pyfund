# src/pyfundlib/utils/logger.py
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import structlog
from structlog.types import Processor


def setup_logging(level: int = logging.INFO, log_dir: Optional[str] = None) -> None:
    """
    Setup institutional-grade structured logging.
    """
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
    ]

    if sys.stderr.isatty():
        # Pretty printing for console
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        # JSON for production/files
        processors.append(structlog.processors.dict_tracebacks)
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    # Bridge standard logging to structlog if needed
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a structured logger.
    """
    return structlog.get_logger(name)


# Initialize with default settings
setup_logging()
logger = get_logger("pyfundlib")
