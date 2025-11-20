# src/pyfundlib/utils/logger.py
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

# Create a dedicated pyfundlib logger (not root!)
logger = logging.getLogger("pyfundlib")
logger.propagate = False  # Prevent double logging if user has their own handlers

# Avoid adding handlers multiple times (idempotent)
if not logger.handlers:
    logger.setLevel(logging.INFO)

    # Console handler (pretty colors!)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Beautiful, compact format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a child logger with proper naming.
    
    Usage:
        logger = get_logger(__name__)           # e.g., pyfundlib.data.fetcher
        logger = get_logger("my_strategy")      # custom name
    """
    if name is None:
        return logger
    return logging.getLogger(f"pyfundlib.{name}" if not name.startswith("pyfundlib") else name)


# Optional: Add file logging with rotation (great for live trading)
def add_file_handler(
    log_dir: str = "logs",
    filename: str = "pyfundlib.log",
    level: int = logging.DEBUG,
    max_mb: int = 10,
    backup_count: int = 5,
) -> None:
    """
    Add rotating file handler (perfect for production/live systems).
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    from logging.handlers import RotatingFileHandler

    file_handler = RotatingFileHandler(
        log_dir / filename,
        maxBytes=max_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)
    logger.info(f"File logging enabled → {log_dir / filename}")


# Optional: Set log level from environment (great for production)
import os
if os.getenv("PYFUNDLIB_LOG_LEVEL") in ("DEBUG", "INFO", "WARNING", "ERROR"):
    logger.setLevel(os.getenv("PYFUNDLIB_LOG_LEVEL"))
    logger.info(f"Log level set to {os.getenv('PYFUNDLIB_LOG_LEVEL')} via env var")