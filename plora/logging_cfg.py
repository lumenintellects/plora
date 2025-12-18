from __future__ import annotations

"""plora.logging_cfg - one-liner helper to enable consistent logging settings.
"""

from pathlib import Path
import logging
from datetime import datetime
import os


def setup_logging(level: str | int = "INFO", log_dir: str | None = None) -> Path | None:
    """Configure root logger to log to console and optional file.

    Returns the path to the log file if written, else None.
    """
    level_num = (
        getattr(logging, str(level).upper(), logging.INFO)
        if isinstance(level, str)
        else level
    )

    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level_num, format=fmt, datefmt=datefmt)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(log_dir) / f"plora_{ts}.log"
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        logging.getLogger().addHandler(fh)
        return log_path
    return None


# Initialise default logging as soon as module is imported, but only once.
if not logging.getLogger().hasHandlers():
    setup_logging(level=os.getenv("PLORA_LOG_LEVEL", "INFO"))
