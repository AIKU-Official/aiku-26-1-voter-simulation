"""Minimal logging: tee to stdout and a per-phase logfile under logs/."""
from __future__ import annotations

import logging
from pathlib import Path

from .config import package_root


def get_logger(name: str, logfile: str | None = None) -> logging.Logger:
    """Return a logger that writes to stdout and (optionally) ``logs/<logfile>``."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if logfile:
        logdir = package_root() / "logs"
        logdir.mkdir(exist_ok=True)
        fh = logging.FileHandler(logdir / logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
