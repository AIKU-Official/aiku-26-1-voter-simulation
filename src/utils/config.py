"""Config + path helpers.

All pipeline paths are relative to the package root (the directory that holds
``configs/``, ``data/``, ``specs/``). Scripts run from there via ``uv run``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def package_root() -> Path:
    """Return the package root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a config-relative path against the package root.

    Absolute paths are returned unchanged.
    """
    p = Path(path)
    return p if p.is_absolute() else package_root() / p
