"""Webアプリが操作するrepository rootを解決する。"""
from __future__ import annotations

import os
from pathlib import Path


def repository_root() -> Path:
    """Return the checkout root when running in development, otherwise cwd."""
    configured = os.environ.get("SOP_REPOSITORY_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not (root / "datasets").is_dir():
            raise RuntimeError(f"SOP_REPOSITORY_ROOTにdatasets/がありません: {root}")
        return root
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "datasets").is_dir():
            return candidate
    return Path.cwd()
