"""Package resources and development-repository defaults for browser apps."""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def repository_root() -> Path:
    """Return the checkout root when running in development, otherwise cwd."""
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "datasets").is_dir():
            return candidate
    return Path.cwd()


def template_text(name: str) -> str:
    return files("small_vlm_sop_check.apps").joinpath("templates", name).read_text(encoding="utf-8")
