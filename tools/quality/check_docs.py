#!/usr/bin/env python3
"""Check local Markdown/HTML links in repository documentation."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote


MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_LINK = re.compile(r"(?:src|href)=[\"']([^\"']+)[\"']")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "data:", "app://")


def documentation_files(repo: Path) -> list[Path]:
    roots = [*repo.glob("*.md"), repo / "docs", repo / "datasets",
             repo / "evaluations", repo / "reports", repo / "tools"]
    files: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix == ".md":
            files.add(root)
        elif root.is_dir():
            files.update(root.rglob("*.md"))
    return sorted(files)


def local_targets(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    targets = MARKDOWN_LINK.findall(text) + HTML_LINK.findall(text)
    return [target.strip().strip("<>") for target in targets]


def check(repo: Path) -> tuple[int, list[str]]:
    checked = 0
    errors: list[str] = []
    for doc in documentation_files(repo):
        for raw_target in local_targets(doc):
            target = raw_target.split()[0]
            if not target or target.startswith("#") or target.startswith(SKIP_PREFIXES):
                continue
            path_part = unquote(target.split("#", 1)[0].split("?", 1)[0])
            if not path_part:
                continue
            resolved = (repo / path_part.lstrip("/")) if target.startswith("/") else (doc.parent / path_part)
            checked += 1
            if not resolved.exists():
                errors.append(f"{doc.relative_to(repo)} -> {raw_target}")
    return checked, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    checked, errors = check(args.repo.resolve())
    if errors:
        print(f"FAIL: {len(errors)} broken local link(s)")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"PASS: {checked} local documentation links")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
