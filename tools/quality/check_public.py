#!/usr/bin/env python3
"""Audit files that would be committed to the public repository."""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


MAX_PUBLIC_FILE_BYTES = 95 * 1024 * 1024
SENSITIVE_PATTERNS = {
    # Literalを分割し、この監査script自身のpattern定義を自己検出しないようにする。
    "macOS user path": re.compile(rb"/" rb"Users/[^/\s]+/"),
    "private temp path": re.compile(rb"/(?:private/)?tmp/(?:claude|codex)[^\s\"']*"),
    "Windows user path": re.compile(rb"[A-Za-z]:\\Users\\[^\\\s]+\\"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "AWS access key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "generic API secret": re.compile(rb"(?i)\b(?:api[_-]?key|secret[_-]?key)\s*[:=]\s*[\"'][^\"']{12,}[\"']"),
}


def public_candidates(repo: Path) -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repo,
    )
    return [repo / raw.decode() for raw in output.split(b"\0") if raw]


def audit(repo: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    checked = 0
    for path in public_candidates(repo):
        if not path.is_file():
            continue
        checked += 1
        relative = path.relative_to(repo)
        size = path.stat().st_size
        if size > MAX_PUBLIC_FILE_BYTES:
            errors.append(f"oversized file ({size} bytes): {relative}")
        if str(relative).startswith("datasets/factory_ego/units/") and "/frames/" in f"/{relative}/":
            errors.append(f"gated Factory Ego frame would be public: {relative}")
        if size > 5 * 1024 * 1024:
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        for label, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(data):
                errors.append(f"{label}: {relative}")
    return checked, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    checked, errors = audit(args.repo.resolve())
    if errors:
        print(f"FAIL: {len(errors)} public-release issue(s)")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"PASS: {checked} public candidate files audited")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
