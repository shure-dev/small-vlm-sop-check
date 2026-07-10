"""Public repository candidates must not expose local paths or gated frames."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_public_release_audit():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "quality" / "check_public.py"), "--repo", str(ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
