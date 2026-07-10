"""Factory Egoのデータ/run境界を通常のpytestでも回帰検証する。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_factory_ego_benchmark_integrity():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "benchmark" / "validate.py"), "--repo", str(ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "units=8 runs=5" in completed.stdout
