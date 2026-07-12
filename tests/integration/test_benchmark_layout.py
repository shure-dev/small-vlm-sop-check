"""Factory Egoのデータ/run境界を通常のpytestでも回帰検証する。"""
from __future__ import annotations

import json
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
    # unit数・run数はデータ側から導出する(データが増えてもテストを直書き修正しない)
    split = json.loads(
        (ROOT / "datasets/factory_ego/splits/development-v001.json").read_text(encoding="utf-8")
    )
    n_units = sum(len(units) for units in split["assignments"].values())
    n_runs = len([path for path in (ROOT / "runs").iterdir() if path.is_dir()])
    assert n_units > 0
    assert f"units={n_units} runs={n_runs}" in completed.stdout
