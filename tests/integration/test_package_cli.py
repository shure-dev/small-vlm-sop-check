"""Installed-package boundaries and Web app entrypoints."""
from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets" / "konro_inspection"


def run_module(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "small_vlm_sop_check", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_module_cli_detects_reference_fixture():
    completed = run_module(
        "detect",
        "--sop", str(DATASET / "sops" / "konro_inspection" / "konro_inspection.yaml"),
        "--answer-log", str(DATASET / "fixtures" / "reference_outputs" / "answer_log.json"),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "検出: 5/6 イベント" in completed.stdout


def test_read_only_viewer_has_a_packaged_entrypoint():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["scripts"]["sop-app"] == "small_vlm_sop_check.apps.launcher:main"
    assert project["scripts"]["sop-view"] == "small_vlm_sop_check.apps.launcher:main"
    assert project["scripts"]["sop-export-ms-swift"] == \
        "small_vlm_sop_check.training.ms_swift:main"
    assert project["scripts"]["sop-train"] == "small_vlm_sop_check.training.run:main"
    assert project["scripts"]["sop-compare"] == "small_vlm_sop_check.evaluation.compare:main"
