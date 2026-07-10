"""Installed-package boundaries: module CLI and packaged HTML resources."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from small_vlm_sop_check.apps.resources import template_text


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


def test_module_cli_judges_reference_fixture():
    completed = run_module(
        "judge",
        "--sop", str(DATASET / "sops" / "konro_inspection" / "correct.yaml"),
        "--answer-log", str(DATASET / "fixtures" / "reference_outputs" / "answer_log.json"),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "総合判定: PASS" in completed.stdout


def test_browser_templates_are_packaged_resources():
    annotator = template_text("annotator.html")
    replay = template_text("replay.html")
    assert "__ANNOTATOR_DATA__" in annotator
    assert "__REPLAY_DATA__" in replay
