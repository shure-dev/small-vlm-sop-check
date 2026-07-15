import json
from pathlib import Path

import pytest

from small_vlm_sop_check.evaluation.compare import compare_runs


def _prediction(run: Path, run_id: str, start: float, end: float):
    run.mkdir(parents=True)
    (run / "predictions").mkdir()
    (run / "run.yaml").write_text(
        f"run_id: {run_id}\nkind: prediction\nstatus: complete\nimmutable: true\n"
        "dataset:\n  id: demo\n  split: test\ntarget_units: [u1]\nground_truth_used: false\n",
        encoding="utf-8",
    )
    (run / "predictions" / "u1.json").write_text(json.dumps({
        "run_id": run_id, "unit_id": "u1", "method": "video_temporal_grounding",
        "interval_convention": "half-open_seconds",
        "events": {"action": [{"start_s": start, "end_s": end}]},
    }), encoding="utf-8")


def test_compare_requires_same_test_units_and_reports_delta(tmp_path: Path):
    dataset = tmp_path / "datasets" / "demo"
    (dataset / "annotations" / "human").mkdir(parents=True)
    (dataset / "dataset.yaml").write_text(
        "benchmark_state:\n  human_ground_truth:\n    status: complete\n", encoding="utf-8")
    (dataset / "annotations" / "human" / "u1.json").write_text(json.dumps({
        "unit_id": "u1", "interval_convention": "half-open_seconds",
        "events": {"action": [{"start_s": 1, "end_s": 3}]},
    }), encoding="utf-8")
    before, after = tmp_path / "before", tmp_path / "after"
    _prediction(before, "before", 0, 2)
    _prediction(after, "after", 1, 3)

    result = compare_runs(tmp_path, before, after)
    assert result["baseline"]["mean_tiou"] == pytest.approx(1 / 3, abs=1e-6)
    assert result["tuned"]["mean_tiou"] == 1.0
    assert result["delta"]["mean_tiou"] == pytest.approx(2 / 3, abs=1e-6)


def test_compare_rejects_partial_annotations(tmp_path: Path):
    dataset = tmp_path / "datasets" / "demo"
    dataset.mkdir(parents=True)
    (dataset / "dataset.yaml").write_text(
        "benchmark_state:\n  human_ground_truth:\n    status: partial\n", encoding="utf-8")
    before, after = tmp_path / "before", tmp_path / "after"
    _prediction(before, "before", 0, 2)
    _prediction(after, "after", 1, 3)
    with pytest.raises(ValueError, match="complete"):
        compare_runs(tmp_path, before, after)
