from __future__ import annotations

import json
from pathlib import Path

from small_vlm_sop_check.apps.catalog import Unit
from small_vlm_sop_check.apps.comparison import RunComparison, discover_run_comparisons


def test_common_viewer_comparison_keeps_text_spans_and_overall_metric(tmp_path: Path):
    dataset = tmp_path / "datasets/demo"
    reference_dir = dataset / "annotations/human"
    reference_dir.mkdir(parents=True)
    (reference_dir / "u1.json").write_text(json.dumps({
        "unit_id": "u1", "annotation_revision": "human",
        "interval_convention": "half-open_seconds",
        "events": {"pick": [{"start_s": 1, "end_s": 3}]},
    }), encoding="utf-8")
    sop_path = dataset / "sops/u1/sop.yaml"
    sop_path.parent.mkdir(parents=True)
    sop_path.write_text(
        "sop: {id: u1, name: U1}\nevents:\n- id: pick\n  ask: 部品を持ち上げる\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "runs/run1"
    (run_dir / "predictions").mkdir(parents=True)
    (run_dir / "run.yaml").write_text(
        "run_id: run1\nkind: prediction\nstatus: complete\n"
        "model:\n  name: Model 2B\ndataset:\n  id: demo\ntarget_units: [u1]\n"
        "inference:\n  query_ontology: unit_sop\n",
        encoding="utf-8",
    )
    (run_dir / "predictions/u1.json").write_text(json.dumps({
        "run_id": "run1", "unit_id": "u1", "method": "temporal_grounding",
        "interval_convention": "half-open_seconds",
        "events": {"pick": [{"start_s": 2, "end_s": 3}]},
    }), encoding="utf-8")
    unit = Unit("demo", "u1", sop_path, tmp_path / "frames", 2.0,
                tmp_path / "gt.json")

    comparison = RunComparison.load(tmp_path, "run1").for_unit(unit)

    assert comparison is not None
    assert comparison["events"][0]["text"] == "部品を持ち上げる"
    assert comparison["events"][0]["reference_spans"] == [{"start_s": 1, "end_s": 3}]
    assert comparison["events"][0]["prediction_spans"] == [{"start_s": 2, "end_s": 3}]
    assert comparison["events"][0]["tiou"] == 0.5
    assert comparison["overall_summary"]["mean_tiou"] == 0.5
    assert list(discover_run_comparisons(tmp_path)) == ["run1"]
