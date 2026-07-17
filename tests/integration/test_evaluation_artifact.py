import hashlib
import json
from pathlib import Path

import pytest

from small_vlm_sop_check.apps.comparison import RunComparison
from small_vlm_sop_check.core.temporal import (
    evaluate_temporal,
    load_annotation,
    load_prediction,
)
from small_vlm_sop_check.evaluation.compare import _aggregate


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PATHS = [
    ROOT / "evaluations/factory_ego_marlin_new10_baseline.json",
    ROOT / "evaluations/factory_ego_marlin_final.json",
]
COMPARISON_EVALUATIONS = [
    (ROOT / "evaluations/factory_ego_marlin_stratified6.json", 6, 15),
    (ROOT / "evaluations/factory_ego_qwen3.5_stratified6.json", 6, 15),
    (ROOT / "evaluations/factory_ego_qwen3-vl-4b_stratified6.json", 6, 15),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("evaluation_path", EVALUATION_PATHS, ids=lambda path: path.stem)
def test_development_evaluation_matches_locked_inputs_and_current_metrics(
    evaluation_path: Path,
):
    artifact = json.loads(evaluation_path.read_text(encoding="utf-8"))
    run_id = artifact["prediction_run_id"]
    run_dir = ROOT / "runs" / run_id
    annotation_dir = ROOT / "datasets/factory_ego/annotations/human"

    assert artifact["formal_accuracy"] is False
    assert artifact["inputs"]["queries_sha256"] == _sha256(run_dir / "queries.json")
    assert artifact["inputs"]["run_manifest_sha256"] == _sha256(run_dir / "run.yaml")
    assert artifact["inputs"]["run_inputs_lock_sha256"] == _sha256(
        run_dir / "inputs.lock.json"
    )

    comparison = RunComparison.load(ROOT, run_id)
    current_reference = all(
        artifact["inputs"]["annotation_sha256"].get(unit_id)
        == _sha256(annotation_dir / f"{unit_id}.json")
        for unit_id in comparison.run["target_units"]
    )
    if not current_reference:
        pytest.skip("evaluation artifact references an older human annotation revision")
    summary = comparison.overall_summary
    published = artifact["metrics"]["all_reference_occurrences"]
    assert published["reference_occurrences"] == summary["reference_occurrences"]
    assert published["predicted_occurrences"] == summary["predicted_occurrences"]
    assert published["mean_tiou"] == summary["mean_tiou"]
    assert published["tiou_at_0_5"] == {
        key: summary["thresholds"]["tiou@0.5"][key]
        for key in ("precision", "recall", "f1")
    }

    single_span_ious = []
    for unit_id in comparison.run["target_units"]:
        annotation_path = annotation_dir / f"{unit_id}.json"
        prediction_path = run_dir / "predictions" / f"{unit_id}.json"
        assert artifact["inputs"]["annotation_sha256"][unit_id] == _sha256(annotation_path)
        assert artifact["inputs"]["prediction_sha256"][unit_id] == _sha256(prediction_path)

        annotation_doc = json.loads(annotation_path.read_text(encoding="utf-8"))
        prediction_doc = json.loads(prediction_path.read_text(encoding="utf-8"))
        result = evaluate_temporal(
            load_annotation(annotation_doc), load_prediction(prediction_doc)
        )
        for event_id, spans in annotation_doc["events"].items():
            if spans is None or len(spans) != 1:
                continue
            matches = [
                row for row in result["events"]
                if row["event"] == event_id and row["status"] == "match"
            ]
            single_span_ious.append(matches[0]["tiou"] if matches else 0.0)

    single = artifact["metrics"]["single_span_event_ids"]
    assert single["event_id_count"] == len(single_span_ious)
    assert single["mean_tiou"] == round(sum(single_span_ious) / len(single_span_ious), 6)


@pytest.mark.parametrize(
    ("evaluation_path", "unit_count", "event_count"),
    COMPARISON_EVALUATIONS,
    ids=lambda value: value.stem if isinstance(value, Path) else str(value),
)
def test_comparison_evaluation_matches_current_inputs_and_metrics(
    evaluation_path: Path, unit_count: int, event_count: int
):
    artifact = json.loads(evaluation_path.read_text(encoding="utf-8"))
    run_dir = ROOT / "runs" / artifact["prediction_run_id"]
    results = []

    assert artifact["formal_accuracy"] is False
    assert artifact["dataset"]["unit_count"] == unit_count
    assert artifact["dataset"]["event_count"] == event_count
    assert set(artifact["dataset"]["unit_ids"]) == set(
        artifact["per_unit_mean_tiou"]
    )
    assert artifact["inputs"]["queries_sha256"] == _sha256(run_dir / "queries.json")
    assert artifact["inputs"]["run_manifest_sha256"] == _sha256(run_dir / "run.yaml")
    assert artifact["inputs"]["run_inputs_lock_sha256"] == _sha256(
        run_dir / "inputs.lock.json"
    )

    for unit_id in artifact["dataset"]["unit_ids"]:
        annotation_path = (
            ROOT / "datasets/factory_ego/annotations/human" / f"{unit_id}.json"
        )
        prediction_path = run_dir / "predictions" / f"{unit_id}.json"
        assert artifact["inputs"]["annotation_sha256"][unit_id] == _sha256(
            annotation_path
        )
        assert artifact["inputs"]["prediction_sha256"][unit_id] == _sha256(
            prediction_path
        )
        annotation = load_annotation(
            json.loads(annotation_path.read_text(encoding="utf-8"))
        )
        prediction = load_prediction(
            json.loads(prediction_path.read_text(encoding="utf-8"))
        )
        event_ids = {
            row["event_id"]
            for row in artifact["per_event"]
            if row["unit_id"] == unit_id
        }
        results.append(evaluate_temporal(
            {event_id: annotation[event_id] for event_id in event_ids},
            {event_id: prediction[event_id] for event_id in event_ids},
        ))

    summary = _aggregate(results)
    assert artifact["metrics"] == {
        "reference_occurrences": summary["gt_occurrences"],
        "predicted_occurrences": summary["predicted_occurrences"],
        "mean_tiou": summary["mean_tiou"],
        "tiou_at_0_5": summary["thresholds"]["tiou@0.5"],
    }
