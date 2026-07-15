import hashlib
import json
from pathlib import Path

from small_vlm_sop_check.apps.comparison import RunComparison
from small_vlm_sop_check.core.temporal import (
    evaluate_temporal,
    load_annotation,
    load_prediction,
)


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PATH = ROOT / "evaluations/factory_ego_marlin_reviewed6.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_development_evaluation_matches_locked_inputs_and_current_metrics():
    artifact = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
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
