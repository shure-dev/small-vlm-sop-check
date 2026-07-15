import pytest

from small_vlm_sop_check.core.temporal import (
    TimeSpan,
    evaluate_temporal,
    load_annotation,
    load_prediction,
    prediction_document,
    temporal_iou,
)


def test_temporal_iou_uses_continuous_half_open_intervals():
    assert temporal_iou(TimeSpan(1, 3), TimeSpan(2, 4)) == pytest.approx(1 / 3)
    assert temporal_iou(TimeSpan(1, 2), TimeSpan(2, 3)) == 0


def test_annotation_and_prediction_share_one_seconds_contract():
    events = load_annotation({
        "unit_id": "unit",
        "annotation_revision": "human",
        "interval_convention": "half-open_seconds",
        "events": {"action": [{"start_s": 0.5, "end_s": 1.5}], "absent": None},
    })
    assert events == {"action": [TimeSpan(0.5, 1.5)], "absent": None}
    prediction = prediction_document("run", "unit", "temporal_grounding", events)
    assert load_prediction(prediction) == events


def test_temporal_evaluation_counts_misses_false_detections_and_absence():
    annotation = {
        "a": [TimeSpan(0, 2), TimeSpan(4, 6)],
        "b": None,
        "missing": [TimeSpan(1, 2)],
    }
    prediction = {
        "a": [TimeSpan(0, 2), TimeSpan(8, 9)],
        "b": None,
    }
    result = evaluate_temporal(annotation, prediction)
    assert result["summary"]["mean_tiou"] == pytest.approx(1 / 3)
    assert result["summary"]["prediction_missing_events"] == 1
    at_05 = result["summary"]["thresholds"]["tiou@0.5"]
    assert at_05 == {"tp": 1, "fp": 1, "fn": 2,
                      "precision": 0.5, "recall": 0.333333, "f1": 0.4}
    assert any(row["status"] == "true_absent" for row in result["events"])


def test_optimal_occurrence_pairing_is_not_forced_by_position():
    annotation = {"a": [TimeSpan(0, 10), TimeSpan(10, 20)]}
    prediction = {"a": [TimeSpan(10, 20), TimeSpan(0, 10)]}
    result = evaluate_temporal(annotation, prediction)
    assert result["summary"]["mean_tiou"] == 1.0
