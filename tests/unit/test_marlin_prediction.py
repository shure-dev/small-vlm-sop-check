import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "benchmark" / "run_marlin_prediction.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("run_marlin_prediction", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_marlin_video_width_preserves_small_hand_actions():
    assert MODULE.VIDEO_WIDTH >= 640


def test_result_span_accepts_valid_marlin_result():
    assert MODULE.result_span({"span": [1.5, 3.0], "format_ok": True}) == (1.5, 3.0)


def test_result_span_rejects_invalid_ranges():
    assert MODULE.result_span({"span": [3.0, 1.5]}) is None
    assert MODULE.result_span({"raw": "not found"}) is None


def test_normalize_prediction_preserves_seconds():
    raw = {"events": {
        "event_a": {"result": {"span": [0.5, 1.0]}},
        "event_b": {"result": {"span": None}},
    }}

    prediction = MODULE.normalize_prediction("run", "unit", raw)

    assert prediction["method"] == "temporal_grounding"
    assert prediction["events"] == {
        "event_a": [{"start_s": 0.5, "end_s": 1.0}],
        "event_b": None,
    }


def test_normalize_prediction_preserves_fractional_boundaries():
    raw = {"events": {"event": {"result": {"span": [0.6, 1.1]}}}}

    prediction = MODULE.normalize_prediction("run", "unit", raw)

    assert prediction["events"]["event"] == [{"start_s": 0.6, "end_s": 1.1}]
