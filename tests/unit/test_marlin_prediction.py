import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "benchmark" / "run_marlin_prediction.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("run_marlin_prediction", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_result_span_accepts_valid_marlin_result():
    assert MODULE.result_span({"span": [1.5, 3.0], "format_ok": True}) == (1.5, 3.0)


def test_result_span_rejects_invalid_ranges():
    assert MODULE.result_span({"span": [3.0, 1.5]}) is None
    assert MODULE.result_span({"raw": "not found"}) is None


def test_normalize_prediction_maps_seconds_to_frame_answers(monkeypatch, tmp_path):
    meta = tmp_path / "meta.json"
    meta.write_text('{"sampling":{"n_frames":5}}', encoding="utf-8")
    monkeypatch.setattr(MODULE, "unit_fps", lambda _unit: 2.0)
    monkeypatch.setattr(MODULE, "unit_paths", lambda _unit: {"meta": meta})
    raw = {"events": {
        "event_a": {"result": {"span": [0.5, 1.0]}},
        "event_b": {"result": {"span": None}},
    }}

    prediction = MODULE.normalize_prediction("run", "unit", raw)

    assert [frame["answers"]["event_a"] for frame in prediction["frames"]] == [
        "no", "yes", "yes", "no", "no"
    ]
    assert all(frame["answers"]["event_b"] == "unclear" for frame in prediction["frames"])


def test_normalize_prediction_covers_fractional_boundary_frames(monkeypatch, tmp_path):
    meta = tmp_path / "meta.json"
    meta.write_text('{"sampling":{"n_frames":6}}', encoding="utf-8")
    monkeypatch.setattr(MODULE, "unit_fps", lambda _unit: 2.0)
    monkeypatch.setattr(MODULE, "unit_paths", lambda _unit: {"meta": meta})
    raw = {"events": {"event": {"result": {"span": [0.6, 1.1]}}}}

    prediction = MODULE.normalize_prediction("run", "unit", raw)

    assert [frame["answers"]["event"] for frame in prediction["frames"]] == [
        "no", "yes", "yes", "yes", "no", "no"
    ]
