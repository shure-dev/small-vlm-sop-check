import importlib.util
from pathlib import Path


PATH = Path(__file__).resolve().parents[2] / "tools" / "benchmark" / "run_qwen_video_prediction.py"
SPEC = importlib.util.spec_from_file_location("run_qwen_video_prediction", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_parse_span_accepts_fenced_json_and_absent():
    assert MODULE.parse_span('```json\n{"start_s": 1.5, "end_s": 4}\n```') == (1.5, 4.0)
    assert MODULE.parse_span('{"start_s": null, "end_s": null}') is None


def test_parse_span_rejects_out_of_video_and_partial_null():
    assert MODULE.parse_span('{"start_s": 19, "end_s": 21}') is None
    assert MODULE.parse_span('{"start_s": null, "end_s": 2}') is None
