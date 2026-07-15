import json
from pathlib import Path

from small_vlm_sop_check.apps.catalog import Unit
from small_vlm_sop_check.training.ms_swift import annotation_events_seconds, build_sample


def test_annotation_events_are_exported_as_half_open_seconds():
    gt = {"interval_convention": "half-open_seconds", "events": {
        "present": [{"start_s": 0.5, "end_s": 1.5}],
        "absent": None,
    }}
    assert annotation_events_seconds(gt) == {
        "present": [{"start_s": 0.5, "end_s": 1.5}],
        "absent": [],
    }


def test_build_sample_keeps_video_and_strict_json_answer(tmp_path: Path):
    sop = tmp_path / "sop.yaml"
    sop.write_text(
        "sop:\n  id: u1\n  name: Unit 1\nevents:\n"
        "  - id: action\n    ask: 作業者が部品を置いている\n    values: ['yes', 'no']\n",
        encoding="utf-8",
    )
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({
        "unit_id": "u1", "interval_convention": "half-open_seconds",
        "events": {"action": [{"start_s": 1.0, "end_s": 2.0}]}
    }), encoding="utf-8")
    video = tmp_path / "u1.mp4"
    video.write_bytes(b"video")
    unit = Unit("ds", "u1", sop, tmp_path / "frames", 2.0, gt)

    sample = build_sample(unit, video)

    assert sample["videos"] == [str(video.resolve())]
    answer = json.loads(sample["messages"][-1]["content"])
    assert answer == {"events": {"action": [{"start_s": 1.0, "end_s": 2.0}]}}
    assert "<video>" in sample["messages"][1]["content"]
