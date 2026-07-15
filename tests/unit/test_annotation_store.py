from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from small_vlm_sop_check.apps.annotation_store import load_document, save_sop_and_document
from small_vlm_sop_check.apps.catalog import Unit


def _unit(tmp_path: Path) -> Unit:
    dataset = tmp_path / "datasets/demo"
    meta = dataset / "units/u1/meta.json"
    meta.parent.mkdir(parents=True)
    meta.write_text("{}", encoding="utf-8")
    (dataset / "dataset.yaml").write_text(yaml.safe_dump({
        "benchmark_state": {"human_ground_truth": {
            "status": "none", "revision": "human",
        }}
    }), encoding="utf-8")
    sop = dataset / "sops/u1/sop.yaml"
    sop.parent.mkdir(parents=True)
    sop.write_text("sop: {id: u1, name: U1}\nevents: []\n", encoding="utf-8")
    frames = tmp_path / "data/demo/units/u1/frames"
    frames.mkdir(parents=True)
    for index in range(4):
        (frames / f"f{index:04d}.jpg").write_bytes(b"jpg")
    return Unit(
        "demo", "u1", sop, frames, 2.0,
        dataset / "annotations/human/u1.json", n_frames=4, meta_path=meta,
    )


def test_save_human_japanese_event_and_seconds_atomically(tmp_path: Path):
    unit = _unit(tmp_path)
    sop = unit.load_sop()
    sop["events"].append({"id": "event_001", "ask": "部品を箱に入れる", "values": ["yes", "no"]})
    document = load_document(unit)
    document["events"] = {"event_001": [{"start_s": 0.5, "end_s": 1.5}]}

    save_sop_and_document(unit, sop, document)

    saved = json.loads(unit.gt_path.read_text(encoding="utf-8"))
    assert saved["events"]["event_001"] == [{"start_s": 0.5, "end_s": 1.5}]
    assert saved["event_labels"] == {"event_001": "部品を箱に入れる"}
    assert set(saved) == {
        "unit_id", "annotation_revision", "interval_convention", "event_labels", "events"
    }
    assert unit.load_sop()["events"][0]["ask"] == "部品を箱に入れる"
    state = yaml.safe_load((unit.meta_path.parents[2] / "dataset.yaml").read_text())
    assert state["benchmark_state"]["human_ground_truth"] == {
        "status": "complete", "revision": "human"
    }
    assert state["benchmark_state"]["formal_accuracy_available"] is False


def test_rejects_out_of_range_and_overlapping_spans(tmp_path: Path):
    unit = _unit(tmp_path)
    sop = unit.load_sop()
    sop["events"].append({"id": "event_001", "ask": "作業する"})
    document = load_document(unit)
    document["events"] = {"event_001": [
        {"start_s": 0.0, "end_s": 1.5},
        {"start_s": 1.0, "end_s": 3.0},
    ]}
    with pytest.raises(ValueError, match="重なっています|必要です"):
        save_sop_and_document(unit, sop, document)


def test_rejects_unnecessary_review_fields(tmp_path: Path):
    unit = _unit(tmp_path)
    document = load_document(unit)
    document["extra_workflow_state"] = "done"

    with pytest.raises(ValueError, match="不要なfield"):
        save_sop_and_document(unit, unit.load_sop(), document)


def test_partial_document_is_saved_as_partial_dataset_state(tmp_path: Path):
    unit = _unit(tmp_path)
    sop = unit.load_sop()
    sop["events"].append({"id": "event_001", "ask": "作業する"})
    document = load_document(unit)
    save_sop_and_document(unit, sop, document)
    assert json.loads(unit.gt_path.read_text())["events"] == {}
    state = yaml.safe_load((unit.meta_path.parents[2] / "dataset.yaml").read_text())
    assert state["benchmark_state"]["human_ground_truth"]["status"] == "partial"
