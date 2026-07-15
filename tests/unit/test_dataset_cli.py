import json

import pytest
import yaml

from small_vlm_sop_check.dataset_cli import (
    create_split,
    init_dataset,
    parse_events,
    validate_dataset,
)


def test_parse_events_builds_sop_events():
    assert parse_events(["inspect_part=作業者が部品を検査している"]) == [{
        "id": "inspect_part", "ask": "作業者が部品を検査している", "values": ["yes", "no"]
    }]


def test_parse_events_rejects_ambiguous_input():
    with pytest.raises(ValueError):
        parse_events(["Missing text"])


def test_init_dataset_creates_public_metadata_only(tmp_path):
    root = init_dataset(tmp_path, "my_factory", "My Factory", "Private factory clips")
    dataset = yaml.safe_load((root / "dataset.yaml").read_text())
    split = json.loads((root / "splits" / "development.json").read_text())
    assert dataset["media_policy"] == "local_not_committed"
    assert dataset["benchmark_state"]["human_ground_truth"]["revision"] == "human"
    assert split["assignments"]["dev_seen"] == []
    assert not (tmp_path / "data").exists()


def test_validate_dataset_detects_group_leakage_and_out_of_range_annotation(tmp_path):
    root = init_dataset(tmp_path, "demo", "Demo", "")
    split_path = root / "splits" / "development.json"
    split = json.loads(split_path.read_text())
    for unit_id, subset in (("u1", "validation"), ("u2", "test")):
        unit = root / "units" / unit_id
        unit.mkdir()
        sop = root / "sops" / unit_id
        sop.mkdir()
        (sop / "sop.yaml").write_text(
            f"sop:\n  id: {unit_id}\nevents:\n  - id: action\n    ask: action\n", encoding="utf-8")
        (unit / "meta.json").write_text(json.dumps({
            "unit_id": unit_id, "dataset_id": "demo",
            "sampling": {"fps": 2, "n_frames": 4},
            "source": {"source_group": "same"},
            "media": {"path": "frames", "availability": "local_not_committed"},
            "sop_ref": {"path": f"../../sops/{unit_id}/sop.yaml"},
        }), encoding="utf-8")
        split["assignments"][subset].append(unit_id)
    split_path.write_text(json.dumps(split), encoding="utf-8")
    annotation_dir = root / "annotations" / "human"
    (annotation_dir / "u1.json").write_text(json.dumps({
            "unit_id": "u1", "annotation_revision": "human",
            "interval_convention": "half-open_seconds",
        "event_labels": {"action": "action"},
        "events": {"action": [{"start_s": 0, "end_s": 3}]},
    }), encoding="utf-8")

    result = validate_dataset(tmp_path, "demo")
    assert result["error_count"] == 2
    assert any("group leakage" in error for error in result["errors"])
    assert any("動画範囲外" in error for error in result["errors"])


def test_create_split_is_deterministic_and_group_disjoint(tmp_path):
    root = init_dataset(tmp_path, "demo", "Demo", "")
    for index in range(10):
        unit = root / "units" / f"u{index}"
        unit.mkdir()
        (unit / "meta.json").write_text(json.dumps({
            "unit_id": f"u{index}", "source": {"source_group": f"worker{index // 2}"}
        }), encoding="utf-8")

    path = create_split(tmp_path, "demo", "benchmark", ["source_group"], 0.2, 0.2, 7)
    split = json.loads(path.read_text())
    assert {name: len(units) for name, units in split["assignments"].items()} == {
        "train": 6, "validation": 2, "test": 2,
    }
    for group in split["groups"].values():
        assert set(group["units"]).issubset(set(split["assignments"][group["split"]]))
