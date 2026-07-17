import hashlib
import json
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from small_vlm_sop_check.apps.server import create_app


ROOT = Path(__file__).resolve().parents[2]


def test_api_loads_all_complete_units_and_available_model_comparisons():
    client = TestClient(create_app(root=ROOT))
    assert '<div id="root"></div>' in client.get("/").text
    bootstrap = client.get("/api/bootstrap").json()
    factory_units = [unit for unit in bootstrap["units"] if unit["dataset"] == "factory_ego"]
    complete = [unit for unit in factory_units if unit["annotation_state"] == "complete"]
    assert len(factory_units) == 20
    assert {unit["unit_id"] for unit in complete} == {
        unit["unit_id"] for unit in factory_units
    }

    comparison_units = set()
    for complete_unit in complete:
        unit_id = complete_unit["unit_id"]
        unit = client.get(f"/api/units/factory_ego/{unit_id}").json()
        assert set(unit["annotation"]) == {
            "unit_id", "annotation_revision", "interval_convention", "event_labels", "events"
        }
        assert unit["sop"]["events"]
        assert unit["media_url"].endswith("/media")

        runs = client.get(f"/api/comparisons/factory_ego/{unit_id}").json()["runs"]
        if not runs:
            continue
        comparison_units.add(unit_id)
        model_keys = [run["model"].get("id") or run["model"]["name"] for run in runs]
        assert len(model_keys) == len(set(model_keys))
        assert all(run["model"]["name"] in {
            "Marlin-2B temporal grounding",
            "MiniCPM-V 4.6 video grounding",
            "Qwen3.5-4B video grounding",
            "Qwen3-VL-4B video grounding",
        } for run in runs)
        assert all(run["comparison"]["summary"]["mean_tiou"] is not None for run in runs)
        assert all(
            isinstance(event["event_id"], str)
            for run in runs for event in run["comparison"]["events"]
        )
        current_sop_sha = hashlib.sha256(
            (
                ROOT / "datasets" / "factory_ego" / "sops" / unit_id / "sop.yaml"
            ).read_bytes()
        ).hexdigest()
        for run in runs:
            lock = json.loads(
                (ROOT / "runs" / run["run_id"] / "inputs.lock.json").read_text(
                    encoding="utf-8"
                )
            )
            assert lock["units"][unit_id]["sop_sha256"] == current_sop_sha

    run_targets = set()
    for run_path in (ROOT / "runs").glob("*/run.yaml"):
        run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
        if run.get("inference", {}).get("event_subset") is True:
            continue
        lock = json.loads((run_path.parent / "inputs.lock.json").read_text(encoding="utf-8"))
        for unit_id in run["target_units"]:
            current_sop_sha = hashlib.sha256(
                (ROOT / "datasets/factory_ego/sops" / unit_id / "sop.yaml").read_bytes()
            ).hexdigest()
            if lock["units"][unit_id]["sop_sha256"] == current_sop_sha:
                run_targets.add(unit_id)
    assert comparison_units == run_targets


def _temporary_dataset(tmp_path: Path) -> tuple[Path, str]:
    dataset = tmp_path / "datasets/demo"
    unit_id = "clip_001"
    unit = dataset / f"units/{unit_id}"
    frames = unit / "frames"
    frames.mkdir(parents=True)
    source = ROOT / "datasets/konro_inspection/units/konro_inspection/frames/f000.jpg"
    (frames / "f0000.jpg").write_bytes(source.read_bytes())
    (unit / "meta.json").write_text(json.dumps({
        "unit_id": unit_id,
        "dataset_id": "demo",
        "sampling": {"fps": 1.0, "n_frames": 1},
        "media": {"availability": "bundled", "path": "frames"},
        "sop_ref": {"path": "../../sops/clip_001/sop.yaml"},
    }), encoding="utf-8")
    sop = dataset / "sops/clip_001/sop.yaml"
    sop.parent.mkdir(parents=True)
    sop.write_text(
        "sop: {id: clip_001, name: Clip 001}\nevents: []\n"
        "benchmark: {status: annotation_pending}\n",
        encoding="utf-8",
    )
    (dataset / "dataset.yaml").write_text(yaml.safe_dump({
        "dataset_id": "demo",
        "benchmark_state": {"human_ground_truth": {"status": "none"}},
    }), encoding="utf-8")
    return dataset, unit_id


def test_api_saves_annotation_and_enforces_read_only(tmp_path):
    dataset, unit_id = _temporary_dataset(tmp_path)
    payload = {
        "sop": {
            "sop": {"id": unit_id, "name": "Clip 001"},
            "events": [{
                "id": "event_001",
                "ask": "作業者が部品を箱へ入れる",
                "values": ["yes", "no"],
            }],
            "benchmark": {"status": "annotation_pending"},
        },
        "annotation": {
            "unit_id": unit_id,
            "annotation_revision": "human",
            "interval_convention": "half-open_seconds",
            "event_labels": {"event_001": "作業者が部品を箱へ入れる"},
            "events": {"event_001": [{"start_s": 0.0, "end_s": 1.0}]},
        },
    }
    client = TestClient(create_app(root=tmp_path, frontend_dir=tmp_path / "missing"))
    response = client.put(f"/api/units/demo/{unit_id}", json=payload)
    assert response.status_code == 200
    stored = json.loads(
        (dataset / f"annotations/human/{unit_id}.json").read_text(encoding="utf-8")
    )
    assert stored["event_labels"]["event_001"] == "作業者が部品を箱へ入れる"

    read_only = TestClient(
        create_app(root=tmp_path, read_only=True, frontend_dir=tmp_path / "missing")
    )
    assert read_only.put(f"/api/units/demo/{unit_id}", json=payload).status_code == 403


def test_api_returns_422_for_malformed_span_and_404_without_media(tmp_path):
    dataset, unit_id = _temporary_dataset(tmp_path)
    payload = {
        "sop": {
            "sop": {"id": unit_id, "name": "Clip 001"},
            "events": [{"id": "event_001", "ask": "部品を置く"}],
        },
        "annotation": {
            "unit_id": unit_id,
            "annotation_revision": "human",
            "interval_convention": "half-open_seconds",
            "event_labels": {"event_001": "部品を置く"},
            "events": {"event_001": [{"start_s": 0.0}]},
        },
    }
    client = TestClient(create_app(root=tmp_path, frontend_dir=tmp_path / "missing"))
    assert client.put(f"/api/units/demo/{unit_id}", json=payload).status_code == 422

    for frame in (dataset / f"units/{unit_id}/frames").glob("*.jpg"):
        frame.unlink()
    assert client.get(f"/api/units/demo/{unit_id}/media").status_code == 404
