"""unit台帳が共通meta契約を読めることの回帰テスト。"""
import json
from pathlib import Path

from small_vlm_sop_check.apps import catalog


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _make_tree(root: Path) -> None:
    # --- konro形: sop_refs はリスト、ground_truth_ref あり、fps=1.0 ---
    ku = root / "datasets" / "konro_inspection" / "units" / "konro_inspection"
    _write(ku / "meta.json", {
        "dataset_id": "konro_inspection", "unit_id": "konro_inspection",
        "sampling": {"fps": 1.0},
        "media": {"availability": "bundled", "path": "frames"},
        "sop_ref": {"path": "../../sops/konro_inspection/konro_inspection.yaml"},
    })
    (ku / "frames").mkdir(parents=True)
    for i in range(3):
        (ku / "frames" / f"f{i:03d}.jpg").write_bytes(b"x")
    # konroには既存GTがある
    _write(root / "datasets" / "konro_inspection" / "annotations" / "human"
           / "konro_inspection.json", {"events": {}})

    # --- factory形: sop_ref は dict、GT参照なし、fps=2.0、4桁フレーム ---
    fu = root / "datasets" / "factory_ego" / "units" / "f001_w004"
    _write(fu / "meta.json", {
        "dataset_id": "factory_ego", "unit_id": "f001_w004",
        "source": {"start_second": 82.0},
        "sampling": {"fps": 2.0, "n_frames": 2},
        "media": {"availability": "local", "path": "frames"},
        "sop_ref": {"id": "f001_w004", "path": "../../sops/f001_w004/sop.yaml"},
    })
    frames = root / "data" / "factory_ego" / "units" / "f001_w004" / "frames"
    frames.mkdir(parents=True)
    for i in range(2):
        (frames / f"f{i:04d}.jpg").write_bytes(b"x")


def test_discover_reads_common_meta_contract(tmp_path):
    _make_tree(tmp_path)
    cat = catalog.discover(tmp_path)

    ids = [u.unit_id for u in cat.units]
    assert ids == ["f001_w004", "konro_inspection"]     # dataset名→unit_id 順
    assert cat.datasets() == ["factory_ego", "konro_inspection"]

    konro = cat.get("konro_inspection")
    assert konro.fps == 1.0
    assert konro.frame_files() == ["f000.jpg", "f001.jpg", "f002.jpg"]
    assert konro.has_frames() and konro.has_gt()        # konroはGTあり
    assert konro.source_start_seconds == 0.0
    assert konro.sop_path.name == "konro_inspection.yaml"
    assert konro.gt_path.parts[-3:] == ("annotations", "human", "konro_inspection.json")

    fac = cat.get("f001_w004")
    assert fac.fps == 2.0
    assert fac.source_start_seconds == 82.0
    assert fac.frame_files() == ["f0000.jpg", "f0001.jpg"]
    assert not fac.has_gt()                              # factoryはGT未作成
    assert fac.gt_path.parts[-3:] == ("annotations", "human", "f001_w004.json")


def test_summary_shape(tmp_path):
    _make_tree(tmp_path)
    cat = catalog.discover(tmp_path)
    s = {row["unit_id"]: row for row in cat.summaries()}
    assert s["konro_inspection"] == {
        "dataset": "konro_inspection", "unit_id": "konro_inspection",
        "fps": 1.0, "duration_s": 3.0, "has_frames": True, "has_gt": True,
        "source_start_seconds": 0.0,
    }
    assert s["f001_w004"]["has_gt"] is False


def test_broken_meta_is_skipped(tmp_path):
    _make_tree(tmp_path)
    broken = tmp_path / "datasets" / "x" / "units" / "u" / "meta.json"
    broken.parent.mkdir(parents=True)
    broken.write_text("{ not json", encoding="utf-8")
    cat = catalog.discover(tmp_path)
    assert [u.unit_id for u in cat.units] == ["f001_w004", "konro_inspection"]


def test_adhoc_unit_prepended_and_deduped(tmp_path):
    _make_tree(tmp_path)
    extra = catalog.adhoc_unit(
        "konro_inspection", tmp_path / "s.yaml", tmp_path / "f",
        1.0, tmp_path / "gt.json")
    cat = catalog.discover(tmp_path, extra=extra)
    # 先頭にextra、同一unit_idの台帳側は除外
    assert cat.units[0].dataset == "custom"
    assert [u.unit_id for u in cat.units] == ["konro_inspection", "f001_w004"]
    assert cat.get("konro_inspection").dataset == "custom"
