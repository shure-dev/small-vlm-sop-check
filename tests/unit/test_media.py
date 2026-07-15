from pathlib import Path

from small_vlm_sop_check.apps.catalog import Unit
from small_vlm_sop_check.apps.media import preview_video


def test_generated_preview_is_written_to_ignored_data_root(tmp_path: Path, monkeypatch):
    dataset = tmp_path / "datasets/demo"
    meta = dataset / "units/clip_001/meta.json"
    frames = meta.parent / "frames"
    frames.mkdir(parents=True)
    meta.write_text("{}", encoding="utf-8")
    (frames / "f0000.jpg").write_bytes(b"frame")
    sop = dataset / "sops/clip_001/sop.yaml"
    sop.parent.mkdir(parents=True)
    sop.write_text("sop: {id: clip_001, name: Clip}\nevents: []\n", encoding="utf-8")
    unit = Unit(
        dataset="demo",
        unit_id="clip_001",
        sop_path=sop,
        frames_dir=frames,
        fps=1.0,
        gt_path=dataset / "annotations/human/clip_001.json",
        n_frames=1,
        meta_path=meta,
    )
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)

    generated = preview_video(unit)

    assert generated == tmp_path / "data/demo/units/clip_001/.annotation-preview.mp4"
    assert not (meta.parent / ".annotation-preview.mp4").exists()
