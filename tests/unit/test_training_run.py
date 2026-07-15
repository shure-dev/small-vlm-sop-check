import argparse
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from small_vlm_sop_check.training.run import build_command, execute, prepare


def _export(root: Path, subset: str, units: list[str]) -> Path:
    root.mkdir(parents=True)
    jsonl = root / "train.jsonl"
    jsonl.write_text('{"messages":[],"videos":[]}\n', encoding="utf-8")
    digest = hashlib.sha256(jsonl.read_bytes()).hexdigest()
    media = root / "media"
    media.mkdir()
    media_doc = {}
    for unit in units:
        video = media / f"{unit}.mp4"
        video.write_bytes(f"video:{unit}".encode())
        media_doc[unit] = {"path": f"media/{unit}.mp4",
                           "sha256": hashlib.sha256(video.read_bytes()).hexdigest()}
    (root / "export.json").write_text(json.dumps({
        "format": "ms-swift-messages-video-sft", "dataset_id": "demo",
        "annotation_state": "complete", "split": {"path": "split.json", "subset": subset},
        "unit_ids": units, "jsonl": {"path": "train.jsonl", "sha256": digest},
        "media": media_doc,
    }), encoding="utf-8")
    return root


def test_build_command_uses_current_ms_swift_video_lora_contract(tmp_path: Path):
    command = build_command(model="Qwen/Qwen2.5-VL-3B-Instruct", model_revision="main",
                            train_jsonl=tmp_path / "train.jsonl",
                            validation_jsonl=None, output_dir=tmp_path / "artifacts",
                            epochs=1, learning_rate=1e-4, lora_rank=8, lora_alpha=32,
                            max_length=2048, gradient_accumulation_steps=16, seed=42)
    assert command[:2] == ["swift", "sft"]
    assert command[command.index("--tuner_type") + 1] == "lora"
    assert command[command.index("--split_dataset_ratio") + 1] == "0"
    assert command[command.index("--seed") + 1] == "42"


def test_prepare_locks_inputs_and_enforces_4b(tmp_path: Path):
    train = _export(tmp_path / "train", "train", ["u1"])
    args = argparse.Namespace(repo=tmp_path, run_id="demo-run", model="model-3b",
                              model_revision="main", seed=42,
                              model_parameters_b=3.0, train_export=train,
                              validation_export=None, epochs=1.0, learning_rate=1e-4,
                              lora_rank=8, lora_alpha=32, max_length=2048,
                              gradient_accumulation_steps=16)
    run_dir = prepare(args)
    run = yaml.safe_load((run_dir / "run.yaml").read_text())
    assert run["status"] == "prepared" and run["model"]["parameters_b"] == 3.0
    assert execute(run_dir, dry_run=True)[:2] == ["swift", "sft"]

    args.run_id = "too-large"
    args.model_parameters_b = 7.0
    with pytest.raises(ValueError, match="4B以下"):
        prepare(args)


def test_prepare_rejects_train_validation_overlap(tmp_path: Path):
    train = _export(tmp_path / "train", "train", ["u1"])
    valid = _export(tmp_path / "valid", "validation", ["u1"])
    args = argparse.Namespace(repo=tmp_path, run_id="overlap", model="model-2b",
                              model_revision="main", seed=42,
                              model_parameters_b=2.0, train_export=train,
                              validation_export=valid, epochs=1.0, learning_rate=1e-4,
                              lora_rank=8, lora_alpha=32, max_length=2048,
                              gradient_accumulation_steps=16)
    with pytest.raises(ValueError, match="重複"):
        prepare(args)


def test_execute_rejects_input_changed_after_prepare(tmp_path: Path):
    train = _export(tmp_path / "train", "train", ["u1"])
    args = argparse.Namespace(repo=tmp_path, run_id="locked", model="model-2b",
                              model_revision="main", seed=42,
                              model_parameters_b=2.0, train_export=train,
                              validation_export=None, epochs=1.0, learning_rate=1e-4,
                              lora_rank=8, lora_alpha=32, max_length=2048,
                              gradient_accumulation_steps=16)
    run_dir = prepare(args)
    (train / "train.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="変更"):
        execute(run_dir, dry_run=True)
