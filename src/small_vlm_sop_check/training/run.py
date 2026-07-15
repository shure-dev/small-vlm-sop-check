"""ms-swift動画LoRAを再現可能な学習runとして準備・実行する。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


MS_SWIFT_VERSION = "4.4.1"
MS_SWIFT_REVISION = "9938a463946beb66d6b502d6f3a7dc64845c4df1"
MAX_MODEL_PARAMETERS_B = 4.0
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _load_export(directory: Path) -> tuple[dict[str, Any], Path]:
    directory = directory.resolve()
    manifest_path = directory / "export.json"
    if not manifest_path.is_file():
        raise ValueError(f"export.jsonがありません: {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "ms-swift-messages-video-sft":
        raise ValueError(f"未対応のexport形式です: {manifest.get('format')!r}")
    jsonl = directory / manifest["jsonl"]["path"]
    if not jsonl.is_file() or _sha256(jsonl) != manifest["jsonl"]["sha256"]:
        raise ValueError(f"学習JSONLが無いかhashが一致しません: {jsonl}")
    if manifest.get("annotation_state") != "complete":
        raise ValueError("正式な学習runにはcomplete annotationだけを使用できます")
    media = manifest.get("media")
    if not isinstance(media, dict) or set(media) != set(manifest.get("unit_ids", [])):
        raise ValueError("export.mediaは全unitの動画hashを含む必要があります")
    for unit_id, item in media.items():
        path = directory / item["path"]
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise ValueError(f"{unit_id}: 学習動画が無いかhashが一致しません: {path}")
    return manifest, jsonl


def build_command(*, model: str, model_revision: str, train_jsonl: Path, output_dir: Path,
                  validation_jsonl: Path | None, epochs: float, learning_rate: float,
                  lora_rank: int, lora_alpha: int, max_length: int,
                  gradient_accumulation_steps: int, seed: int) -> list[str]:
    command = [
        "swift", "sft", "--model", model,
        "--model_revision", model_revision,
        "--dataset", str(train_jsonl),
        "--tuner_type", "lora",
        "--torch_dtype", "bfloat16",
        "--num_train_epochs", str(epochs),
        "--per_device_train_batch_size", "1",
        "--learning_rate", str(learning_rate),
        "--lora_rank", str(lora_rank),
        "--lora_alpha", str(lora_alpha),
        "--target_modules", "all-linear",
        "--freeze_vit", "true",
        "--freeze_aligner", "true",
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--max_length", str(max_length),
        "--output_dir", str(output_dir),
        "--logging_steps", "5",
        "--save_steps", "50",
        "--save_total_limit", "2",
        "--seed", str(seed), "--data_seed", str(seed),
    ]
    if validation_jsonl is not None:
        command.extend(["--val_dataset", str(validation_jsonl),
                        "--per_device_eval_batch_size", "1", "--eval_steps", "50"])
    else:
        # unitを跨ぐランダム分割を暗黙に行わない。
        command.extend(["--split_dataset_ratio", "0"])
    return command


def prepare(args: argparse.Namespace) -> Path:
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ValueError("run IDは英数字で始まる英数字・dot・underscore・hyphenに限定します")
    if not 0 < args.model_parameters_b <= MAX_MODEL_PARAMETERS_B:
        raise ValueError(
            f"モデル規模は0より大きく{MAX_MODEL_PARAMETERS_B:g}B以下が必要です: "
            f"{args.model_parameters_b}B"
        )
    if args.epochs <= 0 or args.learning_rate <= 0 or args.lora_rank <= 0:
        raise ValueError("epochs, learning-rate, lora-rankは正数が必要です")
    train_manifest, train_jsonl = _load_export(args.train_export)
    validation_manifest = None
    validation_jsonl = None
    if args.validation_export:
        validation_manifest, validation_jsonl = _load_export(args.validation_export)
        if validation_manifest["dataset_id"] != train_manifest["dataset_id"]:
            raise ValueError("trainとvalidationのdataset_idが一致しません")
        overlap = set(train_manifest["unit_ids"]) & set(validation_manifest["unit_ids"])
        if overlap:
            raise ValueError(f"train/validation unitが重複しています: {sorted(overlap)}")

    run_dir = (args.repo.resolve() / "training_runs" / args.run_id)
    if run_dir.exists():
        raise ValueError(f"training runは既に存在します: {run_dir}")
    artifacts = run_dir / "artifacts"
    command = build_command(
        model=args.model, model_revision=args.model_revision, train_jsonl=train_jsonl,
        validation_jsonl=validation_jsonl,
        output_dir=artifacts, epochs=args.epochs, learning_rate=args.learning_rate,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps, seed=args.seed,
    )
    now = datetime.now(timezone.utc).isoformat()
    run = {
        "run_id": args.run_id,
        "kind": "training",
        "status": "prepared",
        "immutable": False,
        "created_at": now,
        "updated_at": now,
        "backend": {
            "name": "ms-swift", "version": MS_SWIFT_VERSION,
            "repository": "https://github.com/modelscope/ms-swift",
            "revision": MS_SWIFT_REVISION,
        },
        "model": {"id": args.model, "revision": args.model_revision,
                  "parameters_b": args.model_parameters_b,
                  "maximum_allowed_b": MAX_MODEL_PARAMETERS_B},
        "dataset": {
            "id": train_manifest["dataset_id"],
            "train_subset": train_manifest["split"],
            "validation_subset": validation_manifest["split"] if validation_manifest else None,
        },
        "method": {"name": "video_sft_lora", "tuner_type": "lora", "seed": args.seed,
                   "freeze_vit": True, "freeze_aligner": True},
        "artifacts": {"path": "artifacts", "committed": False},
    }
    lock = {
        "train": {"export": str(args.train_export.resolve() / "export.json"),
                  "export_sha256": _sha256(args.train_export.resolve() / "export.json"),
                  "jsonl": str(train_jsonl), "jsonl_sha256": _sha256(train_jsonl),
                  "unit_ids": train_manifest["unit_ids"]},
        "validation": ({"export": str(args.validation_export.resolve() / "export.json"),
                        "export_sha256": _sha256(args.validation_export.resolve() / "export.json"),
                        "jsonl": str(validation_jsonl), "jsonl_sha256": _sha256(validation_jsonl),
                       "unit_ids": validation_manifest["unit_ids"]}
                       if validation_manifest else None),
        "media": [
            {"unit_id": unit_id,
             "path": str((args.train_export.resolve() / item["path"])),
             "sha256": item["sha256"]}
            for unit_id, item in sorted(train_manifest["media"].items())
        ] + ([
            {"unit_id": unit_id,
             "path": str((args.validation_export.resolve() / item["path"])),
             "sha256": item["sha256"]}
            for unit_id, item in sorted(validation_manifest["media"].items())
        ] if validation_manifest else []),
    }
    run_dir.mkdir(parents=True)
    _atomic(run_dir / "run.yaml", yaml.safe_dump(run, allow_unicode=True, sort_keys=False))
    _atomic(run_dir / "command.json", json.dumps(command, ensure_ascii=False, indent=2) + "\n")
    _atomic(run_dir / "inputs.lock.json", json.dumps(lock, ensure_ascii=False, indent=2) + "\n")
    return run_dir


def execute(run_dir: Path, *, dry_run: bool = False) -> list[str] | int:
    run_dir = run_dir.resolve()
    run_path = run_dir / "run.yaml"
    command_path = run_dir / "command.json"
    run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
    command = json.loads(command_path.read_text(encoding="utf-8"))
    if run.get("immutable") or run.get("status") != "prepared":
        raise ValueError(f"prepared状態のrunだけを実行できます: {run.get('status')}")
    lock = json.loads((run_dir / "inputs.lock.json").read_text(encoding="utf-8"))
    for label in ("train", "validation"):
        entry = lock.get(label)
        if not entry:
            continue
        for path_key, hash_key in (("export", "export_sha256"), ("jsonl", "jsonl_sha256")):
            path = Path(entry[path_key])
            if not path.is_file() or _sha256(path) != entry[hash_key]:
                raise ValueError(f"{label} inputがprepare後に変更されています: {path}")
    for entry in lock.get("media", []):
        path = Path(entry["path"])
        if not path.is_file() or _sha256(path) != entry["sha256"]:
            raise ValueError(f"media inputがprepare後に変更されています: {path}")
    if dry_run:
        return command

    run["status"] = "running"
    run["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic(run_path, yaml.safe_dump(run, allow_unicode=True, sort_keys=False))
    log_path = run_dir / "training.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=run_dir, stdout=log,
                                    stderr=subprocess.STDOUT, text=True, check=False)
    except FileNotFoundError as exc:
        run["status"] = "failed"
        run["immutable"] = True
        run["failure"] = "ms-swift CLI `swift` が見つかりません"
        run["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic(run_path, yaml.safe_dump(run, allow_unicode=True, sort_keys=False))
        raise ValueError(run["failure"]) from exc
    run["status"] = "complete" if result.returncode == 0 else "failed"
    run["immutable"] = True
    run["exit_code"] = result.returncode
    run["log_sha256"] = _sha256(log_path)
    run["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic(run_path, yaml.safe_dump(run, allow_unicode=True, sort_keys=False))
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(prog="sop-train")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="ms-swift動画LoRAの再現可能なrunを作成")
    prep.add_argument("--repo", type=Path, default=Path.cwd())
    prep.add_argument("--run-id", required=True)
    prep.add_argument("--model", required=True)
    prep.add_argument("--model-revision", default="main")
    prep.add_argument("--model-parameters-b", type=float, required=True)
    prep.add_argument("--train-export", type=Path, required=True)
    prep.add_argument("--validation-export", type=Path)
    prep.add_argument("--epochs", type=float, default=1.0)
    prep.add_argument("--learning-rate", type=float, default=1e-4)
    prep.add_argument("--lora-rank", type=int, default=8)
    prep.add_argument("--lora-alpha", type=int, default=32)
    prep.add_argument("--max-length", type=int, default=2048)
    prep.add_argument("--gradient-accumulation-steps", type=int, default=16)
    prep.add_argument("--seed", type=int, default=42)
    run = sub.add_parser("run", help="prepared runを実行して状態とログを確定")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            out = prepare(args)
            print(f"[sop-train] prepared: {out}")
        else:
            result = execute(args.run_dir, dry_run=args.dry_run)
            if args.dry_run:
                print(shlex.join(result))
            elif result:
                raise SystemExit(result)
            else:
                print(f"[sop-train] complete: {args.run_dir}")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"[sop-train] {exc}") from exc


if __name__ == "__main__":
    main()
