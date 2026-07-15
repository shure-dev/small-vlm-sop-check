#!/usr/bin/env python3
"""Validate Factory Ego dataset/run invariants without external services."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class Validation:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.checks = 0

    def require(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.errors.append(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_dataset(
    repo: Path, result: Validation, *, require_media: bool = False
) -> tuple[dict[str, int], dict[str, Any]]:
    """検証に通ったunit集合を {unit_id: n_frames} で返す。"""
    root = repo / "datasets" / "factory_ego"
    dataset_path = root / "dataset.yaml"
    result.require(dataset_path.is_file(), f"missing {dataset_path}")
    if not dataset_path.is_file():
        return {}, {}
    dataset = read_yaml(dataset_path)
    result.require(dataset.get("dataset_id") == "factory_ego", "dataset_id must be factory_ego")
    benchmark_state = dataset.get("benchmark_state", {})
    human_state = benchmark_state.get("human_ground_truth", {})
    result.require(
        human_state.get("status") in {"none", "partial", "complete"},
        "Factory Ego human ground-truth status is invalid",
    )
    result.require(benchmark_state.get("formal_accuracy_available") is False,
                   "formal accuracy requires an independent test split")

    split_path = root / "splits" / "development.json"
    lock_path = root / "manifest.lock.json"
    result.require(split_path.is_file(), f"missing {split_path}")
    result.require(lock_path.is_file(), f"missing {lock_path}")
    if not split_path.is_file() or not lock_path.is_file():
        return {}, {}
    split = read_json(split_path)
    lock = read_json(lock_path)
    assignments = split.get("assignments", {})
    unit_ids = {
        unit_id
        for values in assignments.values()
        for unit_id in values
    }
    assignment_count = sum(len(values) for values in assignments.values())
    result.require(assignment_count == len(unit_ids), "a unit appears in more than one split")
    result.require(not assignments.get("validation"), "current single-group dataset cannot have validation units")
    result.require(not assignments.get("test"), "current seen units cannot have test assignments")
    result.require(set(assignments.get("dev_seen", [])) == unit_ids, "all current units must be dev_seen")

    group_splits: dict[tuple[str, str], set[str]] = {}
    unit_frames: dict[str, int] = {}
    for unit_id in sorted(unit_ids):
        unit_dir = root / "units" / unit_id
        meta_path = unit_dir / "meta.json"
        result.require(meta_path.is_file(), f"missing meta for {unit_id}")
        if not meta_path.is_file():
            continue
        meta = read_json(meta_path)
        unit_frames[unit_id] = meta.get("sampling", {}).get("n_frames", 0)
        result.require(meta.get("unit_id") == unit_id, f"unit_id/path mismatch: {unit_id}")
        result.require(meta.get("benchmark_status") == "dev_seen", f"unit is not dev_seen: {unit_id}")
        result.require(not (unit_dir / "ground_truth.json").exists(),
                       f"ground_truth must live under annotations/: {unit_id}")
        source = meta.get("source", {})
        group = (source.get("factory_id"), source.get("worker_id"))
        group_splits.setdefault(group, set()).add(meta.get("benchmark_status"))

        frame_manifest_path = unit_dir / meta.get("media", {}).get("sha256_manifest", "")
        result.require(frame_manifest_path.is_file(), f"missing frame manifest: {unit_id}")
        if frame_manifest_path.is_file():
            frame_manifest = read_json(frame_manifest_path)
            result.require(len(frame_manifest) == meta.get("sampling", {}).get("n_frames"),
                           f"frame count mismatch: {unit_id}")
            expected_names = [f"f{idx:04d}.jpg" for idx in range(len(frame_manifest))]
            result.require(sorted(frame_manifest) == expected_names, f"non-canonical frame names: {unit_id}")
            for name, expected_hash in frame_manifest.items():
                data_frame = repo / "data" / "factory_ego" / "units" / unit_id / "frames" / name
                frame_path = data_frame
                if require_media:
                    result.require(frame_path.is_file(), f"missing frame: {unit_id}/{name}")
                if frame_path.is_file():
                    result.require(sha256(frame_path) == expected_hash, f"frame hash mismatch: {unit_id}/{name}")

        sop_ref = meta.get("sop_ref", {})
        sop_path = (unit_dir / sop_ref.get("path", "")).resolve()
        result.require(sop_path.is_file(), f"missing SOP: {unit_id}")
        if sop_path.is_file():
            sop = read_yaml(sop_path)
            result.require(sop.get("sop", {}).get("id") == unit_id, f"SOP id mismatch: {unit_id}")
            result.require(isinstance(sop.get("events"), list), f"SOP events are invalid: {unit_id}")

        unit_lock = lock.get("units", {}).get(unit_id)
        result.require(unit_lock is not None, f"unit missing from manifest.lock: {unit_id}")
        if unit_lock:
            result.require(sha256(meta_path) == unit_lock.get("meta_sha256"), f"meta lock mismatch: {unit_id}")
            result.require(sha256(sop_path) == unit_lock.get("sop_sha256"), f"SOP lock mismatch: {unit_id}")
            result.require(sha256(frame_manifest_path) == unit_lock.get("frames_manifest_sha256"),
                           f"frame manifest lock mismatch: {unit_id}")

    for group, statuses in group_splits.items():
        result.require(len(statuses) == 1, f"group leaked across splits: {group} -> {statuses}")
    result.require(set(lock.get("units", {})) == unit_ids, "manifest.lock unit set differs from split")

    annotation_revision = human_state.get("revision")
    annotation_root = root / "annotations" / str(annotation_revision)
    annotation_paths = sorted(annotation_root.glob("*.json")) if annotation_root.is_dir() else []
    result.require({path.stem for path in annotation_paths}.issubset(unit_ids),
                   "human annotations contain an unknown unit")
    complete_count = 0
    for gt_path in annotation_paths:
        gt = read_json(gt_path)
        unit_id = gt_path.stem
        result.require(gt.get("unit_id") == unit_id, f"GT unit mismatch: {gt_path}")
        result.require(gt.get("interval_convention") == "half-open_seconds",
                       f"GT must use half-open seconds: {gt_path}")
        result.require(
            set(gt) == {"unit_id", "annotation_revision", "interval_convention", "event_labels", "events"},
            f"GT contains unnecessary fields: {gt_path}",
        )
        meta = read_json(root / "units" / unit_id / "meta.json")
        sop_path = (root / "units" / unit_id / meta["sop_ref"]["path"]).resolve()
        sop = read_yaml(sop_path)
        known_labels = {
            str(event["id"]): str(event.get("ask", "")) for event in sop.get("events", [])
        }
        known_events = set(known_labels)
        events = gt.get("events", {})
        result.require(gt.get("event_labels") == known_labels,
                       f"GT event_labels differ from SOP: {gt_path}")
        result.require(set(events).issubset(known_events), f"unknown GT event: {gt_path}")
        if set(events) == known_events:
            complete_count += 1
        for event_id, spans in events.items():
            if spans is None:
                continue
            result.require(isinstance(spans, list) and bool(spans),
                           f"GT spans must be a non-empty list: {gt_path}/{event_id}")
            if not isinstance(spans, list):
                continue
            for span in spans:
                start, end = span.get("start_s"), span.get("end_s")
                duration = unit_frames.get(unit_id, 0) / float(meta["sampling"]["fps"])
                valid = (isinstance(start, (int, float)) and isinstance(end, (int, float))
                         and 0 <= start < end <= duration)
                result.require(valid, f"invalid GT span: {gt_path}/{event_id}")

    expected_human_status = (
        "complete" if unit_ids and complete_count == len(unit_ids)
        else "partial" if annotation_paths
        else "none"
    )
    result.require(human_state.get("status") == expected_human_status,
                   f"human annotation status must be {expected_human_status}")

    return unit_frames, lock


def validate_prediction(path: Path, run_id: str, unit_id: str, duration: float,
                        result: Validation) -> None:
    prediction = read_json(path)
    result.require(prediction.get("run_id") == run_id, f"prediction run mismatch: {path}")
    result.require(prediction.get("unit_id") == unit_id, f"prediction unit mismatch: {path}")
    result.require(prediction.get("method") in {"temporal_grounding", "frame_classification"},
                   f"bad prediction method: {path}")
    result.require(prediction.get("interval_convention") == "half-open_seconds",
                   f"prediction must use half-open seconds: {path}")
    events = prediction.get("events")
    result.require(isinstance(events, dict) and bool(events), f"missing prediction events: {path}")
    if isinstance(events, dict):
        for event_id, spans in events.items():
            if spans is None:
                continue
            result.require(isinstance(spans, list), f"prediction spans must be list: {path}/{event_id}")
            if not isinstance(spans, list):
                continue
            for span in spans:
                start, end = span.get("start_s"), span.get("end_s")
                valid = (isinstance(start, (int, float)) and isinstance(end, (int, float))
                         and 0 <= start < end <= duration)
                result.require(valid, f"invalid prediction span: {path}/{event_id}")


def validate_runs(repo: Path, unit_frames: dict[str, int], result: Validation) -> None:
    unit_ids = set(unit_frames)
    runs_root = repo / "runs"
    index_path = runs_root / "index.jsonl"
    result.require(index_path.is_file(), "missing runs/index.jsonl")
    run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir()) if runs_root.is_dir() else []
    discovered: set[str] = set()
    for run_dir in run_dirs:
        run_path = run_dir / "run.yaml"
        result.require(run_path.is_file(), f"missing run.yaml: {run_dir.name}")
        if not run_path.is_file():
            continue
        run = read_yaml(run_path)
        run_id = run.get("run_id")
        discovered.add(run_id)
        result.require(run_id == run_dir.name, f"run id/path mismatch: {run_dir}")
        result.require(run.get("kind") == "prediction", f"non-prediction run in runs/: {run_id}")
        result.require(run.get("status") == "complete", f"incomplete migrated run: {run_id}")
        result.require(run.get("immutable") is True, f"run must be immutable: {run_id}")
        result.require(run.get("ground_truth_used") is False, f"prediction run used GT: {run_id}")
        result.require(run.get("metrics") is None, f"metrics belong in an evaluation run: {run_id}")
        targets = run.get("target_units", [])
        result.require(set(targets).issubset(unit_ids), f"unknown target unit in {run_id}")
        result.require((run_dir / "inputs.lock.json").is_file(), f"missing inputs lock: {run_id}")
        for unit_id in targets:
            prediction_path = run_dir / "predictions" / f"{unit_id}.json"
            raw_path = run_dir / "raw" / f"{unit_id}.json"
            result.require(prediction_path.is_file(), f"missing prediction: {run_id}/{unit_id}")
            result.require(raw_path.is_file(), f"missing raw output: {run_id}/{unit_id}")
            if prediction_path.is_file():
                meta = read_json(repo / "datasets" / "factory_ego" / "units" / unit_id / "meta.json")
                duration = unit_frames.get(unit_id, 0) / float(meta["sampling"]["fps"])
                validate_prediction(prediction_path, run_id, unit_id, duration, result)

    if index_path.is_file():
        rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line]
        result.require({row.get("run_id") for row in rows} == discovered, "runs/index.jsonl is stale")
        result.require(all(row.get("formal_accuracy") is None for row in rows),
                       "prediction index must not claim formal accuracy")


def validate_schemas(repo: Path, result: Validation) -> None:
    schema_root = repo / "schemas" / "benchmark"
    expected = {"annotation.schema.json", "unit.schema.json", "prediction.schema.json",
                "run.schema.json", "split.schema.json"}
    result.require(schema_root.is_dir(), "missing benchmark schemas directory")
    if schema_root.is_dir():
        present = {path.name for path in schema_root.glob("*.json")}
        result.require(expected.issubset(present), f"missing schemas: {sorted(expected - present)}")
        for path in schema_root.glob("*.json"):
            try:
                read_json(path)
            except (OSError, json.JSONDecodeError) as exc:
                result.errors.append(f"invalid JSON schema {path}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--require-media",
        action="store_true",
        help="require local gated frames and verify every SHA; public clones omit these frames",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    result = Validation()
    validate_schemas(repo, result)
    unit_frames, _ = validate_dataset(repo, result, require_media=args.require_media)
    validate_runs(repo, unit_frames, result)
    if result.errors:
        print(f"FAIL: {len(result.errors)} error(s) across {result.checks} checks")
        for error in result.errors:
            print(f"  - {error}")
        return 1
    print(f"PASS: {result.checks} benchmark integrity checks")
    print(f"  units={len(unit_frames)} runs={len([p for p in (repo / 'runs').iterdir() if p.is_dir()])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
