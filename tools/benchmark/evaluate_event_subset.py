#!/usr/bin/env python3
"""prediction runの全件または選択unitを人手GTに対して評価する。"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from small_vlm_sop_check.core.temporal import (  # noqa: E402
    evaluate_temporal,
    load_annotation,
    load_prediction,
)
from small_vlm_sop_check.evaluation.compare import _aggregate  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_run(
    repo: Path, run_dir: Path, unit_ids: list[str] | None = None
) -> dict[str, Any]:
    run = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
    if run.get("status") != "complete" or run.get("kind") != "prediction":
        raise ValueError("completeなprediction runが必要です")
    if run.get("ground_truth_used") is not False:
        raise ValueError("ground_truth_used=falseが必要です")
    is_subset = run.get("inference", {}).get("event_subset") is True

    queries_path = run_dir / "queries.json"
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    if unit_ids:
        missing = sorted(set(unit_ids) - set(queries))
        if missing:
            raise ValueError(f"run queriesにunitがありません: {', '.join(missing)}")
        requested = set(unit_ids)
        queries = {
            unit_id: unit_queries
            for unit_id, unit_queries in queries.items()
            if unit_id in requested
        }
    annotation_hashes: dict[str, str] = {}
    prediction_hashes: dict[str, str] = {}
    results: list[dict[str, Any]] = []
    results_by_unit: dict[str, list[dict[str, Any]]] = {}
    per_event: list[dict[str, Any]] = []

    dataset_id = run["dataset"]["id"]
    for unit_id, unit_queries in queries.items():
        annotation_path = (
            repo / "datasets" / dataset_id / "annotations" / "human" / f"{unit_id}.json"
        )
        prediction_path = run_dir / "predictions" / f"{unit_id}.json"
        annotation_doc = json.loads(annotation_path.read_text(encoding="utf-8"))
        prediction_doc = json.loads(prediction_path.read_text(encoding="utf-8"))
        if set(prediction_doc["events"]) != set(unit_queries):
            raise ValueError(f"predictionとqueryのevent IDが一致しません: {unit_id}")

        annotation = load_annotation(annotation_doc)
        prediction = load_prediction(prediction_doc)
        for event_id, query in unit_queries.items():
            if event_id not in annotation:
                raise ValueError(f"現行annotationにeventがありません: {unit_id}/{event_id}")
            result = evaluate_temporal(
                {event_id: annotation[event_id]}, {event_id: prediction[event_id]}
            )
            results.append(result)
            results_by_unit.setdefault(unit_id, []).append(result)
            summary = result["summary"]
            per_event.append({
                "unit_id": unit_id,
                "event_id": event_id,
                "event_label": annotation_doc["event_labels"][event_id],
                "query": query,
                "reference_occurrences": summary["gt_occurrences"],
                "predicted_occurrences": summary["predicted_occurrences"],
                "mean_tiou": summary["mean_tiou"],
                "tiou_at_0_5": summary["thresholds"]["tiou@0.5"],
            })
        annotation_hashes[unit_id] = sha256_file(annotation_path)
        prediction_hashes[unit_id] = sha256_file(prediction_path)

    overall = _aggregate(results)
    return {
        "kind": (
            "development_event_subset_evaluation" if is_subset
            else "development_evaluation"
        ),
        "created_at": datetime.date.today().isoformat(),
        "dataset": {
            "id": dataset_id,
            "split": run["dataset"]["split"],
            "reference_revision": "human",
            "unit_count": len(queries),
            "event_count": sum(len(events) for events in queries.values()),
            "unit_ids": list(queries),
        },
        "prediction_run_id": run["run_id"],
        "model": run["model"]["name"],
        "formal_accuracy": False,
        "scope": (
            "events whose annotation labels changed since the baseline run"
            if is_subset
            else (
                "all events in the selected run target units"
                if unit_ids
                else "all events in the run target units"
            )
        ),
        "inputs": {
            "queries_sha256": sha256_file(queries_path),
            "run_manifest_sha256": sha256_file(run_dir / "run.yaml"),
            "run_inputs_lock_sha256": sha256_file(run_dir / "inputs.lock.json"),
            "annotation_sha256": annotation_hashes,
            "prediction_sha256": prediction_hashes,
        },
        "metrics": {
            "reference_occurrences": overall["gt_occurrences"],
            "predicted_occurrences": overall["predicted_occurrences"],
            "mean_tiou": overall["mean_tiou"],
            "tiou_at_0_5": overall["thresholds"]["tiou@0.5"],
        },
        "per_unit_mean_tiou": {
            unit_id: _aggregate(unit_results)["mean_tiou"]
            for unit_id, unit_results in results_by_unit.items()
        },
        "per_event": per_event,
        "limitations": [
            "Development evaluation, not a held-out score.",
            "Only the selected units and their events from run queries.json are evaluated.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--unit",
        action="append",
        dest="units",
        help="評価対象unit ID。複数回指定可。省略時はrun内の全unit。",
    )
    args = parser.parse_args()
    artifact = evaluate_run(args.repo.resolve(), args.run.resolve(), args.units)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"mean tIoU: {artifact['metrics']['mean_tiou']}")
    print(f"[evaluate-event-subset] {args.out}")


if __name__ == "__main__":
    main()
