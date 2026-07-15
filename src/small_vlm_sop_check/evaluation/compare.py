"""同一unit集合に対する2つのprediction runをTemporal IoUで比較する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from ..core.temporal import evaluate_temporal, load_annotation, load_prediction


def _load_run(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load((path / "run.yaml").read_text(encoding="utf-8"))
    if doc.get("kind") != "prediction" or doc.get("status") != "complete":
        raise ValueError(f"completeなprediction runが必要です: {path}")
    return doc


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    gt = sum(row["summary"]["gt_occurrences"] for row in results)
    pred = sum(row["summary"]["predicted_occurrences"] for row in results)
    matched_sum = sum(
        event["tiou"] for row in results for event in row["events"]
        if event["status"] == "match" and event["tiou"] is not None
    )
    thresholds: dict[str, Any] = {}
    for key in ("tiou@0.1", "tiou@0.3", "tiou@0.5"):
        tp = sum(row["summary"]["thresholds"][key]["tp"] for row in results)
        fp = sum(row["summary"]["thresholds"][key]["fp"] for row in results)
        fn = sum(row["summary"]["thresholds"][key]["fn"] for row in results)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
        thresholds[key] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 6) if precision is not None else None,
            "recall": round(recall, 6) if recall is not None else None,
            "f1": round(f1, 6) if f1 is not None else None,
        }
    return {
        "unit_count": len(results), "gt_occurrences": gt, "predicted_occurrences": pred,
        "mean_tiou": round(matched_sum / gt, 6) if gt else None,
        "thresholds": thresholds,
    }


def compare_runs(repo: Path, baseline_dir: Path, tuned_dir: Path,
                 *, allow_partial: bool = False) -> dict[str, Any]:
    repo = repo.resolve()
    baseline_dir, tuned_dir = baseline_dir.resolve(), tuned_dir.resolve()
    baseline_run, tuned_run = _load_run(baseline_dir), _load_run(tuned_dir)
    if baseline_run.get("ground_truth_used") is not False or tuned_run.get("ground_truth_used") is not False:
        raise ValueError("比較するprediction runにはground_truth_used: falseが必要です")
    baseline_units = list(baseline_run.get("target_units", []))
    tuned_units = list(tuned_run.get("target_units", []))
    if not baseline_units or set(baseline_units) != set(tuned_units):
        raise ValueError("baselineとtunedは同一の空でないtarget_unitsが必要です")
    baseline_dataset = baseline_run.get("dataset", {})
    tuned_dataset = tuned_run.get("dataset", {})
    if (baseline_dataset.get("id"), baseline_dataset.get("split")) != (
            tuned_dataset.get("id"), tuned_dataset.get("split")):
        raise ValueError("baselineとtunedは同一dataset/splitが必要です")
    dataset_id = baseline_dataset["id"]
    dataset_doc = yaml.safe_load(
        (repo / "datasets" / dataset_id / "dataset.yaml").read_text(encoding="utf-8")
    )
    status = dataset_doc.get("benchmark_state", {}).get("human_ground_truth", {}).get("status")
    if status != "complete" and not allow_partial:
        raise ValueError(f"annotation status={status!r}。正式比較にはcompleteが必要です")

    per_unit = []
    baseline_results, tuned_results = [], []
    for unit_id in sorted(baseline_units):
        annotation_path = repo / "datasets" / dataset_id / "annotations" / "human" / f"{unit_id}.json"
        baseline_path = baseline_dir / "predictions" / f"{unit_id}.json"
        tuned_path = tuned_dir / "predictions" / f"{unit_id}.json"
        missing = [str(path) for path in (annotation_path, baseline_path, tuned_path) if not path.is_file()]
        if missing:
            raise ValueError(f"比較入力が不足しています: {missing}")
        annotation_doc = json.loads(annotation_path.read_text(encoding="utf-8"))
        baseline_doc = json.loads(baseline_path.read_text(encoding="utf-8"))
        tuned_doc = json.loads(tuned_path.read_text(encoding="utf-8"))
        if any(doc.get("unit_id") != unit_id for doc in (annotation_doc, baseline_doc, tuned_doc)):
            raise ValueError(f"unit_idがファイル名と一致しません: {unit_id}")
        annotation = load_annotation(annotation_doc)
        baseline_result = evaluate_temporal(annotation, load_prediction(baseline_doc))
        tuned_result = evaluate_temporal(annotation, load_prediction(tuned_doc))
        baseline_results.append(baseline_result)
        tuned_results.append(tuned_result)
        before = baseline_result["summary"]["mean_tiou"]
        after = tuned_result["summary"]["mean_tiou"]
        per_unit.append({"unit_id": unit_id, "baseline_mean_tiou": before,
                         "tuned_mean_tiou": after,
                         "delta": round(after - before, 6) if before is not None and after is not None else None})

    baseline_summary = _aggregate(baseline_results)
    tuned_summary = _aggregate(tuned_results)
    before = baseline_summary["mean_tiou"]
    after = tuned_summary["mean_tiou"]
    return {
        "comparison": "before_after_temporal_iou",
        "dataset": {"id": dataset_id, "split": baseline_dataset["split"],
                    "annotation_state": status},
        "unit_ids": sorted(baseline_units),
        "baseline": {"run_id": baseline_run["run_id"], **baseline_summary},
        "tuned": {"run_id": tuned_run["run_id"], **tuned_summary},
        "delta": {"mean_tiou": round(after - before, 6) if before is not None and after is not None else None,
                  "tiou@0.5_f1": _delta_f1(baseline_summary, tuned_summary, "tiou@0.5")},
        "per_unit": per_unit,
    }


def _delta_f1(before: dict[str, Any], after: dict[str, Any], key: str) -> float | None:
    a, b = before["thresholds"][key]["f1"], after["thresholds"][key]["f1"]
    return round(b - a, 6) if a is not None and b is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(prog="sop-compare")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--tuned-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    try:
        result = compare_runs(args.repo, args.baseline_run, args.tuned_run,
                              allow_partial=args.allow_partial)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"[sop-compare] {exc}") from exc
    print(f"mean tIoU: {result['baseline']['mean_tiou']} -> {result['tuned']['mean_tiou']} "
          f"(delta {result['delta']['mean_tiou']:+.6f})")
    print(f"[sop-compare] {args.out}")


if __name__ == "__main__":
    main()
