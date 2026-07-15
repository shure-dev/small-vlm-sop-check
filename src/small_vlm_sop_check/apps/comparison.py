"""結果レビューへhuman annotationとpredictionの秒区間比較を供給する。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

from ..core.temporal import evaluate_temporal, load_annotation, load_prediction
from .catalog import Unit


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    reference_count = sum(row["summary"]["gt_occurrences"] for row in results)
    prediction_count = sum(row["summary"]["predicted_occurrences"] for row in results)
    matched_sum = sum(
        event["tiou"] for row in results for event in row["events"]
        if event["status"] == "match" and event["tiou"] is not None
    )
    thresholds = {}
    for key in ("tiou@0.1", "tiou@0.3", "tiou@0.5"):
        tp = sum(row["summary"]["thresholds"][key]["tp"] for row in results)
        fp = sum(row["summary"]["thresholds"][key]["fp"] for row in results)
        fn = sum(row["summary"]["thresholds"][key]["fn"] for row in results)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall else None)
        thresholds[key] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 6) if precision is not None else None,
            "recall": round(recall, 6) if recall is not None else None,
            "f1": round(f1, 6) if f1 is not None else None,
        }
    return {
        "unit_count": len(results),
        "reference_occurrences": reference_count,
        "predicted_occurrences": prediction_count,
        "mean_tiou": round(matched_sum / reference_count, 6) if reference_count else None,
        "thresholds": thresholds,
    }


@dataclass(frozen=True)
class RunComparison:
    repo: Path
    run_dir: Path
    run: dict[str, Any]
    reference_revision: str

    @classmethod
    def load(cls, repo: Path, run_id: str, reference_revision: str | None = None) -> "RunComparison":
        repo = repo.resolve()
        run_dir = repo / "runs" / run_id
        run_path = run_dir / "run.yaml"
        if not run_path.is_file():
            raise ValueError(f"prediction runがありません: {run_id}")
        run = yaml.safe_load(run_path.read_text(encoding="utf-8")) or {}
        if run.get("kind") != "prediction" or run.get("status") != "complete":
            raise ValueError(f"completeなprediction runが必要です: {run_id}")
        if not run.get("target_units"):
            raise ValueError(f"target_unitsが空です: {run_id}")

        if reference_revision is None:
            reference_revision = "human"
        return cls(repo, run_dir, run, reference_revision)

    @property
    def dataset_id(self) -> str:
        return str(self.run.get("dataset", {}).get("id", ""))

    @property
    def run_id(self) -> str:
        return str(self.run["run_id"])

    def public_summary(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "model_name": str(self.run.get("model", {}).get("name", self.run_id)),
            "reference_revision": self.reference_revision,
        }

    def has_complete_inputs(self) -> bool:
        dataset_root = self.repo / "datasets" / self.dataset_id
        return all(
            (self.run_dir / "predictions" / f"{unit_id}.json").is_file()
            and (dataset_root / "annotations" / self.reference_revision
                 / f"{unit_id}.json").is_file()
            for unit_id in self.run["target_units"]
        )

    @cached_property
    def overall_summary(self) -> dict[str, Any]:
        results = []
        dataset_root = self.repo / "datasets" / self.dataset_id
        for unit_id in self.run["target_units"]:
            prediction_path = self.run_dir / "predictions" / f"{unit_id}.json"
            reference_path = (dataset_root / "annotations" / self.reference_revision
                              / f"{unit_id}.json")
            if not prediction_path.is_file() or not reference_path.is_file():
                raise ValueError(f"比較入力が不足しています: {unit_id}")
            prediction = load_prediction(json.loads(prediction_path.read_text(encoding="utf-8")))
            reference = load_annotation(json.loads(reference_path.read_text(encoding="utf-8")))
            results.append(evaluate_temporal(reference, prediction))
        return _aggregate(results)

    def for_unit(self, unit: Unit) -> dict[str, Any] | None:
        if unit.dataset != self.dataset_id or unit.unit_id not in self.run["target_units"]:
            return None
        prediction_path = self.run_dir / "predictions" / f"{unit.unit_id}.json"
        reference_path = (
            self.repo / "datasets" / unit.dataset / "annotations"
            / self.reference_revision / f"{unit.unit_id}.json"
        )
        if not prediction_path.is_file() or not reference_path.is_file():
            return None
        prediction_doc = json.loads(prediction_path.read_text(encoding="utf-8"))
        reference_doc = json.loads(reference_path.read_text(encoding="utf-8"))
        reference = load_annotation(reference_doc)
        prediction = load_prediction(prediction_doc)
        result = evaluate_temporal(reference, prediction)
        event_rows: dict[str, list[dict[str, Any]]] = {}
        for row in result["events"]:
            event_rows.setdefault(row["event"], []).append(row)

        sop_labels = {
            event["id"]: event.get("ask", event["id"])
            for event in unit.load_sop().get("events", [])
        }
        event_ids = list(reference_doc["events"])
        events = []
        for event_id in event_ids:
            reference_spans = reference_doc["events"].get(event_id)
            prediction_spans = prediction_doc["events"].get(event_id)
            rows = event_rows.get(event_id, [])
            reference_count = len(reference_spans or [])
            matched_sum = sum(
                float(row["tiou"]) for row in rows
                if row.get("status") == "match" and row.get("tiou") is not None
            )
            event_tiou = matched_sum / reference_count if reference_count else None
            events.append({
                "event_id": event_id,
                "text": sop_labels.get(event_id, event_id.replace("_", " ")),
                "reference_spans": reference_spans,
                "prediction_spans": prediction_spans,
                "tiou": round(event_tiou, 6) if event_tiou is not None else None,
            })
        return {
            "run_id": self.run_id,
            "model_name": self.run.get("model", {}).get("name", self.run_id),
            "method": prediction_doc.get("method"),
            "reference_revision": self.reference_revision,
            "reference_label": "人手アノテーション",
            "summary": result["summary"],
            "overall_summary": self.overall_summary,
            "events": events,
        }


def discover_run_comparisons(repo: Path) -> dict[str, RunComparison]:
    """結果レビューで比較可能なcomplete runをrepositoryから自動発見する。"""
    comparisons: dict[str, RunComparison] = {}
    runs_root = repo.resolve() / "runs"
    if not runs_root.is_dir():
        return comparisons
    for run_path in sorted(runs_root.glob("*/run.yaml"), reverse=True):
        try:
            run = yaml.safe_load(run_path.read_text(encoding="utf-8")) or {}
            run_id = run.get("run_id")
            if not isinstance(run_id, str):
                continue
            comparison = RunComparison.load(repo, run_id)
            if comparison.has_complete_inputs():
                comparisons[run_id] = comparison
        except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError):
            continue
    return comparisons
