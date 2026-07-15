"""動画イベント区間の読み書きと秒単位Temporal IoU評価。

区間はすべてunit先頭を0秒とするhalf-open interval ``[start_s, end_s)``。
注釈と予測は同じ秒区間契約を使う。フレーム分類の回答列は保存前に秒区間へ変換する。
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .events import detect_events


@dataclass(frozen=True, order=True)
class TimeSpan:
    start_s: float
    end_s: float
    score: float | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.start_s) and math.isfinite(self.end_s)):
            raise ValueError(f"区間に有限でない値があります: {self}")
        if self.start_s < 0 or self.end_s <= self.start_s:
            raise ValueError(f"区間は 0 <= start_s < end_s が必要です: {self}")
        if self.score is not None and not 0 <= self.score <= 1:
            raise ValueError(f"scoreは0..1が必要です: {self.score}")

    def as_dict(self) -> dict[str, float]:
        out = {"start_s": round(self.start_s, 3), "end_s": round(self.end_s, 3)}
        if self.score is not None:
            out["score"] = round(self.score, 6)
        return out


EventSpans = dict[str, list[TimeSpan] | None]


def temporal_iou(a: TimeSpan, b: TimeSpan) -> float:
    intersection = max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))
    union = max(a.end_s, b.end_s) - min(a.start_s, b.start_s)
    return intersection / union if intersection > 0 and union > 0 else 0.0


def _span(value: dict[str, Any]) -> TimeSpan:
    return TimeSpan(float(value["start_s"]), float(value["end_s"]),
                    float(value["score"]) if value.get("score") is not None else None)


def _normalize_span_values(events: dict[str, Any]) -> EventSpans:
    normalized: EventSpans = {}
    for event_id, values in events.items():
        if values is None or values == []:
            normalized[event_id] = None
            continue
        if not isinstance(values, list):
            raise ValueError(f"events.{event_id} は区間リストかnullで指定します")
        spans = sorted((_span(value) for value in values), key=lambda span: span.start_s)
        normalized[event_id] = spans
    return normalized


def load_annotation(document: dict[str, Any]) -> EventSpans:
    """現行の人手注釈を検証して読み込む。"""
    if not document.get("unit_id") or document.get("interval_convention") != "half-open_seconds":
        raise ValueError("annotationにはunit_idとinterval_convention=half-open_secondsが必要です")
    if not isinstance(document.get("events"), dict):
        raise ValueError("annotation.eventsはobjectが必要です")
    return _normalize_span_values(document["events"])


def _frame_step(frames: list[dict[str, Any]]) -> float:
    times = [float(frame["t"]) for frame in frames]
    deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
    return statistics.median(deltas) if deltas else 1.0


def frame_answers_to_spans(frames: list[dict[str, Any]],
                           sop_def: dict[str, Any]) -> EventSpans:
    """フレーム分類の回答列を、現行の秒区間predictionへ変換する。"""
    frames = sorted(frames, key=lambda frame: frame["idx"])
    if not frames:
        raise ValueError("frame predictionにframesがありません")
    step = _frame_step(frames)
    by_idx = {int(frame["idx"]): frame for frame in frames}
    detected = detect_events(sop_def["events"], frames, sop_def.get("defaults"))
    events: EventSpans = {}
    for event in sop_def["events"]:
        runs = detected.get(event["id"], [])
        if not runs:
            events[event["id"]] = None
            continue
        spans = []
        for run in runs:
            start_frame = by_idx.get(run.start_idx)
            end_frame = by_idx.get(run.end_idx)
            if start_frame is None or end_frame is None:
                raise ValueError(f"frame indexが不連続です: {event['id']}")
            spans.append(TimeSpan(float(start_frame["t"]), float(end_frame["t"]) + step))
        events[event["id"]] = spans
    return events


def load_prediction(document: dict[str, Any]) -> EventSpans:
    """現行のモデル予測を検証して読み込む。"""
    required = (document.get("run_id"), document.get("unit_id"), document.get("method"))
    if not all(required) or document.get("interval_convention") != "half-open_seconds":
        raise ValueError(
            "predictionにはrun_id, unit_id, method, interval_convention=half-open_secondsが必要です"
        )
    if not isinstance(document.get("events"), dict):
        raise ValueError("prediction.eventsはobjectが必要です")
    return _normalize_span_values(document["events"])


def prediction_document(run_id: str, unit_id: str, method: str,
                        events: EventSpans) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "unit_id": unit_id,
        "method": method,
        "interval_convention": "half-open_seconds",
        "events": {
            event_id: None if spans is None else [span.as_dict() for span in spans]
            for event_id, spans in events.items()
        },
    }


def _best_pairs(gt: list[TimeSpan], pred: list[TimeSpan]) -> list[tuple[int, int]]:
    """総tIoU最大のone-to-one対応。通常の少数occurrenceに対し厳密探索する。"""
    if not gt or not pred:
        return []
    target = min(len(gt), len(pred))
    if max(len(gt), len(pred)) > 12:
        # 異常に多い反復では指数探索を避け、時系列対応を明示的fallbackにする。
        return list(zip(range(target), range(target)))

    @lru_cache(maxsize=None)
    def solve(gt_index: int, used_mask: int, matched: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if gt_index == len(gt):
            return (0.0, ()) if matched == target else (-math.inf, ())
        remaining_gt = len(gt) - gt_index
        need = target - matched
        best = (-math.inf, ())
        if remaining_gt > need:
            best = solve(gt_index + 1, used_mask, matched)
        if need > 0:
            for pred_index in range(len(pred)):
                if used_mask & (1 << pred_index):
                    continue
                score, pairs = solve(gt_index + 1, used_mask | (1 << pred_index), matched + 1)
                score += temporal_iou(gt[gt_index], pred[pred_index])
                candidate = (score, ((gt_index, pred_index),) + pairs)
                if candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
                    best = candidate
        return best

    return list(solve(0, 0, 0)[1])


def evaluate_temporal(annotation: EventSpans, prediction: EventSpans,
                      thresholds: tuple[float, ...] = (0.1, 0.3, 0.5)) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    matched_ious: list[float] = []
    gt_count = 0
    pred_count = 0
    prediction_missing = 0
    annotated_events = set(annotation)

    for event_id, gt_value in annotation.items():
        has_prediction = event_id in prediction
        pred_value = prediction.get(event_id)
        if not has_prediction:
            prediction_missing += 1
            rows.append({"event": event_id, "status": "prediction_missing",
                         "gt": None if gt_value is None else [span.as_dict() for span in gt_value],
                         "predicted": None, "tiou": None})
            gt_count += len(gt_value or [])
            continue
        if gt_value is None:
            if pred_value is None:
                rows.append({"event": event_id, "status": "true_absent", "gt": None,
                             "predicted": None, "tiou": None})
            else:
                pred_count += len(pred_value)
                for index, span in enumerate(pred_value, 1):
                    rows.append({"event": event_id, "occurrence": index,
                                 "status": "false_detection", "gt": None,
                                 "predicted": span.as_dict(), "tiou": None})
            continue

        gt_count += len(gt_value)
        pred_spans = pred_value or []
        pred_count += len(pred_spans)
        pairs = _best_pairs(gt_value, pred_spans)
        paired_gt = {gt_index for gt_index, _ in pairs}
        paired_pred = {pred_index for _, pred_index in pairs}
        for gt_index, pred_index in sorted(pairs):
            iou = temporal_iou(gt_value[gt_index], pred_spans[pred_index])
            matched_ious.append(iou)
            rows.append({"event": event_id, "occurrence": gt_index + 1,
                         "status": "match", "gt": gt_value[gt_index].as_dict(),
                         "predicted": pred_spans[pred_index].as_dict(),
                         "tiou": round(iou, 6)})
        for gt_index, span in enumerate(gt_value):
            if gt_index not in paired_gt:
                rows.append({"event": event_id, "occurrence": gt_index + 1,
                             "status": "miss", "gt": span.as_dict(),
                             "predicted": None, "tiou": 0.0})
        for pred_index, span in enumerate(pred_spans):
            if pred_index not in paired_pred:
                rows.append({"event": event_id, "occurrence": pred_index + 1,
                             "status": "false_detection", "gt": None,
                             "predicted": span.as_dict(), "tiou": None})

    for event_id in sorted(set(prediction) - annotated_events):
        spans = prediction[event_id]
        rows.append({"event": event_id, "status": "no_gt", "gt": None,
                     "predicted": None if spans is None else [span.as_dict() for span in spans],
                     "tiou": None})

    metrics: dict[str, Any] = {}
    for threshold in thresholds:
        true_positive = sum(iou >= threshold for iou in matched_ious)
        false_positive = pred_count - true_positive
        false_negative = gt_count - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall else None)
        key = f"tiou@{threshold:g}"
        metrics[key] = {
            "tp": true_positive, "fp": false_positive, "fn": false_negative,
            "precision": round(precision, 6) if precision is not None else None,
            "recall": round(recall, 6) if recall is not None else None,
            "f1": round(f1, 6) if f1 is not None else None,
        }
    mean_tiou = sum(matched_ious) / gt_count if gt_count else None
    return {
        "metric": "temporal_iou_seconds",
        "interval_convention": "half-open_seconds",
        "summary": {
            "mean_tiou": round(mean_tiou, 6) if mean_tiou is not None else None,
            "gt_occurrences": gt_count,
            "predicted_occurrences": pred_count,
            "prediction_missing_events": prediction_missing,
            "thresholds": metrics,
        },
        "events": rows,
    }


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_temporal_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "Temporal event evaluation (seconds, half-open intervals)",
        f"mean tIoU: {summary['mean_tiou']}",
        f"GT occurrences: {summary['gt_occurrences']} / predictions: {summary['predicted_occurrences']}",
    ]
    for threshold, metrics in summary["thresholds"].items():
        lines.append(
            f"{threshold}: P={metrics['precision']} R={metrics['recall']} F1={metrics['f1']} "
            f"(TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']})"
        )
    if summary["prediction_missing_events"]:
        lines.append(f"prediction missing events: {summary['prediction_missing_events']}")
    return "\n".join(lines)
