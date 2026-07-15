"""人手動画アノテーションの読み書き。

UIフレームワークから独立した小さな保存層にし、HTTP APIやCLIからも
同じ契約を使えるようにする。人手の正本は日本語ラベルを持つ
SOP YAMLと、half-open秒区間を持つannotation JSONの組である。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any

import yaml

from ..core.sop import save_sop, validate_sop
from .catalog import Unit


ANNOTATION_REVISION = "human"
_LOCKS_GUARD = threading.Lock()
_DATASET_LOCKS: dict[Path, threading.RLock] = {}


def _dataset_lock(unit: Unit) -> threading.RLock:
    # 異なるunitも同じdataset.yamlの進捗を更新するため、dataset単位で直列化する。
    key = unit.gt_path.resolve().parents[2]
    with _LOCKS_GUARD:
        return _DATASET_LOCKS.setdefault(key, threading.RLock())


def load_document(unit: Unit) -> dict[str, Any]:
    """未着手unitにも同じ形の空documentを返す。"""
    labels = {
        str(event["id"]): str(event.get("ask", ""))
        for event in unit.load_sop().get("events", [])
    }
    if unit.gt_path.is_file():
        document = json.loads(unit.gt_path.read_text(encoding="utf-8"))
    else:
        document = {
            "unit_id": unit.unit_id,
            "annotation_revision": ANNOTATION_REVISION,
            "interval_convention": "half-open_seconds",
            "event_labels": labels,
            "events": {},
        }
    document.setdefault("event_labels", labels)
    document.setdefault("events", {})
    return document


def validate_document(
    unit: Unit, document: dict[str, Any], *, known_events: set[str] | None = None
) -> None:
    if document.get("unit_id") != unit.unit_id:
        raise ValueError("annotationのunit_idが選択中の動画と一致しません")
    if document.get("annotation_revision") != ANNOTATION_REVISION:
        raise ValueError("annotation_revisionはhumanが必要です")
    if document.get("interval_convention") != "half-open_seconds":
        raise ValueError("interval_conventionはhalf-open_secondsが必要です")
    unexpected = set(document) - {
        "unit_id", "annotation_revision", "interval_convention", "event_labels", "events"
    }
    if unexpected:
        raise ValueError(f"annotationに不要なfieldがあります: {sorted(unexpected)}")
    events = document.get("events")
    if not isinstance(events, dict):
        raise ValueError("eventsはobjectが必要です")
    known = known_events if known_events is not None else {
        event["id"] for event in unit.load_sop().get("events", [])
    }
    unknown = set(events) - known
    if unknown:
        raise ValueError(f"SOPにないイベントがあります: {sorted(unknown)}")
    event_labels = document.get("event_labels")
    if not isinstance(event_labels, dict) or set(event_labels) != known:
        raise ValueError("event_labelsは全SOPイベントの日本語文を持つ必要があります")
    if any(not isinstance(label, str) or not label.strip() for label in event_labels.values()):
        raise ValueError("event_labelsの日本語イベント文は空にできません")
    for event_id, spans in events.items():
        if spans is None:
            continue
        if not isinstance(spans, list) or not spans:
            raise ValueError(f"{event_id}: 区間リストまたはnullが必要です")
        for span in spans:
            if not isinstance(span, dict) or set(span) != {"start_s", "end_s"}:
                raise ValueError(f"{event_id}: 各区間にはstart_sとend_sが必要です")
        previous_end = -1.0
        for span in sorted(spans, key=lambda item: float(item["start_s"])):
            start = float(span["start_s"])
            end = float(span["end_s"])
            if start < 0 or end <= start or end > unit.duration_s + 1e-6:
                raise ValueError(
                    f"{event_id}: 0 <= 開始 < 終了 <= {unit.duration_s:g} が必要です"
                )
            if start < previous_end - 1e-9:
                raise ValueError(f"{event_id}: 同じイベントの区間が重なっています")
            previous_end = end


def save_document(unit: Unit, document: dict[str, Any]) -> None:
    """検証済みdocumentを原子的に保存し、dataset進捗も同期する。"""
    document = dict(document)
    document["events"] = {
        key: (None if value is None else sorted(value, key=lambda span: span["start_s"]))
        for key, value in document["events"].items()
    }
    validate_document(unit, document)
    unit.gt_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        unit.gt_path,
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
    )
    sync_dataset_progress(unit)


def save_sop_and_document(
    unit: Unit, sop: dict[str, Any], document: dict[str, Any]
) -> None:
    """イベント定義と区間を、両方検証してから順に原子的保存する。"""
    with _dataset_lock(unit):
        event_ids = [event.get("id") for event in sop.get("events", [])]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("イベントIDが重複しています")
        if any(not event_id or not str(event_id).isidentifier() for event_id in event_ids):
            raise ValueError("イベントIDが不正です")
        if any(not str(event.get("ask", "")).strip() for event in sop.get("events", [])):
            raise ValueError("すべてのイベントに日本語イベント文が必要です")
        document["event_labels"] = {
            str(event["id"]): str(event["ask"]).strip() for event in sop.get("events", [])
        }
        validate_sop(sop, unit.sop_path)
        validate_document(unit, document, known_events=set(event_ids))
        save_sop(unit.sop_path, sop)
        save_document(unit, document)


def sync_dataset_progress(unit: Unit) -> None:
    if unit.meta_path is None:
        return
    dataset_root = unit.meta_path.parents[2]
    dataset_path = dataset_root / "dataset.yaml"
    if not dataset_path.is_file():
        return
    dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8")) or {}
    state = dataset.setdefault("benchmark_state", {})
    human = state.setdefault("human_ground_truth", {})
    annotations = dataset_root / "annotations" / ANNOTATION_REVISION
    documents = []
    for path in annotations.glob("*.json"):
        try:
            documents.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    total = len(list((dataset_root / "units").glob("*/meta.json")))
    complete = sum(
        set(item.get("events", {})) == set(item.get("event_labels", {}))
        for item in documents
    )
    human.clear()
    human.update({
        "status": "complete" if total > 0 and complete == total else "partial" if documents else "none",
        "revision": ANNOTATION_REVISION,
    })
    # 全動画の人手確認完了と、未見testに対する正式精度の可否は別概念。
    # アノテーションUIがformal accuracyを自動昇格させてはならない。
    state.setdefault("formal_accuracy_available", False)
    _atomic_text(
        dataset_path,
        yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False),
    )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
