#!/usr/bin/env python3
"""Marlin-2Bの動画groundingを既存のFactory Ego prediction runへ正規化する。

Marlinの ``find(video, event)`` はイベントの開始・終了秒を返す。このスクリプトは
その区間を各抽出フレームのyes/noに変換し、既存の検証・評価・replay viewerが読む
``frame_question_answers`` schemaで保存する。コアの区間検出・評価方法は変更しない。
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from small_vlm_sop_check.core.sop import load_sop  # noqa: E402
from run_local_prediction import (  # noqa: E402
    DATASET_ID,
    SCHEMA_VERSION,
    SPLIT_ID,
    build_inputs_lock,
    git_revision,
    hf_snapshot_revision,
    unit_fps,
    unit_paths,
)

DEFAULT_MODEL = "lunahr/Marlin-2B-ungated"
DEFAULT_REVISION = "de783b96b80f477c5e665d2202571a84cb0761da"


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def validate_queries(queries: dict[str, dict[str, str]]) -> None:
    if not queries:
        raise SystemExit("queriesが空です")
    for unit_id, events in queries.items():
        sop = load_sop(unit_paths(unit_id)["sop"])
        expected = {event["id"] for event in sop["events"]}
        actual = set(events)
        if actual != expected:
            raise SystemExit(
                f"queryのevent IDがSOPと一致しません: {unit_id} "
                f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
            )
        if any(not isinstance(query, str) or not query.strip() for query in events.values()):
            raise SystemExit(f"空または文字列でないqueryがあります: {unit_id}")


def ensure_video(unit_id: str, video_dir: Path) -> Path:
    """gated framesからMarlin入力用MP4を決定論的に生成する。"""
    out = video_dir / f"{unit_id}.mp4"
    if out.is_file():
        return out
    paths = unit_paths(unit_id)
    if not any(paths["frames"].glob("f*.jpg")):
        raise SystemExit(f"framesがありません(gated媒体を先にfetch): {paths['frames']}")
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(unit_fps(unit_id)),
        "-i", str(paths["frames"] / "f%04d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("ffmpegが必要です（macOS: brew install ffmpeg）") from exc
    return out


def result_span(result: Any) -> tuple[float, float] | None:
    """Marlin find()の戻り値から妥当な単一区間を取り出す。"""
    span = result.get("span") if isinstance(result, dict) else result
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return None
    try:
        start, end = float(span[0]), float(span[1])
    except (TypeError, ValueError):
        return None
    if start < 0 or end < start:
        return None
    return start, end


def normalize_prediction(run_id: str, unit_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    fps = unit_fps(unit_id)
    meta = json.loads(unit_paths(unit_id)["meta"].read_text(encoding="utf-8"))
    n_frames = int(meta["sampling"]["n_frames"])
    spans = {
        event_id: (
            None if (span := result_span(record["result"])) is None
            else (math.floor(span[0] * fps), math.ceil(span[1] * fps))
        )
        for event_id, record in raw["events"].items()
    }
    frames = []
    for idx in range(n_frames):
        timestamp = round(idx / fps, 3)
        answers = {
            event_id: (
                "unclear" if span is None
                else "yes" if span[0] <= idx <= span[1]
                else "no"
            )
            for event_id, span in spans.items()
        }
        frames.append({"idx": idx, "t": timestamp, "answers": answers})
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "unit_id": unit_id,
        "prediction_type": "frame_question_answers",
        "answer_source": "Marlin find() timestamp span quantized by floor(start*fps)/ceil(end*fps)",
        "frame_count": n_frames,
        "frames": frames,
    }


def resolve_device(device: str) -> str:
    import torch

    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_id: str, revision: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM

    resolved = resolve_device(device)
    dtype = torch.float16 if resolved in {"mps", "cuda"} else torch.float32
    print(f"[marlin] model={model_id} device={resolved} dtype={dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, dtype=dtype,
    ).to(resolved)
    return model, resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, help="unit -> event ID -> English query のJSON")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION,
                        help="trust_remote_codeで読むモデルcommit（既定は実験時revision）")
    parser.add_argument("--model-name", default="Marlin-2B temporal grounding")
    parser.add_argument("--role", default="local_small_vlm_temporal_grounding")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--video-dir", default=str(ROOT / "out" / "marlin-videos"))
    args = parser.parse_args()

    run_dir = ROOT / "runs" / args.run_id
    if (run_dir / "run.yaml").exists():
        raise SystemExit(f"既存runは不変です。上書きしません: {run_dir}")
    query_path = Path(args.queries).resolve()
    queries = json.loads(query_path.read_text(encoding="utf-8"))
    validate_queries(queries)
    units = list(queries)

    pending = []
    for unit_id, events in queries.items():
        raw_path = run_dir / "raw" / f"{unit_id}.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else {"events": {}}
        if raw_path.exists() and raw.get("model_id") != args.model:
            raise SystemExit(f"resume rawのmodelが異なります: {raw_path}")
        if raw_path.exists() and raw.get("model_revision") != args.revision:
            raise SystemExit(f"resume rawのrevisionが異なります: {raw_path}")
        for event_id in events:
            record = raw.get("events", {}).get(event_id)
            if record and record.get("query") != events[event_id]:
                raise SystemExit(f"resume rawのqueryが異なります: {unit_id}/{event_id}")
            if record is None:
                pending.append((unit_id, event_id))

    model = None
    resolved_device = resolve_device(args.device)
    if pending:
        model, resolved_device = load_model(args.model, args.revision, args.device)

    for unit_id, events in queries.items():
        raw_path = run_dir / "raw" / f"{unit_id}.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else {
            "schema_version": SCHEMA_VERSION,
            "unit_id": unit_id,
            "model_id": args.model,
            "model_revision": args.revision,
            "events": {},
        }
        video = None
        for event_id, query in events.items():
            if event_id in raw["events"]:
                continue
            video = video or ensure_video(unit_id, Path(args.video_dir))
            result = model.find(str(video), event=query)
            raw["events"][event_id] = {"query": query, "result": result}
            write_json_atomic(raw_path, raw)
            print(f"[marlin] {unit_id} {event_id}: {result}", flush=True)
        prediction = normalize_prediction(args.run_id, unit_id, raw)
        write_json_atomic(run_dir / "predictions" / f"{unit_id}.json", prediction)

    write_json_atomic(run_dir / "inputs.lock.json", build_inputs_lock(units))
    model_revision = args.revision or hf_snapshot_revision(args.model)
    run_doc = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "kind": "prediction",
        "status": "complete",
        "immutable": True,
        "created_at": datetime.date.today().isoformat(),
        "model": {"name": args.model_name, "role": args.role,
                  "id": args.model, "revision": model_revision},
        "dataset": {"id": DATASET_ID, "split": SPLIT_ID},
        "target_units": units,
        "ground_truth_used": False,
        "metrics": None,
        "inference_code_revision": git_revision(),
        "notes": [
            "Generated by tools/benchmark/run_marlin_prediction.py.",
            "Marlin find()の単一区間をfloor(start*fps)/ceil(end*fps)でフレームyes/noへ正規化。人手GTは推論に不使用。",
        ],
        "inference": {
            "backend": "transformers_remote_code",
            "method": "Marlin.find(video, event)",
            "device": resolved_device,
            "query_file": (str(query_path.relative_to(ROOT))
                           if query_path.is_relative_to(ROOT) else str(query_path)),
            "sampling_fps": sorted({unit_fps(unit) for unit in units}),
            "frame_input": "full video encoded from canonical gated frames",
        },
    }
    (run_dir / "run.yaml").write_text(
        yaml.safe_dump(run_doc, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    index_path = ROOT / "runs" / "index.jsonl"
    entry = {"dataset": DATASET_ID, "formal_accuracy": None, "kind": "prediction",
             "model": args.model_name, "role": args.role, "run_id": args.run_id,
             "split": SPLIT_ID, "unit_count": len(units)}
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"[marlin] 完了: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
