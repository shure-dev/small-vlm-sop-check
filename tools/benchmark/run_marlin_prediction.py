#!/usr/bin/env python3
"""Marlin-2Bの動画groundingをFactory Egoの秒区間predictionとして保存する。

Marlinの ``find(video, event)`` はイベントの開始・終了秒を返す。このスクリプトは
その区間を量子化せず、共通のprediction契約で保存する。
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from small_vlm_sop_check.core.sop import load_sop  # noqa: E402
from small_vlm_sop_check.core.temporal import (  # noqa: E402
    TimeSpan,
    prediction_document,
)
from run_local_prediction import (  # noqa: E402
    DATASET_ID,
    SPLIT_ID,
    build_inputs_lock,
    git_revision,
    hf_snapshot_revision,
    sha256_file,
    unit_fps,
    unit_paths,
)

DEFAULT_MODEL = "lunahr/Marlin-2B-ungated"
DEFAULT_REVISION = "de783b96b80f477c5e665d2202571a84cb0761da"
VIDEO_WIDTH = 640
MAX_MODEL_PARAMETERS_B = 4.0


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def validate_queries(
    queries: dict[str, dict[str, str]],
    *,
    require_sop_match: bool,
    allow_event_subset: bool = False,
) -> None:
    if not queries:
        raise SystemExit("queriesが空です")
    for unit_id, events in queries.items():
        if require_sop_match:
            sop = load_sop(unit_paths(unit_id)["sop"])
            expected = {event["id"] for event in sop["events"]}
            actual = set(events)
            valid_ids = actual <= expected if allow_event_subset else actual == expected
            if not actual or not valid_ids:
                raise SystemExit(
                    f"queryのevent IDがSOPと一致しません: {unit_id} "
                    f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
                )
        if any(not isinstance(query, str) or not query.strip() for query in events.values()):
            raise SystemExit(f"空または文字列でないqueryがあります: {unit_id}")


def ensure_video(unit_id: str, video_dir: Path) -> Path:
    """gated framesからMarlin入力用MP4を決定論的に生成する。"""
    out = video_dir / f"{unit_id}.mp4"
    paths = unit_paths(unit_id)
    if not any(paths["frames"].glob("f*.jpg")):
        if out.is_file():
            return out
        raise SystemExit(f"framesがありません(gated媒体を先にfetch): {paths['frames']}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.stem + ".tmp" + out.suffix)
    command = [
        # -threads 1: マルチスレッドlibx264はエンコード結果が非決定的で、同一フレームでも
        # 生成MP4のバイトが変わりMarlinの秒区間出力がぶれる。単一スレッドで決定論化し、
        # runの再現性を確保する（再エンコードしても同じpredictionになる）。
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(unit_fps(unit_id)),
        "-i", str(paths["frames"] / "f%04d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", f"scale={VIDEO_WIDTH}:-2", "-threads", "1",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("ffmpegが必要です（macOS: brew install ffmpeg）") from exc
    os.replace(temporary, out)
    return out


def video_metadata(path: Path) -> dict[str, Any]:
    """実際にモデルへ渡すMP4のハッシュと映像属性を記録する。"""
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
        "-show_entries", "format=duration", "-of", "json", str(path),
    ]
    try:
        value = json.loads(subprocess.check_output(command, text=True))
    except FileNotFoundError as exc:
        raise SystemExit("ffprobeが必要です（ffmpegに同梱）") from exc
    stream = value["streams"][0]
    return {
        "sha256": sha256_file(path),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": stream["avg_frame_rate"],
        "n_frames": int(stream["nb_frames"]),
        "duration_s": float(value["format"]["duration"]),
    }


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


def normalize_prediction(run_id: str, unit_id: str,
                         raw: dict[str, Any]) -> dict[str, Any]:
    """Marlinの秒区間を量子化せず共通predictionとして保持する。"""
    events = {}
    for event_id, record in raw["events"].items():
        span = result_span(record.get("result"))
        if span is None or span[1] <= span[0]:
            events[event_id] = None
        else:
            events[event_id] = [TimeSpan(span[0], span[1])]
    return prediction_document(run_id, unit_id, "temporal_grounding", events)


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
    parser.add_argument(
        "--allow-event-subset", action="store_true",
        help="annotation差分診断用。各unitのqueryをSOP event IDの非空部分集合として許可",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION,
                        help="trust_remote_codeで読むモデルcommit（既定は実験時revision）")
    parser.add_argument("--model-name", default="Marlin-2B temporal grounding")
    parser.add_argument("--model-parameters-b", type=float, default=2.0)
    parser.add_argument("--role", default="local_small_vlm_temporal_grounding")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--video-dir", default=str(ROOT / "out" / "marlin-videos"))
    args = parser.parse_args()

    if not 0 < args.model_parameters_b <= MAX_MODEL_PARAMETERS_B:
        raise SystemExit(
            f"このplatformは4B以下のみを標準実行します: {args.model_parameters_b}B"
        )

    run_dir = ROOT / "runs" / args.run_id
    if (run_dir / "run.yaml").exists():
        raise SystemExit(f"既存runは不変です。上書きしません: {run_dir}")
    query_path = Path(args.queries).resolve()
    queries = json.loads(query_path.read_text(encoding="utf-8"))
    validate_queries(
        queries,
        require_sop_match=True,
        allow_event_subset=args.allow_event_subset,
    )
    units = list(queries)
    query_snapshot = run_dir / "queries.json"
    write_json_atomic(query_snapshot, queries)

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

    input_lock = build_inputs_lock(units)
    input_lock["queries_sha256"] = sha256_file(query_snapshot)
    input_lock["videos"] = {
        unit_id: video_metadata(Path(args.video_dir) / f"{unit_id}.mp4")
        for unit_id in units
    }
    write_json_atomic(run_dir / "inputs.lock.json", input_lock)
    model_revision = args.revision or hf_snapshot_revision(args.model)
    run_doc = {
        "run_id": args.run_id,
        "kind": "prediction",
        "status": "complete",
        "immutable": True,
        "created_at": datetime.date.today().isoformat(),
        "model": {"name": args.model_name, "role": args.role,
                  "id": args.model, "revision": model_revision,
                  "parameters_b": args.model_parameters_b},
        "dataset": {"id": DATASET_ID, "split": SPLIT_ID},
        "target_units": units,
        "ground_truth_used": False,
        "metrics": None,
        "inference_code_revision": git_revision(),
        "notes": [
            "Generated by tools/benchmark/run_marlin_prediction.py.",
            "Marlin find()の秒区間をpredictions/へ量子化せず保存。人手GTは推論に不使用。",
        ],
        "inference": {
            "backend": "transformers_remote_code",
            "method": "Marlin.find(video, event)",
            "device": resolved_device,
            "query_file": str(query_snapshot.relative_to(ROOT)),
            "query_source": (str(query_path.relative_to(ROOT))
                             if query_path.is_relative_to(ROOT) else str(query_path)),
            "query_ontology": (
                "unit_sop_event_subset" if args.allow_event_subset else "unit_sop"
            ),
            "event_subset": args.allow_event_subset,
            "decoding": "greedy",
            "sampling_fps": sorted({unit_fps(unit) for unit in units}),
            "frame_input": "full video encoded from canonical gated frames",
            "video_width": VIDEO_WIDTH,
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
