#!/usr/bin/env python3
"""Qwen2.5-VL-3Bへ20秒動画を直接入力し、eventごとの秒区間を保存する。"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from small_vlm_sop_check.core.temporal import TimeSpan, prediction_document  # noqa: E402
from run_local_prediction import (  # noqa: E402
    DATASET_ID, SPLIT_ID, build_inputs_lock, git_revision, hf_snapshot_revision,
)
from run_marlin_prediction import (  # noqa: E402
    VIDEO_WIDTH, ensure_video, validate_queries, write_json_atomic,
)


DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
PROMPT_TEMPLATE = """Event: {query}
Locate this event in the 20-second video. Return exactly one JSON object: {{"start_s": number, "end_s": number}}. If the event is absent, return {{"start_s": null, "end_s": null}}. Times are seconds from video start."""


def parse_span(text: str) -> tuple[float, float] | None:
    """fenceや説明が混ざっても最初のJSON objectだけを厳密に読む。"""
    for match in re.finditer(r"\{", text):
        try:
            value, _ = json.JSONDecoder().raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        start, end = value.get("start_s"), value.get("end_s")
        if start is None and end is None:
            return None
        try:
            start, end = float(start), float(end)
        except (TypeError, ValueError):
            continue
        if 0 <= start < end <= 20:
            return start, end
    return None


def normalize(run_id: str, unit_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    events = {}
    for event_id, record in raw["events"].items():
        span = parse_span(record["response"])
        events[event_id] = None if span is None else [TimeSpan(*span)]
    return prediction_document(run_id, unit_id, "temporal_grounding", events)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-name", default="Qwen2.5-VL-3B video grounding")
    parser.add_argument("--model-parameters-b", type=float, default=3.0)
    parser.add_argument("--video-dir", default=str(ROOT / "out" / "model-videos"))
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-pixels", type=int, default=50176)
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()
    if not 0 < args.model_parameters_b <= 4.0:
        raise SystemExit(
            f"このplatformは4B以下のみを標準実行します: {args.model_parameters_b}B"
        )
    query_path = Path(args.queries).resolve()
    queries = json.loads(query_path.read_text(encoding="utf-8"))
    validate_queries(queries, require_sop_match=True)
    run_dir = ROOT / "runs" / args.run_id
    if (run_dir / "run.yaml").exists():
        raise SystemExit(f"既存runは不変です。上書きしません: {run_dir}")

    pending = []
    for unit_id, events in queries.items():
        raw_path = run_dir / "raw" / f"{unit_id}.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else {"events": {}}
        for event_id, query in events.items():
            record = raw["events"].get(event_id)
            if record and record.get("query") != query:
                raise SystemExit(f"resume rawのqueryが異なります: {unit_id}/{event_id}")
            if record is None:
                pending.append((unit_id, event_id))

    model = processor = None
    if pending:
        from mlx_vlm import load
        print(f"[qwen-video] loading {args.model}; pending={len(pending)}", flush=True)
        model, processor = load(args.model)

    for unit_index, (unit_id, events) in enumerate(queries.items(), 1):
        raw_path = run_dir / "raw" / f"{unit_id}.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else {
            "unit_id": unit_id, "model_id": args.model, "events": {},
        }
        video = None
        for event_id, query in events.items():
            if event_id in raw["events"]:
                continue
            from mlx_vlm import apply_chat_template, generate
            video = video or ensure_video(unit_id, Path(args.video_dir))
            instruction = PROMPT_TEMPLATE.format(query=query)
            prompt = apply_chat_template(processor, model.config, instruction,
                                         video=str(video), fps=args.fps,
                                         max_pixels=args.max_pixels)
            result = generate(model, processor, prompt, video=str(video), fps=args.fps,
                              max_tokens=args.max_tokens, temperature=0, verbose=False)
            raw["events"][event_id] = {
                "query": query, "prompt": instruction, "response": result.text,
                "prompt_tokens": result.prompt_tokens,
                "generation_tokens": result.generation_tokens,
                "peak_memory_gb": result.peak_memory,
            }
            write_json_atomic(raw_path, raw)
            print(f"[qwen-video] {unit_index}/{len(queries)} {unit_id} {event_id}: "
                  f"{result.text.strip()}", flush=True)
        write_json_atomic(run_dir / "predictions" / f"{unit_id}.json",
                          normalize(args.run_id, unit_id, raw))

    units = list(queries)
    write_json_atomic(run_dir / "inputs.lock.json", build_inputs_lock(units))
    revision = hf_snapshot_revision(args.model)
    run_doc = {
        "run_id": args.run_id, "kind": "prediction", "status": "complete",
        "immutable": True, "created_at": datetime.date.today().isoformat(),
        "model": {"name": args.model_name, "role": "local_small_vlm_temporal_grounding",
                  "id": args.model, "revision": revision,
                  "parameters_b": args.model_parameters_b},
        "dataset": {"id": DATASET_ID, "split": SPLIT_ID}, "target_units": units,
        "ground_truth_used": False, "metrics": None,
        "inference_code_revision": git_revision(),
        "notes": ["Full 20-second video input; human GT is never read by inference."],
        "inference": {
            "backend": "mlx-vlm", "method": "per-event video prompting",
            "query_file": str(query_path.relative_to(ROOT)),
            "query_ontology": "unit_sop",
            "prompt_template": PROMPT_TEMPLATE, "video_fps": args.fps,
            "video_width": VIDEO_WIDTH, "max_pixels": args.max_pixels,
            "max_tokens": args.max_tokens,
            "temperature": 0,
        },
    }
    (run_dir / "run.yaml").write_text(
        yaml.safe_dump(run_doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    with (ROOT / "runs" / "index.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"dataset": DATASET_ID, "formal_accuracy": None,
                                 "kind": "prediction", "model": args.model_name,
                                 "role": "local_small_vlm_temporal_grounding",
                                 "run_id": args.run_id, "split": SPLIT_ID,
                                 "unit_count": len(units)}, ensure_ascii=False,
                                sort_keys=True) + "\n")
    print(f"[qwen-video] 完了: {run_dir}")


if __name__ == "__main__":
    main()
