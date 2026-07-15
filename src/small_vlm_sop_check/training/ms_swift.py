"""人手annotationをms-swift動画SFT用JSONLへexportする。

学習エンジンは依存に含めない。このmoduleはrepositoryのunit/SOP/annotation契約を、
ms-swiftが直接読めるmessages+videos形式へ決定論的に変換するだけを担当する。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from ..apps import catalog
from ..core.sop import load_sop


SYSTEM_PROMPT = (
    "You locate industrial procedure events in egocentric video. "
    "Return strict JSON only. Timestamps are seconds relative to the start of the input video."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotation_events_seconds(gt: dict[str, Any]) -> dict[str, list[dict[str, float]]]:
    """現行annotationのhalf-open秒区間をSFT応答用に整形する。"""
    if gt.get("interval_convention") != "half-open_seconds":
        raise ValueError("annotation interval_convention must be half-open_seconds")
    events: dict[str, list[dict[str, float]]] = {}
    for event_id, spans in gt["events"].items():
        if spans is None:
            events[event_id] = []
            continue
        events[event_id] = [{"start_s": span["start_s"], "end_s": span["end_s"]}
                            for span in spans]
    return events


def build_sample(unit: catalog.Unit, video_path: Path) -> dict[str, Any]:
    gt = json.loads(unit.gt_path.read_text(encoding="utf-8"))
    events = annotation_events_seconds(gt)
    sop = load_sop(unit.sop_path)
    questions = {event["id"]: event.get("ask", event["id"])
                 for event in sop["events"] if event["id"] in events}
    if not questions:
        raise ValueError(f"{unit.unit_id}: annotationとSOPに共通イベントがありません")
    query_lines = ["<video>", "Locate every listed event. Use an empty list when absent."]
    query_lines.extend(f"- {event_id}: {question}" for event_id, question in questions.items())
    answer = {"events": {event_id: events[event_id] for event_id in questions}}
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(query_lines)},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False,
                                                            separators=(",", ":"))},
        ],
        "videos": [str(video_path.resolve())],
        "unit_id": unit.unit_id,
    }


def ensure_video(unit: catalog.Unit, media_dir: Path) -> Path:
    """unitの抽出フレームをfps付きMP4へまとめる。既存出力は再利用する。"""
    out = media_dir / f"{unit.unit_id}.mp4"
    frames = unit.frame_files()
    if not frames:
        raise ValueError(f"{unit.unit_id}: framesがありません: {unit.frames_dir}")
    digits = len(Path(frames[0]).stem.removeprefix("f"))
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(unit.fps),
        "-i", str(unit.frames_dir / f"f%0{digits}d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise ValueError("ffmpegが必要です") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"{unit.unit_id}: MP4生成に失敗しました") from exc
    return out


def _load_split(path: Path, subset: str) -> list[str]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    assignments = doc.get("assignments", {})
    if subset not in assignments:
        raise ValueError(f"split subsetがありません: {subset} ({path})")
    return list(assignments[subset])


def export(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo.resolve()
    dataset_root = root / "datasets" / args.dataset
    dataset_doc = yaml.safe_load((dataset_root / "dataset.yaml").read_text(encoding="utf-8"))
    benchmark = dataset_doc.get("benchmark_state", {})
    human = benchmark.get("human_ground_truth")
    if isinstance(human, dict):
        annotation_state = human.get("status", "unknown")
    elif benchmark.get("human_ground_truth_available") is True:
        annotation_state = "complete"
    else:
        annotation_state = "none"
    if annotation_state != "complete" and not args.allow_partial:
        raise ValueError(
            f"annotation status={annotation_state!r}。全unitが完了していないデータは既定でexportしません。"
            "試験目的だけなら --allow-partial を明示してください。"
        )

    split_path = args.split.resolve()
    target_ids = _load_split(split_path, args.subset)
    units = catalog.discover(root)
    selected = []
    for unit_id in target_ids:
        unit = units.get(unit_id)
        if unit and unit.dataset == args.dataset and unit.has_gt():
            selected.append(unit)
    if not selected:
        raise ValueError("対象subsetにannotation済みunitがありません")

    out_dir = args.out.resolve()
    media_dir = out_dir / "media"
    videos = {unit.unit_id: ensure_video(unit, media_dir) for unit in selected}
    samples = [build_sample(unit, videos[unit.unit_id]) for unit in selected]
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "train.jsonl"
    tmp = jsonl_path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
                   encoding="utf-8")
    os.replace(tmp, jsonl_path)

    manifest = {
        "format": "ms-swift-messages-video-sft",
        "dataset_id": args.dataset,
        "annotation_revision": human.get("revision") if isinstance(human, dict) else None,
        "annotation_state": annotation_state,
        "split": {"path": str(split_path), "subset": args.subset,
                  "sha256": _sha256(split_path)},
        "sample_count": len(samples),
        "unit_ids": [unit.unit_id for unit in selected],
        "jsonl": {"path": jsonl_path.name, "sha256": _sha256(jsonl_path)},
        "media": {
            unit_id: {"path": str(path.relative_to(out_dir)), "sha256": _sha256(path)}
            for unit_id, path in sorted(videos.items())
        },
        "interval_convention": "half-open_seconds",
    }
    manifest_path = out_dir / "export.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--subset", default="train")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true",
                        help="一部unitだけのannotationを試験exportする")
    args = parser.parse_args()
    try:
        manifest = export(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"[export-ms-swift] {exc}") from exc
    print(f"[export-ms-swift] {manifest['sample_count']} samples -> {args.out}")


if __name__ == "__main__":
    main()
