"""手元の動画を公開リポジトリ安全なdataset layoutへ登録するCLI。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml


ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _yaml(path: Path, value: Any) -> None:
    _atomic_text(path, yaml.safe_dump(value, allow_unicode=True, sort_keys=False))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_id(value: str, label: str) -> str:
    if not ID_PATTERN.fullmatch(value):
        raise ValueError(f"{label}は小文字の英数字とunderscoreで指定します: {value!r}")
    return value


def parse_events(values: list[str]) -> list[dict[str, Any]]:
    events = []
    seen = set()
    for value in values:
        event_id, separator, ask = value.partition("=")
        validate_id(event_id, "event ID")
        if not separator or not ask.strip():
            raise ValueError("--eventは event_id=質問文 で指定します")
        if event_id in seen:
            raise ValueError(f"event IDが重複しています: {event_id}")
        seen.add(event_id)
        events.append({"id": event_id, "ask": ask.strip(), "values": ["yes", "no"]})
    return events


def init_dataset(repo: Path, dataset_id: str, name: str, description: str) -> Path:
    validate_id(dataset_id, "dataset ID")
    root = repo / "datasets" / dataset_id
    if root.exists():
        raise ValueError(f"datasetは既に存在します: {root}")
    for directory in ("annotations/human", "sops", "splits", "units"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    _yaml(root / "dataset.yaml", {
        "dataset_id": dataset_id,
        "name": name,
        "description": description,
        "media_policy": "local_not_committed",
        "benchmark_state": {
            "human_ground_truth": {
                "status": "none", "revision": "human",
            },
            "formal_accuracy_available": False,
        },
    })
    _json(root / "splits" / "development.json", {
        "split_id": "development",
        "group_by": ["source_group"],
        "assignments": {"dev_seen": [], "validation": [], "test": []},
        "policy": {"test_must_be_unseen": True, "current_units_never_promote_to_test": True},
    })
    _atomic_text(root / "README.md", (
        f"# {name}\n\n{description}\n\n"
        "動画と抽出フレームは `data/` にあり、Gitには含まれません。\n"
        f"`sop-view --dataset {dataset_id}` で確認できます。\n"
    ))
    return root


def _duration(video: Path) -> float:
    command = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", str(video)]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return float(result.stdout.strip())
    except FileNotFoundError as exc:
        raise ValueError("ffprobeが必要です（ffmpegをインストールしてください）") from exc
    except (subprocess.CalledProcessError, ValueError) as exc:
        raise ValueError(f"動画の長さを取得できません: {video}") from exc


def _run_ffmpeg(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise ValueError("ffmpegが必要です") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError("ffmpegによる動画変換に失敗しました") from exc


def add_video(repo: Path, dataset_id: str, unit_id: str, video: Path,
              events: list[dict[str, Any]], fps: float, start_s: float,
              end_s: float | None, source_group: str) -> Path:
    validate_id(dataset_id, "dataset ID")
    validate_id(unit_id, "unit ID")
    if fps <= 0:
        raise ValueError("--fpsは正数で指定します")
    video = video.resolve()
    if not video.is_file():
        raise ValueError(f"動画が見つかりません: {video}")
    dataset_root = repo / "datasets" / dataset_id
    dataset_path = dataset_root / "dataset.yaml"
    if not dataset_path.is_file():
        raise ValueError(f"先にdatasetをinitしてください: {dataset_id}")
    unit_root = dataset_root / "units" / unit_id
    data_root = repo / "data" / dataset_id / "units" / unit_id
    if unit_root.exists() or data_root.exists():
        raise ValueError(f"unitは既に存在します: {unit_id}")

    source_duration = _duration(video)
    end_s = source_duration if end_s is None else end_s
    if not 0 <= start_s < end_s <= source_duration + 0.01:
        raise ValueError(f"区間が不正です: 0 <= start < end <= {source_duration:.3f}")
    clip_duration = end_s - start_s
    data_root.mkdir(parents=True)
    try:
        output_video = data_root / "video.mp4"
        _run_ffmpeg([
            "ffmpeg", "-y", "-loglevel", "error", "-ss", str(start_s), "-i", str(video),
            "-t", str(clip_duration), "-map", "0:v:0", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(output_video),
        ])
        frames_dir = data_root / "frames"
        frames_dir.mkdir()
        _run_ffmpeg([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(output_video),
            "-vf", f"fps={fps}", "-q:v", "3", str(frames_dir / "f%04d.jpg"),
        ])
        frames = sorted(frames_dir.glob("f*.jpg"))
        if not frames:
            raise ValueError("フレームが1枚も抽出されませんでした")
        # ffmpegは1始まりなので、リポジトリ契約の0始まりへ改名する。
        for index, frame in enumerate(frames):
            frame.rename(frames_dir / f"frame-{index:04d}.jpg")
        for frame in sorted(frames_dir.glob("frame-*.jpg")):
            index = int(frame.stem.split("-")[1])
            frame.rename(frames_dir / f"f{index:04d}.jpg")

        unit_root.mkdir(parents=True)
        manifest = {frame.name: _sha256(frame) for frame in sorted(frames_dir.glob("f*.jpg"))}
        _json(unit_root / "frames.sha256.json", manifest)
        sop_path = dataset_root / "sops" / unit_id / "sop.yaml"
        _yaml(sop_path, {
            "sop": {"id": unit_id, "name": unit_id.replace("_", " ").title(),
                    "domain_hint": "産業作業の一人称視点動画"},
            "events": events,
        })
        _json(unit_root / "meta.json", {
            "unit_id": unit_id,
            "dataset_id": dataset_id,
            "benchmark_status": "dev_seen",
            "source": {
                "kind": "local_import", "original_filename": video.name,
                "source_sha256": _sha256(video), "source_group": source_group,
                "start_second": start_s, "end_second": end_s,
            },
            "sampling": {"fps": fps, "n_frames": len(manifest)},
            "media": {
                "availability": "local_not_committed", "path": "frames",
                "video_path": "video.mp4", "sha256_manifest": "frames.sha256.json",
            },
            "sop_ref": {"id": unit_id, "path": f"../../sops/{unit_id}/sop.yaml"},
        })

        split_path = dataset_root / "splits" / "development.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        split["assignments"]["dev_seen"].append(unit_id)
        _json(split_path, split)
    except Exception:
        shutil.rmtree(data_root, ignore_errors=True)
        shutil.rmtree(unit_root, ignore_errors=True)
        raise
    return unit_root


def validate_dataset(repo: Path, dataset_id: str, *, require_media: bool = False) -> dict[str, Any]:
    """dataset metadata・SOP・annotation・split分離を一括検査する。"""
    validate_id(dataset_id, "dataset ID")
    repo = repo.resolve()
    root = repo / "datasets" / dataset_id
    dataset_path = root / "dataset.yaml"
    errors: list[str] = []
    warnings: list[str] = []
    if not dataset_path.is_file():
        raise ValueError(f"dataset.yamlがありません: {dataset_path}")
    dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8")) or {}
    if dataset.get("dataset_id") != dataset_id:
        errors.append(f"dataset.yamlのdataset_idが一致しません: {dataset.get('dataset_id')!r}")

    metas: dict[str, dict[str, Any]] = {}
    durations: dict[str, float] = {}
    sop_events: dict[str, set[str]] = {}
    sop_labels: dict[str, dict[str, str]] = {}
    for meta_path in sorted((root / "units").glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{meta_path}: JSONを読めません: {exc}")
            continue
        unit_id = meta.get("unit_id")
        if not isinstance(unit_id, str) or unit_id != meta_path.parent.name:
            errors.append(f"{meta_path}: unit_idがdirectory名と一致しません")
            continue
        if unit_id in metas:
            errors.append(f"unit_idが重複しています: {unit_id}")
            continue
        metas[unit_id] = meta
        sampling = meta.get("sampling", {})
        try:
            fps, n_frames = float(sampling["fps"]), int(sampling["n_frames"])
            if fps <= 0 or n_frames <= 0:
                raise ValueError
            durations[unit_id] = n_frames / fps
        except (KeyError, TypeError, ValueError):
            errors.append(f"{unit_id}: sampling.fps/n_framesが不正です")
        ref = meta.get("sop_ref", {})
        sop_path = (meta_path.parent / ref.get("path", "")).resolve()
        try:
            sop = yaml.safe_load(sop_path.read_text(encoding="utf-8"))
            ids = [event["id"] for event in sop["events"]]
            if len(ids) != len(set(ids)):
                errors.append(f"{unit_id}: SOP event IDが重複しています")
            sop_events[unit_id] = set(ids)
            sop_labels[unit_id] = {
                str(event["id"]): str(event.get("ask", "")) for event in sop["events"]
            }
        except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
            errors.append(f"{unit_id}: SOPを読めません: {sop_path} ({exc})")
        media = meta.get("media", {})
        data_dir = repo / "data" / dataset_id / "units" / unit_id / media.get("path", "frames")
        bundled_dir = meta_path.parent / media.get("path", "frames")
        frames_dir = bundled_dir if media.get("availability") == "bundled" else data_dir
        if require_media and len(list(frames_dir.glob("f*.jpg"))) != sampling.get("n_frames"):
            errors.append(f"{unit_id}: media frame数がmetaと一致しません: {frames_dir}")

    for annotation_path in sorted((root / "annotations" / "human").glob("*.json")):
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            unit_id = annotation["unit_id"]
            if annotation_path.stem != unit_id or unit_id not in metas:
                errors.append(f"{annotation_path}: 未知または不一致のunit_idです")
                continue
            if annotation.get("interval_convention") != "half-open_seconds":
                errors.append(f"{unit_id}: interval_conventionはhalf-open_secondsが必要です")
            expected_fields = {
                "unit_id", "annotation_revision", "interval_convention", "event_labels", "events"
            }
            if set(annotation) != expected_fields:
                errors.append(f"{unit_id}: annotation fieldは{sorted(expected_fields)}だけが必要です")
            if annotation.get("event_labels") != sop_labels.get(unit_id, {}):
                errors.append(f"{unit_id}: event_labelsがSOPの日本語イベント文と一致しません")
            unknown = set(annotation.get("events", {})) - sop_events.get(unit_id, set())
            if unknown:
                errors.append(f"{unit_id}: SOPにないannotation eventがあります: {sorted(unknown)}")
            duration = durations.get(unit_id)
            for event_id, spans in annotation.get("events", {}).items():
                for span in spans or []:
                    start, end = float(span["start_s"]), float(span["end_s"])
                    if start < 0 or end <= start or (duration is not None and end > duration + 1e-6):
                        errors.append(f"{unit_id}.{event_id}: annotation区間が動画範囲外です: {span}")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{annotation_path}: annotationが不正です: {exc}")

    split_paths = sorted((root / "splits").glob("*.json"))
    if not split_paths:
        warnings.append("splitファイルがありません（demo用途以外は作成してください）")
    for split_path in split_paths:
        try:
            split = json.loads(split_path.read_text(encoding="utf-8"))
            assignments = split.get("assignments", {})
            locations: dict[str, str] = {}
            for name, units in assignments.items():
                for unit_id in units:
                    if unit_id not in metas:
                        errors.append(f"{split_path.name}: 未知のunitです: {unit_id}")
                    if unit_id in locations:
                        errors.append(f"{split_path.name}: unitが複数subsetにあります: {unit_id}")
                    locations[unit_id] = name
            group_fields = split.get("group_by", [])
            groups: dict[tuple[Any, ...], set[str]] = {}
            for unit_id, subset in locations.items():
                source = metas.get(unit_id, {}).get("source", {})
                key = tuple(source.get(field) for field in group_fields)
                if group_fields and any(value is None for value in key):
                    errors.append(f"{unit_id}: group_byに必要なsource fieldがありません: {group_fields}")
                groups.setdefault(key, set()).add(subset)
            for key, subsets in groups.items():
                if group_fields and len(subsets) > 1:
                    errors.append(f"{split_path.name}: group leakage {key}: {sorted(subsets)}")
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"{split_path}: splitが不正です: {exc}")
    return {"dataset_id": dataset_id, "unit_count": len(metas),
            "error_count": len(errors), "warning_count": len(warnings),
            "errors": errors, "warnings": warnings}


def create_split(repo: Path, dataset_id: str, split_id: str, group_by: list[str],
                 validation_ratio: float, test_ratio: float, seed: int) -> Path:
    """source groupを跨がせず、hash順で決定論的なtrain/validation/testを作る。"""
    validate_id(dataset_id, "dataset ID")
    validate_id(split_id, "split ID")
    if not group_by or any(not ID_PATTERN.fullmatch(field) for field in group_by):
        raise ValueError("--group-byはsource内のfield名を1件以上指定します")
    if validation_ratio < 0 or test_ratio < 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("validation/test ratioは0以上で、合計を1未満にします")
    root = repo.resolve() / "datasets" / dataset_id
    out = root / "splits" / f"{split_id}.json"
    if out.exists():
        raise ValueError(f"splitは既に存在します: {out}")
    groups: dict[tuple[str, ...], list[str]] = {}
    for meta_path in sorted((root / "units").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        source = meta.get("source", {})
        missing = [field for field in group_by if source.get(field) is None]
        if missing:
            raise ValueError(f"{meta.get('unit_id')}: source fieldがありません: {missing}")
        key = tuple(str(source[field]) for field in group_by)
        groups.setdefault(key, []).append(meta["unit_id"])
    if not groups:
        raise ValueError("split対象unitがありません")
    ordered = sorted(groups, key=lambda key: hashlib.sha256(
        f"{seed}:{json.dumps(key, ensure_ascii=False)}".encode()).hexdigest())
    count = len(ordered)
    n_test = round(count * test_ratio)
    n_validation = round(count * validation_ratio)
    if test_ratio > 0 and n_test == 0 and count >= 2:
        n_test = 1
    if validation_ratio > 0 and n_validation == 0 and count - n_test >= 2:
        n_validation = 1
    if n_test + n_validation >= count:
        raise ValueError("group数が少なすぎてtrainを確保できません")
    group_subset = {
        key: ("test" if index < n_test else
              "validation" if index < n_test + n_validation else "train")
        for index, key in enumerate(ordered)
    }
    assignments = {name: [] for name in ("train", "validation", "test")}
    group_doc = {}
    for key in sorted(groups):
        subset = group_subset[key]
        units = sorted(groups[key])
        assignments[subset].extend(units)
        group_doc["/".join(key)] = {"split": subset, "units": units}
    _json(out, {
        "split_id": split_id, "group_by": group_by, "seed": seed,
        "ratios": {"validation": validation_ratio, "test": test_ratio},
        "assignments": assignments, "groups": group_doc,
        "policy": {"group_disjoint": True, "create_before_test_review": True},
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(prog="sop-dataset")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="空のdatasetを作成")
    init.add_argument("--dataset", required=True)
    init.add_argument("--name", required=True)
    init.add_argument("--description", default="")
    add = sub.add_parser("add-video", help="手元の動画区間とSOPをunitとして登録")
    add.add_argument("--dataset", required=True)
    add.add_argument("--unit", required=True)
    add.add_argument("--video", type=Path, required=True)
    add.add_argument("--event", action="append", default=[], metavar="ID=QUESTION")
    add.add_argument("--fps", type=float, default=2.0)
    add.add_argument("--start", type=float, default=0.0)
    add.add_argument("--end", type=float, default=None)
    add.add_argument("--source-group", default="local")
    validate = sub.add_parser("validate", help="dataset契約とgroup leakageを検査")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--require-media", action="store_true")
    split = sub.add_parser("split", help="source group単位で決定論的にdatasetを分割")
    split.add_argument("--dataset", required=True)
    split.add_argument("--split-id", default="benchmark")
    split.add_argument("--group-by", action="append", default=None)
    split.add_argument("--validation-ratio", type=float, default=0.1)
    split.add_argument("--test-ratio", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.command == "init":
            out = init_dataset(args.repo.resolve(), args.dataset, args.name, args.description)
        elif args.command == "add-video":
            out = add_video(args.repo.resolve(), args.dataset, args.unit, args.video,
                            parse_events(args.event), args.fps, args.start, args.end,
                            args.source_group)
        elif args.command == "validate":
            result = validate_dataset(args.repo, args.dataset, require_media=args.require_media)
            for warning in result["warnings"]:
                print(f"WARNING: {warning}")
            for error in result["errors"]:
                print(f"ERROR: {error}")
            print(f"[sop-dataset] {result['unit_count']} units, "
                  f"{result['error_count']} errors, {result['warning_count']} warnings")
            if result["errors"]:
                raise SystemExit(1)
            return
        else:
            out = create_split(args.repo, args.dataset, args.split_id,
                               args.group_by or ["source_group"],
                               args.validation_ratio, args.test_ratio, args.seed)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"[sop-dataset] {exc}") from exc
    print(f"[sop-dataset] created: {out}")


if __name__ == "__main__":
    main()
