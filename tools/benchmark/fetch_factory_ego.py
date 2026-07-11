#!/usr/bin/env python3
"""Egocentric-10K -> Factory Ego local media, reproducibly.

Gated upstream(builddotai/Egocentric-10K)から必要なclipだけを取得し、
unit meta.json のsampling条件で1fpsフレームを抽出して
``datasets/factory_ego/units/<unit>/frames/`` を再構成する。

The command is intentionally safe by default: without ``--apply`` it
extracts into a work directory, compares SHA-256 against each unit's
``frames.sha256.json`` and prints a report — the repository is not touched.
``--apply`` writes frames only when they match the manifest (or the unit has
no manifest yet). A differing frame is never overwritten unless
``--update-manifest`` is also given, which rewrites ``frames.sha256.json``
and ``manifest.lock.json`` to adopt the newly extracted bytes as canonical.
Historical prediction runs keep their own ``inputs.lock.json`` untouched —
they record what those runs actually saw.

前提:
- Hugging Faceで builddotai/Egocentric-10K のgated accessに同意済みであること
  (`hf auth login` 済み、または HF_TOKEN 環境変数)。
- 依存: huggingface_hub, opencv-python。

使い方:
  python3 tools/benchmark/fetch_factory_ego.py                 # dry-run(取得+照合のみ)
  python3 tools/benchmark/fetch_factory_ego.py --apply         # 一致フレームを配置
  python3 tools/benchmark/fetch_factory_ego.py --apply --update-manifest  # 不一致を正として採用
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

REPO_ID = "builddotai/Egocentric-10K"
# 上流のtar配置: <factory_xxx>/workers/<worker_xxx>/factoryXXX_workerXXX_partNN.tar
# 対象factory/workerはunitのmeta.json(source)から導出し、part数は上流をlsして数える。
CLIP_ID_RE = re.compile(r"^factory(\d{3})_worker(\d{3})_\d{5}$")
SCHEMA_VERSION = "1.0"
BLOCK = 512  # tar block size
# 既存manifestとバイト一致する値を実測で逆算した(2026-07、q85で160/160一致)。
# cv2既定の95ではないので変更しないこと。
JPEG_QUALITY = 85


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class UnitPlan:
    unit_id: str
    clip_id: str
    start_second: int
    end_second: int
    n_frames: int
    frames_dir: Path
    manifest_path: Path
    meta_path: Path
    sop_path: Path


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_units(dataset_root: Path) -> list[UnitPlan]:
    plans = []
    for meta_path in sorted(dataset_root.glob("units/*/meta.json")):
        meta = load_json(meta_path)
        source = meta["source"]
        sampling = meta["sampling"]
        unit_dir = meta_path.parent
        plans.append(UnitPlan(
            unit_id=meta["unit_id"],
            clip_id=source["clip_id"],
            start_second=source["start_second"],
            end_second=source["end_second"],
            n_frames=sampling["n_frames"],
            frames_dir=unit_dir / meta["media"]["path"],
            manifest_path=unit_dir / meta["media"]["sha256_manifest"],
            meta_path=meta_path,
            sop_path=(unit_dir / meta["sop_ref"]["path"]).resolve(),
        ))
    if not plans:
        raise FetchError(f"no unit meta.json found under {dataset_root}/units")
    return plans


# --- upstream tar streaming -------------------------------------------------

def iter_tar_members(fileobj: BinaryIO) -> Iterator[tuple[str, int, int]]:
    """Yield (name, data_offset, size) by walking tar headers only.

    通常のtarfile.getmembers()と違いデータ本体をシークで飛ばすため、
    HfFileSystem越しでもヘッダ分(512B/entry近傍)の転送で済む。
    """
    offset = 0
    while True:
        fileobj.seek(offset)
        header = fileobj.read(BLOCK)
        if len(header) < BLOCK or header == b"\0" * BLOCK:
            return
        name = header[0:100].split(b"\0", 1)[0].decode("utf-8", "replace")
        size = int(header[124:136].split(b"\0", 1)[0].strip() or b"0", 8)
        typeflag = header[156:157]
        if typeflag in (b"L", b"K"):  # GNU longname/longlink: 実名は次エントリ
            fileobj.seek(offset + BLOCK)
            name = fileobj.read(size).split(b"\0", 1)[0].decode("utf-8", "replace")
            offset += BLOCK + ((size + BLOCK - 1) // BLOCK) * BLOCK
            fileobj.seek(offset)
            header = fileobj.read(BLOCK)
            if len(header) < BLOCK:
                return
            size = int(header[124:136].split(b"\0", 1)[0].strip() or b"0", 8)
        yield name, offset + BLOCK, size
        offset += BLOCK + ((size + BLOCK - 1) // BLOCK) * BLOCK


def copy_members(fh: BinaryIO, remaining: set[str], work_dir: Path) -> dict[str, Path]:
    """fh(tar)を走査し、stemがremainingに入るmp4をwork_dirへ書き出して回収する。"""
    found: dict[str, Path] = {}
    for name, data_offset, size in iter_tar_members(fh):
        stem = Path(name).stem
        if stem not in remaining:
            continue
        out = work_dir / f"{stem}.mp4"
        print(f"    found {name} ({size / 1e6:.1f} MB) -> {out}", flush=True)
        fh.seek(data_offset)
        left = size
        with out.open("wb") as sink:
            while left > 0:
                chunk = fh.read(min(1 << 22, left))
                if not chunk:
                    raise FetchError(f"truncated tar entry: {name}")
                sink.write(chunk)
                left -= len(chunk)
        remaining.discard(stem)
        found[stem] = out
        if not remaining:
            break
    return found


def clip_source_dir(clip_id: str) -> tuple[str, str]:
    """clip_id から上流の (factory_xxx, worker_xxx) ディレクトリ名を導出する。"""
    match = CLIP_ID_RE.match(clip_id)
    if match is None:
        raise FetchError(f"unrecognized clip_id: {clip_id}")
    return f"factory_{match.group(1)}", f"worker_{match.group(2)}"


def fetch_clips(clip_ids: set[str], work_dir: Path, revision: str) -> dict[str, Path]:
    """必要なclipのmp4をwork_dirへ取得し {clip_id: path} を返す。取得済みは再利用。"""
    remaining = {c for c in clip_ids if not (work_dir / f"{c}.mp4").exists()}
    found = {c: work_dir / f"{c}.mp4" for c in clip_ids - remaining}
    for clip_id, path in sorted(found.items()):
        print(f"  cached: {clip_id} ({path.stat().st_size / 1e6:.1f} MB)")
    if not remaining:
        return found

    try:
        from huggingface_hub import HfFileSystem
    except ImportError as exc:
        raise FetchError("huggingface_hub が必要です: pip install huggingface_hub") from exc

    fs = HfFileSystem()
    work_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], set[str]] = {}
    for clip_id in remaining:
        groups.setdefault(clip_source_dir(clip_id), set()).add(clip_id)
    for (factory, worker), wanted in sorted(groups.items()):
        worker_dir = f"datasets/{REPO_ID}@{revision}/{factory}/workers/{worker}"
        tar_paths = sorted(p for p in fs.ls(worker_dir, detail=False) if p.endswith(".tar"))
        if not tar_paths:
            raise FetchError(f"no tar parts under upstream {factory}/workers/{worker}")
        for tar_path in tar_paths:
            if not wanted:
                break
            print(f"  scanning {factory}/{worker}/{Path(tar_path).name} ...", flush=True)
            with fs.open(tar_path, "rb", block_size=1 << 20, cache_type="bytes") as fh:
                found.update(copy_members(fh, wanted, work_dir))
        if wanted:
            raise FetchError(f"clips not found in upstream tars: {sorted(wanted)}")
    return found


# --- frame extraction --------------------------------------------------------

def extract_unit_frames(clip_path: Path, plan: UnitPlan, out_dir: Path) -> dict[str, bytes]:
    """秒 start..end(両端含む)ごとに round(t*fps) フレーム目をフル解像度JPEGで書き出す。"""
    try:
        import cv2
    except ImportError as exc:
        raise FetchError("opencv-python が必要です: pip install opencv-python") from exc

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise FetchError(f"動画を開けませんでした: {clip_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, bytes] = {}
    seconds = range(plan.start_second, plan.end_second + 1)
    if len(seconds) != plan.n_frames:
        raise FetchError(f"{plan.unit_id}: meta sampling mismatch "
                         f"({len(seconds)} seconds vs n_frames={plan.n_frames})")
    for idx, second in enumerate(seconds):
        frame_no = min(int(round(second * video_fps)), total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            raise FetchError(f"{plan.unit_id}: フレーム読み出し失敗 (second={second})")
        name = f"f{idx:04d}.jpg"
        path = out_dir / name
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
            raise FetchError(f"JPEG書き出し失敗: {path}")
        frames[name] = path.read_bytes()
    cap.release()
    return frames


# --- compare / apply ---------------------------------------------------------

def compare_unit(plan: UnitPlan, frames: dict[str, bytes]) -> tuple[dict[str, str], list[str]]:
    """抽出結果のSHAとmanifestを突き合わせ、(新manifest, 不一致フレーム名) を返す。"""
    new_manifest = {name: sha256_bytes(data) for name, data in sorted(frames.items())}
    if not plan.manifest_path.is_file():
        return new_manifest, sorted(new_manifest)
    old_manifest = load_json(plan.manifest_path)
    mismatched = sorted(set(old_manifest) ^ set(new_manifest)
                        | {n for n in old_manifest.keys() & new_manifest.keys()
                           if old_manifest[n] != new_manifest[n]})
    return new_manifest, mismatched


def update_manifest_lock(dataset_root: Path, plan: UnitPlan) -> None:
    """unitのlockエントリを現ファイルから再計算する(新規unitはエントリを作成)。"""
    lock_path = dataset_root / "manifest.lock.json"
    lock = load_json(lock_path)
    manifest = load_json(plan.manifest_path)
    lock.setdefault("units", {})[plan.unit_id] = {
        "frame_count": len(manifest),
        "frames_manifest_sha256": sha256_file(plan.manifest_path),
        "meta_sha256": sha256_file(plan.meta_path),
        "sop_sha256": sha256_file(plan.sop_path),
    }
    lock_path.write_text(dump_json(lock), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="clip mp4と抽出結果の置き場(既定: <repo>/out/factory_ego_fetch)")
    parser.add_argument("--revision", default="main", help="upstream dataset revision")
    parser.add_argument("--unit", action="append", default=None,
                        help="対象unit id(複数可)。既定は全unit")
    parser.add_argument("--apply", action="store_true",
                        help="照合済みフレームを datasets/ 配下へ書き込む(既定はdry-run)")
    parser.add_argument("--update-manifest", action="store_true",
                        help="不一致時に frames.sha256.json と manifest.lock.json を再生成して"
                             "抽出結果を正として採用する(要 --apply)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.update_manifest and not args.apply:
        print("error: --update-manifest には --apply が必要です", file=sys.stderr)
        return 2
    dataset_root = args.repo / "datasets" / "factory_ego"
    work_dir = args.work_dir or (args.repo / "out" / "factory_ego_fetch")

    plans = load_units(dataset_root)
    if args.unit:
        wanted = set(args.unit)
        unknown = wanted - {p.unit_id for p in plans}
        if unknown:
            print(f"error: unknown unit: {sorted(unknown)}", file=sys.stderr)
            return 2
        plans = [p for p in plans if p.unit_id in wanted]

    print(f"units: {len(plans)}, clips: {sorted({p.clip_id for p in plans})}")
    print(f"work dir: {work_dir}")
    try:
        clips = fetch_clips({p.clip_id for p in plans}, work_dir / "clips", args.revision)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    all_match = True
    adopted = 0
    for plan in plans:
        frames = extract_unit_frames(clips[plan.clip_id], plan,
                                     work_dir / "extracted" / plan.unit_id)
        new_manifest, mismatched = compare_unit(plan, frames)
        status = "match" if not mismatched else f"MISMATCH({len(mismatched)}/{len(new_manifest)})"
        print(f"  {plan.unit_id}: {status}")
        if mismatched and not args.update_manifest:
            all_match = False
            continue
        if args.apply:
            plan.frames_dir.mkdir(parents=True, exist_ok=True)
            for name, data in frames.items():
                (plan.frames_dir / name).write_bytes(data)
            if mismatched and args.update_manifest:
                plan.manifest_path.write_text(dump_json(new_manifest), encoding="utf-8")
                update_manifest_lock(dataset_root, plan)
                adopted += 1

    if args.apply:
        print(f"done: frames written{f', manifests regenerated: {adopted} unit(s)' if adopted else ''}")
        print("次: python3 tools/benchmark/validate.py --require-media で検証してください")
    elif not all_match:
        print("dry-run: 不一致あり。採用するには --apply --update-manifest を付けてください")
        return 1
    else:
        print("dry-run: 全フレームがmanifestと一致(--apply で配置できます)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
