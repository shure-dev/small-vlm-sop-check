#!/usr/bin/env python3
"""annotated-egocentric-10k から Factory Ego unit候補を層化サンプリングする。

fit-alessandro-berti/annotated-egocentric-10k-dataset のローカルcloneを入力に、
process_mining_event_logs のCSVからクリップごとの作業種類(process)を集計し、
作業種類→工場→worker の優先度で決定論的に N クリップを選ぶ。
各クリップでは「10秒間に3〜6個のイベント開始が入る遷移の多い区間」を
窓候補としてスコアリングし、最良の10秒窓を採用する。

設計方針:
- 乱数を使わない決定論的選択(ソート順のみ)。再実行しても同じ結果になる。
- 追記型: 既存 units/*/meta.json の clip_id は母集団から除外されるため、
  データを増やすときは --n を増やして再実行すれば既存選定は変わらない。
- 上流タイムスタンプはLLM生成のため±数秒の誤差前提。窓は「候補」であり、
  イベント定義とアノテーションは実フレームを正とする。

使い方:
  python3 tools/benchmark/sample_units.py --annotations-root <anno10kのclone> \
      [--n 20] [--out plan.json] [--apply]

--apply で datasets/factory_ego/units/<unit_id>/meta.json を書き出す。
既定はdry-run(選定結果の表示のみ)。フレーム抽出は fetch_factory_ego.py が担う。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "datasets" / "factory_ego"
UPSTREAM_DATASET = "builddotai/Egocentric-10K"
ANNOTATIONS_REPO = "fit-alessandro-berti/annotated-egocentric-10k-dataset"
EPOCH = datetime(2000, 1, 1)
WINDOW_SECONDS = 10
MIN_EVENTS, MAX_EVENTS = 3, 6
WORKER_CAP = 2
FACTORY_MIN = 2
SLUG_STOPWORDS = {"and", "of", "the", "a", "an", "to"}
CLIP_RE = re.compile(r"^factory(\d{3})_worker(\d{3})_(\d{5})$")
SEGMENT_RE = re.compile(
    r"\[(\d{2}):(\d{2}):(\d{2})\s*-\s*(\d{2}):(\d{2}):(\d{2})\]", re.M
)


def event_seconds(ts: str) -> float:
    return (datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") - EPOCH).total_seconds()


@dataclass
class Window:
    start: int
    n_events: int
    n_kinds: int
    activities: list[str]

    @property
    def end(self) -> int:  # フレーム両端含む: start..end で n_frames 秒
        return self.start + WINDOW_SECONDS - 1

    @property
    def score(self) -> tuple:
        return (self.n_kinds, min(self.n_events, 5), -self.start)


@dataclass
class Candidate:
    clip_id: str
    factory: str
    worker: str
    main_process: str
    processes: list[str]
    duration: float
    window: Window
    transcript_excerpt: list[str] = field(default_factory=list)
    unit_id: str = ""


def load_events(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                {
                    "start": event_seconds(row["start_timestamp"]),
                    "end": event_seconds(row["end_timestamp"]),
                    "activity": row["activity"],
                    "process": row["process"],
                }
            )
    return rows


def best_window(rows: list[dict], duration: float) -> Window | None:
    best: Window | None = None
    for anchor in sorted({int(r["start"]) for r in rows}):
        if anchor + WINDOW_SECONDS > duration:
            continue
        inside = [r for r in rows if anchor <= r["start"] < anchor + WINDOW_SECONDS]
        n_events = len(inside)
        if not (MIN_EVENTS <= n_events <= MAX_EVENTS):
            continue
        window = Window(
            start=anchor,
            n_events=n_events,
            n_kinds=len({r["activity"] for r in inside}),
            activities=[r["activity"] for r in inside],
        )
        if best is None or window.score > best.score:
            best = window
    return best


def collect_candidates(logs_root: Path, exclude_clips: set[str]) -> list[Candidate]:
    candidates = []
    for csv_path in sorted(logs_root.glob("factory_*/worker_*/*.csv")):
        clip_id = csv_path.stem
        if clip_id in exclude_clips or not CLIP_RE.match(clip_id):
            continue
        rows = load_events(csv_path)
        if not rows:
            continue
        duration = max(r["end"] for r in rows)
        window = best_window(rows, duration)
        if window is None:
            continue
        processes = Counter(r["process"] for r in rows)
        candidates.append(
            Candidate(
                clip_id=clip_id,
                factory=csv_path.parts[-3],
                worker=csv_path.parts[-2],
                main_process=processes.most_common(1)[0][0],
                processes=[p for p, _ in processes.most_common()],
                duration=duration,
                window=window,
            )
        )
    return candidates


def select(candidates: list[Candidate], n: int, seen_processes: set[str]) -> list[Candidate]:
    """作業種類→工場→workerの優先度で層化し、決定論的にn件選ぶ。"""
    chosen: list[Candidate] = []
    used_processes = set(seen_processes)
    worker_counts: Counter = Counter()
    remaining = sorted(
        candidates, key=lambda c: (c.window.score, c.clip_id), reverse=True
    )

    def eligible(cand: Candidate, require_new_process: bool) -> bool:
        if any(c.clip_id == cand.clip_id for c in chosen):
            return False
        if worker_counts[(cand.factory, cand.worker)] >= WORKER_CAP:
            return False
        if require_new_process and cand.main_process in used_processes:
            return False
        return True

    def take(cand: Candidate) -> None:
        chosen.append(cand)
        used_processes.add(cand.main_process)
        worker_counts[(cand.factory, cand.worker)] += 1

    factories = sorted({c.factory for c in candidates})
    # Phase 1: 各工場から最低FACTORY_MIN件(未使用のprocessを優先)
    for _round in range(FACTORY_MIN):
        for factory in factories:
            if len(chosen) >= n:
                break
            pool = [c for c in remaining if c.factory == factory]
            pick = next((c for c in pool if eligible(c, True)), None) or next(
                (c for c in pool if eligible(c, False)), None
            )
            if pick:
                take(pick)
    # Phase 2: 残りをグローバルに(未使用のprocess優先→スコア順)
    while len(chosen) < n:
        pick = next((c for c in remaining if eligible(c, True)), None) or next(
            (c for c in remaining if eligible(c, False)), None
        )
        if pick is None:
            break
        take(pick)
    return chosen


def parse_transcript(text: str) -> list[tuple[int, int, str]]:
    segments = []
    matches = list(SEGMENT_RE.finditer(text))
    for i, m in enumerate(matches):
        h1, m1, s1, h2, m2, s2 = (int(g) for g in m.groups())
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():body_end].strip()
        segments.append((h1 * 3600 + m1 * 60 + s1, h2 * 3600 + m2 * 60 + s2, body))
    return segments


def attach_excerpts(cands: list[Candidate], transcripts_root: Path) -> None:
    for cand in cands:
        txt_path = (
            transcripts_root / cand.factory / cand.worker / f"{cand.clip_id}.txt"
        )
        if not txt_path.exists():
            continue
        w0, w1 = cand.window.start - 5, cand.window.start + WINDOW_SECONDS + 5
        for seg_start, seg_end, body in parse_transcript(
            txt_path.read_text(encoding="utf-8")
        ):
            if seg_end >= w0 and seg_start <= w1:
                stamp = f"[{seg_start}s-{seg_end}s]"
                cand.transcript_excerpt.append(f"{stamp} {body}")


def make_slug(process: str, used: set[str]) -> str:
    words = [
        w
        for w in re.sub(r"[^a-z0-9 ]", " ", process.lower()).split()
        if w not in SLUG_STOPWORDS
    ]
    slug = "_".join(words[:2]) or "work"
    base, serial = slug, 2
    while slug in used:
        slug = f"{base}_{serial}"
        serial += 1
    used.add(slug)
    return slug


def assign_unit_ids(cands: list[Candidate], existing_ids: set[str]) -> None:
    used_slugs = {
        uid.split("_", 2)[2] for uid in existing_ids if uid.count("_") >= 2
    }
    for cand in cands:
        m = CLIP_RE.match(cand.clip_id)
        assert m is not None
        slug = make_slug(cand.main_process, used_slugs)
        cand.unit_id = f"f{m.group(1)}_w{m.group(2)}_{slug}"


def annotations_revision(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_meta(cand: Candidate, revision: str) -> dict:
    start, end = cand.window.start, cand.window.end
    n_frames = WINDOW_SECONDS
    return {
        "schema_version": "1.0",
        "unit_id": cand.unit_id,
        "dataset_id": "factory_ego",
        "benchmark_status": "dev_seen",
        "ground_truth": {"available": False, "required_source": "human"},
        "source": {
            "dataset": UPSTREAM_DATASET,
            "factory_id": cand.factory,
            "worker_id": cand.worker,
            "clip_id": cand.clip_id,
            "start_second": start,
            "end_second": end,
        },
        "sampling": {
            "fps": 1.0,
            "n_frames": n_frames,
            "output_naming": f"f0000.jpg...f{n_frames - 1:04d}.jpg",
            "source_frame_start": start,
            "source_frame_end": end,
        },
        "media": {
            "availability": "local_gated_source_not_committed",
            "path": "frames",
            "sha256_manifest": "frames.sha256.json",
        },
        "sop_ref": {
            "id": cand.unit_id,
            "path": f"../../sops/{cand.unit_id}/v001.yaml",
            "version": "v001",
            "status": "provisional",
        },
        "selection": {
            "method": "tools/benchmark/sample_units.py (deterministic stratified)",
            "annotations_repo": ANNOTATIONS_REPO,
            "annotations_revision": revision,
            "main_process": cand.main_process,
            "window_event_starts": cand.window.n_events,
            "window_activities": cand.window.activities,
            "transcript_excerpt": cand.transcript_excerpt,
            "caveat": "window boundaries derive from LLM-generated timestamps; frames are canonical",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-root",
        type=Path,
        required=True,
        help="annotated-egocentric-10k-dataset のローカルclone",
    )
    parser.add_argument("--n", type=int, default=20, help="選出するunit数")
    parser.add_argument("--out", type=Path, help="選定計画JSONの出力先")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="units/<unit_id>/meta.json を書き出す(既定はdry-run)",
    )
    args = parser.parse_args()

    logs_root = args.annotations_root / "process_mining_event_logs"
    transcripts_root = args.annotations_root / "raw_transcriptions"
    if not logs_root.is_dir():
        parser.error(f"event logs not found: {logs_root}")

    existing_clips: set[str] = set()
    existing_ids: set[str] = set()
    seen_processes: set[str] = set()
    for meta_path in sorted(DATASET_ROOT.glob("units/*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        existing_clips.add(meta["source"]["clip_id"])
        existing_ids.add(meta["unit_id"])
        seen_processes.add(meta.get("selection", {}).get("main_process", ""))

    candidates = collect_candidates(logs_root, existing_clips)
    chosen = select(candidates, args.n, seen_processes - {""})
    attach_excerpts(chosen, transcripts_root)
    assign_unit_ids(chosen, existing_ids)
    revision = annotations_revision(args.annotations_root)

    metas = [build_meta(c, revision) for c in chosen]
    print(f"candidates: {len(candidates)} clips (excluded {len(existing_clips)} existing)")
    print(f"selected: {len(chosen)} / requested {args.n}")
    for meta in metas:
        sel, src = meta["selection"], meta["source"]
        print(
            f"  {meta['unit_id']:40s} {src['clip_id']}"
            f"  [{src['start_second']:4d}s-{src['end_second']:4d}s]"
            f"  {sel['window_event_starts']} events  {sel['main_process']}"
        )
    if len(chosen) < args.n:
        print(f"WARNING: only {len(chosen)} candidates satisfied the constraints")

    if args.out:
        args.out.write_text(
            json.dumps(metas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"plan written: {args.out}")
    if args.apply:
        for meta in metas:
            unit_dir = DATASET_ROOT / "units" / meta["unit_id"]
            unit_dir.mkdir(parents=True, exist_ok=False)
            (unit_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(f"meta.json written for {len(metas)} units")
    else:
        print("dry-run: pass --apply to write units/<unit_id>/meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
