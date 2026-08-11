"""Ego4D GoalStep データセットアダプタ。

アノテーション読み込み・1秒解像度タイムライン・動画統計。
別データセットを使う場合はこのモジュールと同じインタフェースを実装する。
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANN_DIR = ROOT / "data" / "goalstep" / "annotations"
VIDEO_DIR = ROOT / "data" / "goalstep" / "v2" / "video_540ss"
FRAME_DIR = ROOT / "data" / "goalstep" / "frames"


def normalize_label(s: str) -> str:
    return " ".join(s.strip().lower().rstrip(".").split())


_ARTICLES = {"a", "an", "the"}


def _label_key(label: str) -> str:
    return " ".join(t for t in label.split() if t not in _ARTICLES)


def canonical_labels(video: dict) -> dict[str, str]:
    """動画内ラベルの表記揺れを最頻表記へ統合する対応表。

    GoalStep には同一動画内に冠詞だけ違うラベルが共存する
    (例: "stir ingredient in a pan" / "stir ingredient in pan")。
    冠詞を除いたトークン列が一致するものを1つの表記(最頻・同数なら短い方)に寄せる。
    意味が異なりうる語の差(ingredient/recipe 等)は統合しない。
    """
    from collections import Counter, defaultdict

    raw = [normalize_label(s["step_description"]) for s in video.get("segments") or []]
    counts = Counter(raw)
    groups: dict[str, list[str]] = defaultdict(list)
    for lab in counts:
        groups[_label_key(lab)].append(lab)
    out: dict[str, str] = {}
    for labs in groups.values():
        canon = sorted(labs, key=lambda l: (-counts[l], len(l), l))[0]
        for l in labs:
            out[l] = canon
    return out


def load_split(name: str) -> list[dict]:
    return json.loads((ANN_DIR / f"goalstep_{name}.json").read_text())["videos"]


def load_annotations() -> dict[str, dict]:
    """train+valid の全動画を uid -> video dict で返す。"""
    return {v["video_uid"]: v for split in ("train", "valid") for v in load_split(split)}


MANIFEST = ROOT / "data" / "manifest.json"


def load_manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text())["videos"]


def steps_for(video: dict) -> list[dict]:
    """stepレベルの区間を開始時刻順・正規化+表記揺れ統合済みラベルで返す。"""
    canon = canonical_labels(video)
    steps = [
        {
            "start": float(s["start_time"]),
            "end": float(s["end_time"]),
            "label": canon[normalize_label(s["step_description"])],
        }
        for s in video.get("segments") or []
    ]
    return sorted(steps, key=lambda s: s["start"])


def duration_of(video: dict) -> int:
    return int(float(video["end_time"]) - float(video["start_time"]))


def vocab_for(video: dict) -> list[str]:
    return sorted({s["label"] for s in steps_for(video)})


def build_timeline(video: dict) -> list[str | None]:
    """label(t) (t=0..T-1秒)。時刻 t を含む最初(開始が早い)の step のラベル。"""
    T = duration_of(video)
    tl: list[str | None] = [None] * T
    for s in steps_for(video):
        for t in range(max(0, int(s["start"])), min(T, int(s["end"]) + 1)):
            if tl[t] is None and s["start"] <= t < s["end"]:
                tl[t] = s["label"]
    return tl


def frame_path(uid: str, t: int) -> Path:
    return FRAME_DIR / uid / f"f{t:05d}.jpg"


def video_stats(video: dict) -> dict:
    """選定・分析用の動画統計(時間・step数・カバレッジ・overlap)。"""
    dur = float(video["end_time"]) - float(video["start_time"])
    steps = steps_for(video)
    covered = sum(s["end"] - s["start"] for s in steps)
    overlap = 0.0
    for prev, cur in zip(steps, steps[1:]):
        if cur["start"] < prev["end"]:
            overlap += prev["end"] - cur["start"]
    return {
        "uid": video["video_uid"],
        "duration": dur,
        "goal": video["goal_category"],
        "steps": [
            {"start": s["start"], "end": s["end"], "description": s["label"], "label": s["label"]}
            for s in steps
        ],
        "n_steps": len(steps),
        "coverage": covered / dur if dur > 0 else 0.0,
        "overlap_s": overlap,
    }
