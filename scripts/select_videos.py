"""学習・評価に使う動画を選定し data/manifest.json を生成する。

選定基準: 料理系 / 5〜20分 / step 8以上 / カバレッジ0.85+ / overlapほぼ無し /
単一ソース動画(非grp) / ユニーク語彙6種以上。
「同時アクティブは1ステップ」の前提が成り立つクリーンな動画だけを使う。

使い方: python scripts/select_videos.py [--train 20] [--valid 5]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtva.goalstep import MANIFEST, ROOT, load_split, video_stats

CRITERIA = {
    "goal_prefix": "COOKING",
    "duration_s": [300, 1200],
    "min_steps": 8,
    "min_coverage": 0.85,
    "max_overlap_s": 1.0,
    "min_vocab": 6,
    "exclude_groups": True,
}


def eligible(s: dict) -> bool:
    vocab = {st["label"] for st in s["steps"]}
    return (
        s["goal"].startswith(CRITERIA["goal_prefix"])
        and CRITERIA["duration_s"][0] <= s["duration"] <= CRITERIA["duration_s"][1]
        and s["n_steps"] >= CRITERIA["min_steps"]
        and s["coverage"] >= CRITERIA["min_coverage"]
        and s["overlap_s"] <= CRITERIA["max_overlap_s"]
        and len(vocab) >= CRITERIA["min_vocab"]
        and not s["uid"].startswith("grp-")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=20)
    ap.add_argument("--valid", type=int, default=5)
    args = ap.parse_args()

    selected = []
    for split, n in (("train", args.train), ("valid", args.valid)):
        pool = sorted(
            (s for s in (video_stats(v) for v in load_split(split)) if eligible(s)),
            key=lambda s: -s["coverage"],
        )
        for s in pool[:n]:
            selected.append({
                "uid": s["uid"], "split": split,
                "duration_s": round(s["duration"], 1),
                "goal": s["goal"], "n_steps": s["n_steps"],
                "n_vocab": len({st["label"] for st in s["steps"]}),
                "coverage": round(s["coverage"], 3),
            })

    manifest = {"dataset": "ego4d_goalstep", "selection_criteria": CRITERIA, "videos": selected}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    total = sum(v["duration_s"] for v in selected) / 60
    print(f"{MANIFEST}: {len(selected)}本 (計 {total:.0f}分)")


if __name__ == "__main__":
    main()
