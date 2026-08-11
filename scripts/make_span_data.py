"""スパンSFTデータ生成: 過去はテキスト・直近は動画 → 30秒スパンのイベント列。

スパン [t-30, t) のうち [t-30, t-10) は既検出イベントのテキスト(スパン座標0-20s)、
[t-10, t) は動画フレーム(既定: 2秒間隔の5枚。336^2 と合わせ系列約1,300=メモリ実績域)。
出力はスパン全体(0-30s)のイベント列。
学習させる本質は「接合」: テキスト最後のイベントが動画内で継続しているか、
切り替わったかを判定し、開始時刻(テキスト由来)と現在(映像由来)を統合する。

- 履歴ノイズ(学習のみ): 境界±2秒のジッタ。推論時は自分の過去出力が履歴に
  なるため、正確すぎるGT履歴だけで学習しない
- カテゴリ: seam_switch(動画内で新イベント開始) / video_multi(動画内2件以上) /
  seam_continue(テキストのイベントが継続するだけ) / video_empty
- 評価セット: valid動画・動画部分が非重複(stride=10)・ノイズなし=オラクル履歴条件

使い方: python scripts/make_span_data.py [--span 30] [--video 10] [--max-train 900]
出力: data/goalstep/sft_span/{train,validation_eval}.jsonl + meta.json
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtva.goalstep import ROOT, build_timeline, frame_path, load_annotations, load_manifest, vocab_for
from rtva.prompts import system_prompt_span, user_prompt_span
from rtva.windows import clip_events


def categorize_span(gold: list[dict], text_s: int) -> str:
    video_ev = [e for e in gold if e["end_s"] > text_s]
    if len(video_ev) >= 2:
        return "video_multi"
    if any(e["start_s"] >= text_s for e in video_ev):
        return "seam_switch"
    if video_ev:
        return "seam_continue"
    return "video_empty"


def history_of(gold: list[dict], text_s: int, rng: random.Random | None) -> list[dict]:
    """スパン前半のテキスト履歴。学習時は境界に±2秒のジッタを入れる。"""
    hist = []
    for e in gold:
        if e["start_s"] >= text_s:
            continue
        h = {"start_s": e["start_s"], "end_s": min(e["end_s"], text_s), "step": e["step"]}
        if rng is not None and rng.random() < 0.5:
            h["start_s"] = max(0, h["start_s"] + rng.choice([-2, -1, 1, 2]))
            if h["end_s"] < text_s:  # 継続中(=text_s終端)の見かけの端はジッタしない
                h["end_s"] = min(text_s, max(h["start_s"] + 1, h["end_s"] + rng.choice([-2, -1, 1, 2])))
        hist.append(h)
    return hist


def before_label(tl, s0: int) -> str | None:
    for t in range(s0 - 1, max(-1, s0 - 31), -1):
        if 0 <= t < len(tl) and tl[t]:
            return tl[t]
    return None


def make_record(uid, tl, t, span, video_s, frame_stride, sys_p, rng):
    s0 = t - span
    text_s = span - video_s
    gold = clip_events(tl, s0, span)
    user = user_prompt_span(history_of(gold, text_s, rng), before_label(tl, s0), video_s)
    frames = [str(frame_path(uid, tt).relative_to(ROOT))
              for tt in range(t - video_s, t, frame_stride)]
    return {
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps({"events": gold}, ensure_ascii=False)},
        ],
        "images": frames,
        "uid": uid, "w_start": s0, "category": categorize_span(gold, text_s),
    }


def subsample_evenly(items: list, n: int) -> list:
    if n >= len(items):
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--span", type=int, default=30)
    ap.add_argument("--video", type=int, default=10)
    ap.add_argument("--frame-stride", type=int, default=2,
                    help="動画部分のフレーム間隔(秒)。2なら10秒→5フレーム。"
                         "メモリ実績域(系列約1,400)に収めるための主ノブ")
    ap.add_argument("--train-stride", type=int, default=5)
    ap.add_argument("--plain-ratio", type=float, default=0.6,
                    help="seam_continue/video_empty を境界系の何倍まで採るか")
    ap.add_argument("--extra-train-uids", default="data/goalstep/v2_extra_train_uids.txt")
    ap.add_argument("--max-train", type=int, default=900)
    ap.add_argument("--out-dir", default="data/goalstep/sft_span")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    span, video_s = args.span, args.video

    ann = load_annotations()
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    extra = [l.strip() for l in open(args.extra_train_uids) if l.strip()] \
        if args.extra_train_uids and Path(args.extra_train_uids).exists() else []

    for split, fname in (("train", "train.jsonl"), ("valid", "validation_eval.jsonl")):
        rng = random.Random(args.seed) if split == "train" else None
        stride = args.train_stride if split == "train" else video_s
        uids = [v["uid"] for v in load_manifest() if v["split"] == split]
        if split == "train":
            uids += extra
        boundary_like, plain = [], []
        for uid in uids:
            tl = build_timeline(ann[uid])
            sys_p = system_prompt_span(vocab_for(ann[uid]), span, video_s,
                                       n_frames=video_s // args.frame_stride)
            for t in range(span, len(tl) + 1, stride):
                r = make_record(uid, tl, t, span, video_s, args.frame_stride, sys_p, rng)
                (boundary_like if r["category"] in ("seam_switch", "video_multi") else plain).append(r)
        if split == "train":
            keep = boundary_like + subsample_evenly(plain, int(args.plain_ratio * len(boundary_like)))
            keep.sort(key=lambda r: (r["uid"], r["w_start"]))
            if args.max_train and len(keep) > args.max_train:
                keep = subsample_evenly(keep, args.max_train)
        else:
            keep = boundary_like + plain
            keep.sort(key=lambda r: (r["uid"], r["w_start"]))
        with (out / fname).open("w") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(keep)}スパン {dict(Counter(r['category'] for r in keep))}")

    (out / "meta.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
