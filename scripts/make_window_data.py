"""窓ベースSFTデータ生成: W秒の動画窓 → タイムスタンプ付きイベント区間リスト。

出力契約(rtva.prompts と同一):
  {"events": [{"start_s": 0, "end_s": 3, "step": "toast bread"},
              {"start_s": 3, "end_s": 5, "step": "crack eggs"}]}

データ設計:
- 窓の種類(境界あり/複数遷移/単一継続/無ラベル混在)を分類し、構成を表示する
- 境界を含む窓は --boundary-stride でスライドさせて増強(遷移が窓内の様々な位置に来る)
- 境界を含まない窓(単一継続)は境界窓数 × --plain-ratio を等間隔サンプル
- 評価セット(validation_eval.jsonl)は増強なし・非重複・全窓

使い方:
    python scripts/make_window_data.py [--window 5] [--boundary-stride 3]
                                       [--plain-ratio 0.6] [--with-history]
出力: data/goalstep/sft_window/{train,validation,validation_eval}.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtva.goalstep import ROOT, build_timeline, frame_path, load_annotations, load_manifest, vocab_for
from rtva.prompts import USER_PLAIN, system_prompt
from rtva.windows import categorize, clip_events, history_text

OUT_DIR = ROOT / "data" / "goalstep" / "sft_window"


def record(uid, tl, w_start, w, sys_p, with_history):
    evs = clip_events(tl, w_start, w)
    frames = [str(frame_path(uid, t).relative_to(ROOT)) for t in range(w_start, w_start + w)]
    user = history_text(tl, w_start) if with_history else USER_PLAIN
    return {
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps({"events": evs}, ensure_ascii=False)},
        ],
        "images": frames,
        "uid": uid, "w_start": w_start, "category": categorize(evs, w),
    }


def subsample_evenly(items: list, n: int) -> list:
    if n >= len(items):
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--boundary-stride", type=int, default=3,
                    help="境界を含む窓の増強ストライド(小さいほど増える)")
    ap.add_argument("--plain-ratio", type=float, default=0.6,
                    help="単一継続窓を境界系窓数の何倍採るか")
    ap.add_argument("--with-history", action="store_true", help="過去60秒の要約テキストを注入(条件B)")
    ap.add_argument("--extra-train-uids", default=None,
                    help="manifest外の追加学習動画uidリストファイル(1行1uid)")
    ap.add_argument("--max-train", type=int, default=None,
                    help="学習窓の総数上限(動画ごとに等間隔サブサンプルで削減)")
    ap.add_argument("--out-dir", default=None, help="出力先(既定 data/goalstep/sft_window)")
    args = ap.parse_args()
    w = args.window

    global OUT_DIR
    if args.out_dir:
        OUT_DIR = ROOT / args.out_dir

    ann = load_annotations()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    extra_uids = []
    if args.extra_train_uids:
        extra_uids = [l.strip() for l in open(args.extra_train_uids) if l.strip()]

    for split, fname in (("train", "train.jsonl"), ("valid", "validation.jsonl")):
        boundary_like, plain = [], []
        eval_all = []
        uids = [v["uid"] for v in load_manifest() if v["split"] == split]
        if split == "train":
            uids += extra_uids
        for uid in uids:
            tl = build_timeline(ann[uid])
            sys_p = system_prompt(vocab_for(ann[uid]), w)
            if any(not frame_path(uid, t).exists() for t in range(len(tl))):
                raise SystemExit(f"{uid}: フレーム未抽出。先に extract_frames.py を実行")
            # 学習用: 境界窓を細かいストライドで増強
            for w_start in range(0, len(tl) - w + 1, args.boundary_stride):
                r = record(uid, tl, w_start, w, sys_p, args.with_history)
                (boundary_like if r["category"] in ("boundary", "multi", "gap") else plain).append(r)
            # 評価用: 非重複・全窓
            for w_start in range(0, len(tl) - w + 1, w):
                eval_all.append(record(uid, tl, w_start, w, sys_p, args.with_history))

        # 同一遷移の重複を抑えるため、境界系は uid+時刻順のまま全件、単一継続はサンプル
        keep = boundary_like + subsample_evenly(plain, int(args.plain_ratio * len(boundary_like)))
        keep.sort(key=lambda r: (r["uid"], r["w_start"]))
        if split == "train" and args.max_train and len(keep) > args.max_train:
            # uid順ソート済みリストの等間隔サンプル = 全動画から比例配分で削減
            keep = subsample_evenly(keep, args.max_train)
        with (OUT_DIR / fname).open("w") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        cats = Counter(r["category"] for r in keep)
        print(f"{split}: 学習{len(keep)}窓 {dict(cats)}")
        if split == "valid":
            with (OUT_DIR / "validation_eval.jsonl").open("w") as f:
                for r in eval_all:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"validation_eval: {len(eval_all)}窓(非重複・全窓) {dict(Counter(r['category'] for r in eval_all))}")

    (OUT_DIR / "meta.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
