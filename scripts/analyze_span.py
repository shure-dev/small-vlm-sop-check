"""v3スパン評価の区間別・カテゴリ別分析(per_window記録から)。

v3タスクの本質「接合」がどこまでできているかを分離計測する:
- 区間別秒正解率: テキスト由来の過去20秒(履歴の反映) vs 動画の直近10秒(視覚認識)
- イベント再現率の3分解:
    text-complete = 履歴としてプロンプトに全体が与えられたイベント(コピー力)
    seam-crossing = 履歴の途中まで+動画へ継続するイベント(接合・延長力)
    video-only    = 動画内で新たに始まるイベント(純粋な視覚検出力)
- カテゴリ別(seam_continue / seam_switch / video_multi)の全指標

使い方: python scripts/analyze_span.py runs/eval-*.json [...]  (複数指定で比較表)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtva.metrics import event_match, summarize
from rtva.windows import events_to_secondly

SPAN, TEXT_S = 30, 20


def region_secondly(per: list[dict]) -> tuple[float, float]:
    """秒正解率を (テキスト区間0-20s, 動画区間20-30s) に分離。"""
    t_ok = v_ok = 0
    for x in per:
        p = events_to_secondly(x["pred"], SPAN)
        g = events_to_secondly(x["gold"], SPAN)
        t_ok += sum(a == b for a, b in zip(p[:TEXT_S], g[:TEXT_S]))
        v_ok += sum(a == b for a, b in zip(p[TEXT_S:], g[TEXT_S:]))
    n = len(per)
    return t_ok / (n * TEXT_S), v_ok / (n * (SPAN - TEXT_S))


def recall_decomposition(per: list[dict]) -> dict:
    """goldイベントを由来3種に分け、それぞれの再現率(ラベル一致+tIoU>=0.5)を出す。"""
    kinds = {
        "text-complete": lambda g: g["end_s"] <= TEXT_S,
        "seam-crossing": lambda g: g["start_s"] < TEXT_S < g["end_s"],
        "video-only": lambda g: g["start_s"] >= TEXT_S,
    }
    out = {}
    for kind, sel in kinds.items():
        tp = fn = 0
        for x in per:
            gold_k = [g for g in x["gold"] if sel(g)]
            if not gold_k:
                continue
            m = event_match(x["pred"], gold_k)
            tp += m["tp"]
            fn += m["fn"]
        out[kind] = {"recall": tp / (tp + fn) if tp + fn else None, "n": tp + fn}
    return out


def analyze(path: str) -> dict:
    d = json.load(open(path))
    per = d["per_window"]
    t_acc, v_acc = region_secondly(per)
    rec = recall_decomposition(per)
    by_cat = {}
    groups = defaultdict(list)
    for x in per:
        groups[x.get("category") or "?"].append(x)
    for cat, xs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        by_cat[cat] = summarize(xs)
    return {
        "name": Path(path).stem.replace("eval-", ""),
        "overall": {k: d[k] for k in ("n_windows", "secondly_acc", "event_f1",
                                      "format_ok_rate", "latency_ms_p50")},
        "region_secondly": {"text_0_20s": t_acc, "video_20_30s": v_acc},
        "recall_decomposition": rec,
        "by_category": by_cat,
    }


def pct(v) -> str:
    return "  -  " if v is None else f"{v * 100:5.1f}"


def main() -> None:
    results = [analyze(p) for p in sys.argv[1:]]
    names = [r["name"] for r in results]
    w = max(len(n) for n in names) + 2

    print("== 全体 ==")
    print(" " * w + "  秒正解  eventF1  形式   p50ms")
    for r in results:
        o = r["overall"]
        print(f"{r['name']:<{w}}  {pct(o['secondly_acc'])}  {pct(o['event_f1'])}  "
              f"{pct(o['format_ok_rate'])}  {o['latency_ms_p50']:6.0f}")

    print("\n== 区間別秒正解率(接合の分離採点) ==")
    print(" " * w + "  テキスト0-20s  動画20-30s")
    for r in results:
        g = r["region_secondly"]
        print(f"{r['name']:<{w}}  {pct(g['text_0_20s'])}          {pct(g['video_20_30s'])}")

    print("\n== イベント再現率の3分解 ==")
    print(" " * w + "  履歴コピー   接合またぎ   動画内新規")
    for r in results:
        rd = r["recall_decomposition"]
        cells = [f"{pct(rd[k]['recall'])}(n={rd[k]['n']})"
                 for k in ("text-complete", "seam-crossing", "video-only")]
        print(f"{r['name']:<{w}}  " + "  ".join(cells))

    print("\n== カテゴリ別秒正解率 / eventF1 ==")
    cats = [c for c in ("seam_continue", "video_multi", "seam_switch", "video_empty")
            if any(c in r["by_category"] for r in results)]
    print(" " * w + "  " + "  ".join(f"{c:<20}" for c in cats))
    for r in results:
        cells = []
        for c in cats:
            s = r["by_category"].get(c)
            cells.append(f"{pct(s['secondly_acc'])}/{pct(s['event_f1'])} n={s['n_windows']:<3}"
                         if s else " " * 18)
        print(f"{r['name']:<{w}}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
