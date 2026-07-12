#!/usr/bin/env python3
"""prediction run同士を、reference run(Fable/Opus)とのtIoUで予備比較する。

人手GTがないFactory Egoで許される予備比較(モデル間一致・境界差)の1つとして、
両runの回答から決定論的ルールでイベント区間を導き、イベントごとのtemporal IoUを測る。
referenceは正解(ground truth)ではないため、この数値は精度ではない。

- 比較はrun同士の共通unit・共通フレームidxに制限する
  (Opus runは1 unit・先頭10フレームしか持たないため、そのunitは両者を10フレームで比較する)
- mean tIoUは両runがイベントを検出したペアのみの平均。片側のみの検出は別カウント
  (core.evaluate の mean tIoU と同じ流儀)

使い方:
  python3 tools/benchmark/reference_tiou.py --reference <reference run_id>
  python3 tools/benchmark/reference_tiou.py --reference <run_id> --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from small_vlm_sop_check.core.events import detect_events  # noqa: E402
from small_vlm_sop_check.core.evaluate import tiou  # noqa: E402
from small_vlm_sop_check.core.sop import load_sop  # noqa: E402

DATASET_ROOT = ROOT / "datasets" / "factory_ego"
RUNS_ROOT = ROOT / "runs"


def load_run(run_id: str) -> dict:
    run_dir = RUNS_ROOT / run_id
    run = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
    predictions = {}
    for unit_id in run["target_units"]:
        predictions[unit_id] = json.loads(
            (run_dir / "predictions" / f"{unit_id}.json").read_text(encoding="utf-8"))
    return {"run": run, "predictions": predictions}


def unit_sop(unit_id: str) -> dict:
    meta = json.loads((DATASET_ROOT / "units" / unit_id / "meta.json").read_text(encoding="utf-8"))
    return load_sop((DATASET_ROOT / "units" / unit_id / meta["sop_ref"]["path"]).resolve())


def unit_events(sop: dict, prediction: dict, common_idx: set[int]) -> dict:
    frames = [{"idx": f["idx"], "t": f["t"], "answers": f["answers"]}
              for f in prediction["frames"] if f["idx"] in common_idx]
    return detect_events(sop["events"], frames, sop.get("defaults"))


def answer_agreement(ref_pred: dict, cand_pred: dict, common_idx: set[int]) -> tuple[int, int]:
    """共通フレーム×共通質問で回答(yes/no/unclear)が一致したスロット数と総スロット数を返す。

    tIoUはイベント区間の一致だが、これは回答レベルの素の一致率。JSON崩壊(全unclear)や
    全yes/全no退化しているモデルは、tIoUを測る前にここで低い一致率として表面化する。
    """
    ref_by_idx = {f["idx"]: f["answers"] for f in ref_pred["frames"]}
    cand_by_idx = {f["idx"]: f["answers"] for f in cand_pred["frames"]}
    matched = total = 0
    for idx in common_idx:
        ref_ans, cand_ans = ref_by_idx.get(idx, {}), cand_by_idx.get(idx, {})
        for qid in set(ref_ans) & set(cand_ans):
            total += 1
            if ref_ans[qid] == cand_ans[qid]:
                matched += 1
    return matched, total


def compare(reference: dict, candidate: dict) -> dict:
    """candidate runをreference runと突き合わせ、イベント区間tIoUの要約を返す。"""
    ref_units = set(reference["run"]["target_units"])
    cand_units = set(candidate["run"]["target_units"])
    rows = []
    agree_matched = agree_total = 0
    for unit_id in sorted(ref_units & cand_units):
        sop = unit_sop(unit_id)
        ref_pred = reference["predictions"][unit_id]
        cand_pred = candidate["predictions"][unit_id]
        common_idx = ({f["idx"] for f in ref_pred["frames"]}
                      & {f["idx"] for f in cand_pred["frames"]})
        m, t = answer_agreement(ref_pred, cand_pred, common_idx)
        agree_matched += m
        agree_total += t
        ref_events = unit_events(sop, ref_pred, common_idx)
        cand_events = unit_events(sop, cand_pred, common_idx)
        for name in sop["events"]:
            ref_run, cand_run = ref_events.get(name), cand_events.get(name)
            if ref_run and cand_run:
                status, value = "both", tiou(ref_run, cand_run)
            elif ref_run:
                status, value = "ref_only", None
            elif cand_run:
                status, value = "cand_only", None
            else:
                status, value = "both_absent", None
            rows.append({"unit": unit_id, "event": name, "status": status, "tiou": value,
                         "n_frames": len(common_idx)})

    matched = [r["tiou"] for r in rows if r["tiou"] is not None]
    count = lambda s: sum(1 for r in rows if r["status"] == s)
    return {
        "candidate_run": candidate["run"]["run_id"],
        "candidate_model": candidate["run"]["model"]["name"],
        "units": len(ref_units & cand_units),
        "events_total": len(rows),
        "both_detected": count("both"),
        "ref_only": count("ref_only"),
        "cand_only": count("cand_only"),
        "both_absent": count("both_absent"),
        "mean_tiou": round(sum(matched) / len(matched), 3) if matched else None,
        "answer_agreement": round(agree_matched / agree_total, 3) if agree_total else None,
        "answer_slots": agree_total,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True, help="基準にするrun ID(Fable/Opus等)")
    ap.add_argument("--candidates", nargs="*", default=None,
                    help="比較するrun ID(既定: reference以外の全run)")
    ap.add_argument("--json", default=None, help="詳細(イベント別)をJSONでも保存する出力先")
    args = ap.parse_args()

    reference = load_run(args.reference)
    candidate_ids = args.candidates or sorted(
        p.name for p in RUNS_ROOT.iterdir()
        if p.is_dir() and p.name != args.reference)

    results = [compare(reference, load_run(run_id)) for run_id in candidate_ids]
    results.sort(key=lambda r: (r["mean_tiou"] is None, -(r["mean_tiou"] or 0)))

    ref_model = reference["run"]["model"]["name"]
    print(f"reference: {args.reference} ({ref_model})")
    print("referenceは人手GTではないため、以下は精度ではなくモデル間の一致(予備比較)。")
    print("回答一致率=共通フレーム×共通質問でyes/no/unclearが一致した割合。mean tIoU=両者が検出したイベント区間の平均tIoU。\n")
    print("| model | units | 回答一致率 | mean tIoU |")
    print("|---|---:|---:|---:|")
    for r in results:
        mt = f"{r['mean_tiou']:.2f}" if r["mean_tiou"] is not None else "—"
        aa = f"{r['answer_agreement']:.0%}" if r["answer_agreement"] is not None else "—"
        print(f"| {r['candidate_model']} | {r['units']} | {aa} | {mt} |")

    # 片側しか検出しなかったイベントはtIoUを測れないので、表の外に注記する
    notes = []
    for r in results:
        for row in r["rows"]:
            if row["status"] == "ref_only":
                notes.append(f"- {r['candidate_model']}: {row['unit']} の {row['event']} を検出せず"
                             f"(referenceは検出。tIoU平均から除外)")
            elif row["status"] == "cand_only":
                notes.append(f"- {r['candidate_model']}: {row['unit']} の {row['event']} を"
                             f"referenceは検出していない(tIoU平均から除外)")
    if notes:
        print("\ntIoUを測れなかったイベント:")
        print("\n".join(notes))

    if args.json:
        payload = {"reference_run": args.reference, "reference_model": ref_model,
                   "note": "similarity to reference predictions; not accuracy",
                   "results": results}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(f"\ndetail -> {args.json}")


if __name__ == "__main__":
    main()
