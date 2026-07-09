"""正解アノテーション(ground_truth.json) × 観察ログ → 観察精度の評価。

成功の一次定義はあくまで expect(verdict+違反理由)の一致(judge.check_expectation)。
このモジュールはその「説明変数」を測る診断層で、3つのレイヤーを持つ:
  - イベント区間  : 検出区間 vs 正解区間の tIoU。境界の完全一致は要求しない
                    (Ego4D等の時間的アクション検出と同じく、許容誤差は注釈側ではなく
                    指標のしきい値側で吸収する)
  - relations正答 : SOPの各relationを正解区間で評価した結論と検出区間で評価した結論が
                    同じか。judgeの合否を実際に分けるのはここ
  - フレーム回答  : 正解区間から導出したフレームラベルとVLM回答の一致(参考値)

ground_truth.json のスキーマ(tools/annotator/serve.py が書き出す):
  {
    "schema_version": "0.1",
    "sop_id": "konro_inspection",
    "fps": 1.0,
    "n_frames": 16,
    "events": {
      "ignite": {"start_idx": 1, "end_idx": 3},   # 起きた区間(フレームidx・両端含む)
      "gloves_worn": null                           # 「起きていない」と注釈済み
      # キー自体が無い = 未注釈(評価から除外)
    }
  }
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from judge import (Run, judge, check_relations, not_only_events, parse_clauses,
                   JudgeResult, check_expectation)


def load_ground_truth(path: str | Path) -> dict[str, Any]:
    gt = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("fps", "n_frames", "events"):
        if key not in gt:
            raise ValueError(f"{path}: ground_truthに必須キー {key!r} がありません")
    for name, span in gt["events"].items():
        if span is None:
            continue
        s, e = span.get("start_idx"), span.get("end_idx")
        if not (isinstance(s, int) and isinstance(e, int) and 0 <= s <= e < gt["n_frames"]):
            raise ValueError(f"{path}: events.{name} の区間が不正です: {span}")
    return gt


def gt_runs(gt: dict[str, Any]) -> dict[str, Run | None]:
    """正解区間を judge.Run に変換する(キーが無い=未注釈のイベントは含めない)。"""
    fps = gt["fps"]
    runs: dict[str, Run | None] = {}
    for name, span in gt["events"].items():
        if span is None:
            runs[name] = None
        else:
            s, e = span["start_idx"], span["end_idx"]
            runs[name] = Run(start_idx=s, end_idx=e,
                             t=round((s + e) / 2 / fps, 2), hits=e - s + 1)
    return runs


def tiou(a: Run, b: Run) -> float:
    """フレームidx集合同士の temporal IoU。idxs(実際に一致したフレーム)を持つRunは
    その集合で数え、max_gapで橋渡しした隙間フレームは重なりにも母数にも入れない
    (yes,no,yes を3フレーム連続の検出として扱わない)。idxsが無いRun(正解区間など)は
    区間(両端含む)を連続集合とみなす。"""
    a_set = set(range(a.start_idx, a.end_idx + 1)) if a.idxs is None else set(a.idxs)
    b_set = set(range(b.start_idx, b.end_idx + 1)) if b.idxs is None else set(b.idxs)
    inter = len(a_set & b_set)
    if inter == 0:
        return 0.0
    return round(inter / len(a_set | b_set), 3)


def _event_rows(sop_def: dict, gts: dict[str, Run | None],
                result: JudgeResult) -> list[dict]:
    """イベントごとの GT×検出 の突き合わせ。status:
    match(両方あり) / miss(GTあり検出なし) / false_detection(GTなし検出あり) /
    true_absent(両方なし) / no_gt(未注釈=評価対象外)
    """
    rows = []
    for name in sop_def["events"]:
        det = result.events.get(name)
        if name not in gts:
            status = "no_gt"
            gt_run = None
        else:
            gt_run = gts[name]
            if gt_run and det:
                status = "match"
            elif gt_run:
                status = "miss"
            elif det:
                status = "false_detection"
            else:
                status = "true_absent"
        rows.append({
            "event": name,
            "gt": [gt_run.start_idx, gt_run.end_idx] if gt_run else None,
            "detected": [det.start_idx, det.end_idx] if det else None,
            "tiou": tiou(gt_run, det) if (gt_run and det) else None,
            "status": status,
        })
    return rows


def _relation_rows(sop_def: dict, gts: dict[str, Run | None],
                   result: JudgeResult) -> list[dict]:
    """各relationについて「正解区間で評価した成否」と「検出区間で評価した成否」を比べる。
    未注釈イベントを含むrelationは評価不能として agree=None。"""
    relations = sop_def.get("relations", [])
    tolerance = sop_def.get("defaults", {}).get("order_tolerance_s", 0.0)
    det_violated = {d["relation"] for d in result.violation_details}
    rows = []
    for rel in relations:
        names = [w for w in rel.replace("not_overlaps", " ").split()
                 if w not in ("before", "overlaps", "not")]
        if any(n not in gts for n in names):
            rows.append({"relation": rel, "gt_ok": None, "detected_ok": None, "agree": None})
            continue
        gt_viol = check_relations([rel], gts, tolerance_s=tolerance)
        gt_ok, det_ok = not gt_viol, rel not in det_violated
        rows.append({"relation": rel, "gt_ok": gt_ok, "detected_ok": det_ok,
                     "agree": gt_ok == det_ok})
    return rows


def _frame_rows(sop_def: dict, gt: dict[str, Any], frames: list[dict]) -> list[dict]:
    """正解区間からフレームラベルを導出してVLM回答と突き合わせる(参考値)。

    (question, value) ごとに、正例 = その節を参照する注釈済みイベントの正解区間の和集合。
    区間外でその値を答えたら偽陽性と数える。この解釈は「occurrenceで全出現を注釈する」
    運用(annotatorの前提)でのみ正しい。
    """
    positives: dict[tuple[str, str], set[int]] = {}
    for name, spec in sop_def["events"].items():
        if name not in gt["events"]:
            continue  # 未注釈イベントの節は導出できない
        evidence = spec if isinstance(spec, str) else spec["evidence"]
        span = gt["events"][name]
        idxs = set(range(span["start_idx"], span["end_idx"] + 1)) if span else set()
        for clause in parse_clauses(evidence):
            positives.setdefault(clause, set()).update(idxs)

    rows = []
    for (q, v), pos in sorted(positives.items()):
        tp = sum(1 for f in frames if f["idx"] in pos and f["answers"].get(q) == v)
        fn = len(pos) - tp
        fp = sum(1 for f in frames if f["idx"] not in pos and f["answers"].get(q) == v)
        rows.append({
            "question": q, "value": v, "gt_frames": len(pos),
            "precision": round(tp / (tp + fp), 3) if (tp + fp) else None,
            "recall": round(tp / (tp + fn), 3) if pos else None,
            "false_positives": fp,
        })
    return rows


def evaluate(sop_def: dict[str, Any], gt: dict[str, Any],
             frames: list[dict]) -> dict[str, Any]:
    """観察ログ(frames)を正解アノテーション(gt)と突き合わせた評価一式を返す。"""
    result = judge(sop_def, frames)
    gts = gt_runs(gt)

    events = _event_rows(sop_def, gts, result)
    relations = _relation_rows(sop_def, gts, result)
    frame_rows = _frame_rows(sop_def, gt, frames)

    matched = [r["tiou"] for r in events if r["tiou"] is not None]
    n_gt_present = sum(1 for r in events if r["status"] in ("match", "miss"))
    tiou_at = {f"tiou@{th}": sum(1 for t in matched if t >= th)
               for th in (0.1, 0.3, 0.5)}

    # 正解区間そのものをjudgeの規則で評価した場合のverdict。
    # アノテーションとSOPが両方正しければ expect.verdict と一致するはず(注釈の自己検証)。
    excluded = not_only_events(sop_def.get("relations", []))
    required = [n for n in sop_def["events"] if n not in excluded and n in gts]
    gt_coverage = (sum(1 for n in required if gts[n] is not None) / len(required)
                   if required else 1.0)
    gt_violations = [r for r in relations if r["gt_ok"] is False]
    gt_verdict = "PASS" if (gt_coverage == 1.0 and not gt_violations) else "FAIL"

    exp = check_expectation(sop_def, result)
    return {
        "events": events,
        "relations": relations,
        "frames": frame_rows,
        "summary": {
            "mean_tiou": round(sum(matched) / len(matched), 3) if matched else None,
            **tiou_at,
            "n_gt_present": n_gt_present,
            "relation_agree": sum(1 for r in relations if r["agree"]),
            "relation_total": sum(1 for r in relations if r["agree"] is not None),
            "detected_verdict": result.verdict,
            "gt_verdict": gt_verdict,
            "expect_verdict": (sop_def.get("expect") or {}).get("verdict"),
            "expect_localized": exp["localized"] if exp else None,
        },
    }


def format_report(ev: dict[str, Any]) -> str:
    """evaluate()の結果を人間向けのテキストにする。"""
    s = ev["summary"]
    lines = ["", "イベント区間 (正解 vs 検出):",
             f"{'event':16s} {'GT(idx)':>9s} {'検出(idx)':>9s} {'tIoU':>6s}  状態"]
    label = {"match": "✓ 検出", "miss": "✗ 見逃し", "false_detection": "✗ 誤検出",
             "true_absent": "✓ 正しく未検出", "no_gt": "- 未注釈"}
    for r in ev["events"]:
        fmt = lambda sp: f"{sp[0]}-{sp[1]}" if sp else "なし"
        ti = f"{r['tiou']:.2f}" if r["tiou"] is not None else "-"
        lines.append(f"{r['event']:16s} {fmt(r['gt']):>9s} {fmt(r['detected']):>9s}"
                     f" {ti:>6s}  {label[r['status']]}")
    if s["mean_tiou"] is not None:
        ths = "  ".join(f"{k.split('@')[1]}:{v}/{s['n_gt_present']}"
                        for k, v in s.items() if k.startswith("tiou@"))
        lines.append(f"mean tIoU = {s['mean_tiou']:.2f}   検出数(tIoU>=しきい値) {ths}")

    lines += ["", "relationsの正答 (正解区間と同じ結論を出せたか):"]
    for r in ev["relations"]:
        if r["agree"] is None:
            lines.append(f"  - {r['relation']}  (未注釈イベントを含むため評価不能)")
        else:
            ok = lambda b: "成立" if b else "違反"
            mark = "✓" if r["agree"] else "✗"
            lines.append(f"  {mark} {r['relation']}  GT:{ok(r['gt_ok'])} 検出:{ok(r['detected_ok'])}")
    lines.append(f"  一致 {s['relation_agree']}/{s['relation_total']}")

    if ev["frames"]:
        lines += ["", "フレーム回答 (正解区間から導出・参考値):"]
        for r in ev["frames"]:
            p = f"{r['precision']:.2f}" if r["precision"] is not None else "  - "
            rc = f"{r['recall']:.2f}" if r["recall"] is not None else "  - "
            lines.append(f"  {r['question']}=={r['value']:8s} precision {p}  recall {rc}"
                         f"  (GT {r['gt_frames']}フレーム / 偽陽性 {r['false_positives']})")

    lines += ["", f"verdict: 検出={s['detected_verdict']} / 正解区間から={s['gt_verdict']}"
                  + (f" / expect={s['expect_verdict']}" if s["expect_verdict"] else "")]
    if s["gt_verdict"] != (s["expect_verdict"] or s["gt_verdict"]):
        lines.append("  ⚠ 正解区間から導いたverdictがexpectと不一致 — アノテーションかSOPのどちらかを見直すこと")
    lines.append("")
    return "\n".join(lines)
