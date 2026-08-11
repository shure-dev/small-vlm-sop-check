"""評価指標: 秒単位正解率・イベントP/R/F1(tIoU)・境界誤差。"""

from .windows import events_to_secondly


def secondly_accuracy(pred: list[dict], gold: list[dict], w: int) -> tuple[int, int]:
    """秒ごとのラベル一致数と総秒数を返す(無ラベル秒も採点対象: None==Noneは正解)。"""
    p, g = events_to_secondly(pred, w), events_to_secondly(gold, w)
    return sum(1 for a, b in zip(p, g) if a == b), w


def _tiou(a: dict, b: dict) -> float:
    inter = max(0, min(a["end_s"], b["end_s"]) - max(a["start_s"], b["start_s"]))
    union = max(a["end_s"], b["end_s"]) - min(a["start_s"], b["start_s"])
    return inter / union if union > 0 else 0.0


def event_match(pred: list[dict], gold: list[dict], iou_thr: float = 0.5) -> dict:
    """ラベル一致かつ tIoU>=閾値 の貪欲マッチングで TP/FP/FN を数える。"""
    used = set()
    tp = 0
    boundary_err = []
    for g in gold:
        best, best_iou = None, iou_thr
        for i, p in enumerate(pred):
            if i in used or p["step"] != g["step"]:
                continue
            iou = _tiou(p, g)
            if iou >= best_iou:
                best, best_iou = i, iou
        if best is not None:
            used.add(best)
            tp += 1
            p = pred[best]
            boundary_err += [abs(p["start_s"] - g["start_s"]), abs(p["end_s"] - g["end_s"])]
    return {
        "tp": tp, "fp": len(pred) - len(used), "fn": len(gold) - tp,
        "boundary_err": boundary_err,
    }


def summarize(per_window: list[dict]) -> dict:
    """窓ごとの計測値リストを集約して最終メトリクスにする。"""
    sec_ok = sum(x["sec_ok"] for x in per_window)
    sec_all = sum(x["sec_all"] for x in per_window)
    tp = sum(x["tp"] for x in per_window)
    fp = sum(x["fp"] for x in per_window)
    fn = sum(x["fn"] for x in per_window)
    be = [e for x in per_window for e in x["boundary_err"]]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n_windows": len(per_window),
        "secondly_acc": sec_ok / sec_all if sec_all else 0.0,
        "event_precision": prec,
        "event_recall": rec,
        "event_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "boundary_mae_s": sum(be) / len(be) if be else None,
        "format_ok_rate": sum(x["format_ok"] for x in per_window) / len(per_window),
    }
