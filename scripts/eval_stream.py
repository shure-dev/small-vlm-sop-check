"""自己履歴ストリーミング評価: 履歴を自分の過去出力から構築して動画全体を追跡する。

eval_windows(オラクル履歴=GT由来)との差が「自己誤りの伝搬コスト」。
動画をstride=10秒で歩き、各スパンの履歴テキストは過去の自分の検出結果
(状態)から作る。状態は各スパンの動画区間[20,30)の予測のみで更新する
(テキスト区間の再述は採点はするが状態には書き戻さない)。

注意: 動画先頭20秒はどのスパンの動画区間にも入らないため追跡対象外。
採点は t>=20 の区間で行う(コールドスタートの正直な扱い)。
学習ジョブ実行中はGPUメモリ競合するため実行しないこと。

配管検証(モデル不使用): python scripts/eval_stream.py --mock-oracle
本実行:               python scripts/eval_stream.py --adapter training_runs/span-v3 \
                          --out runs/eval-stream-sft-v3.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtva.goalstep import ROOT, build_timeline, frame_path, load_annotations, load_manifest, vocab_for
from rtva.metrics import event_match
from rtva.parsing import parse_events
from rtva.prompts import system_prompt_span, user_prompt_span
from rtva.windows import clip_events, merge_windows

SPAN, VIDEO_S = 30, 10
TEXT_S = SPAN - VIDEO_S


def history_from_state(state: list[dict], s0: int) -> list[dict]:
    """自分の検出状態(絶対秒セグメント)からスパン座標の履歴(end<=TEXT_S)を作る。"""
    hist = []
    for seg in state:
        a, b = seg["start_s"] - s0, seg["end_s"] - s0
        if b <= 0 or a >= TEXT_S:
            continue
        hist.append({"start_s": max(0, a), "end_s": min(TEXT_S, b), "step": seg["step"]})
    return hist


def before_label_from_state(state: list[dict], s0: int) -> str | None:
    for seg in reversed(state):
        if seg["end_s"] <= s0 and seg["end_s"] > s0 - 30:
            return seg["step"]
    return None


def video_region_events(pred: list[dict]) -> list[dict]:
    """スパン予測から動画区間[TEXT_S, SPAN)だけを切り出す(状態更新用)。"""
    out = []
    for ev in pred:
        a, b = max(TEXT_S, ev["start_s"]), min(SPAN, ev["end_s"])
        if a < b:
            out.append({"start_s": a, "end_s": b, "step": ev["step"]})
    return out


def track_video(uid: str, tl: list, predict, per_window: list) -> list[dict]:
    """1動画をストリーミング追跡。predict(s0, hist, before) -> (events, latency_ms)"""
    contributions = []  # [(w_start, 動画区間イベント[スパン座標])]
    state = []
    for t in range(SPAN, len(tl) + 1, VIDEO_S):
        s0 = t - SPAN
        hist = history_from_state(state, s0)
        before = before_label_from_state(state, s0)
        pred, latency = predict(s0, hist, before)
        contributions.append((s0, video_region_events(pred)))
        state = merge_windows(contributions, SPAN)
        per_window.append({"uid": uid[:8], "w_start": s0, "pred": pred,
                           "gold": clip_events(tl, s0, SPAN),
                           "history_used": hist, "latency_ms": latency})
    return state


def score_video(state: list[dict], tl: list) -> dict:
    """t>=TEXT_S の区間で 秒正解率 + イベントP/R/F1(絶対座標)。"""
    T = len(tl)
    pred_tl = [None] * T
    for seg in state:
        for t in range(max(0, seg["start_s"]), min(T, seg["end_s"])):
            if pred_tl[t] is None:
                pred_tl[t] = seg["step"]
    lo = TEXT_S
    sec_ok = sum(1 for t in range(lo, T) if pred_tl[t] == tl[t])
    gold_segs = [{"start_s": e["start_s"] + lo, "end_s": e["end_s"] + lo, "step": e["step"]}
                 for e in clip_events(tl, lo, T - lo)]
    m = event_match([s for s in state if s["end_s"] > lo], gold_segs)
    return {"sec_ok": sec_ok, "sec_all": T - lo, **m}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--max-pixels", type=int, default=112896)
    ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--mock-oracle", action="store_true",
                    help="モデル不使用: goldをそのまま返す予測器で配管を検証")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ann = load_annotations()
    uids = [v["uid"] for v in load_manifest() if v["split"] == "valid"]

    model = processor = config = None
    if not args.mock_oracle:
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from rtva.imaging import set_max_image_pixels
        model, processor = load("mlx-community/Qwen3.5-0.8B-MLX-4bit",
                                adapter_path=args.adapter,
                                processor_config={"trust_remote_code": True})
        set_max_image_pixels(processor, args.max_pixels)
        config = model.config.__dict__

    per_video, per_window = [], []
    for uid in uids:
        tl = build_timeline(ann[uid])
        vocab = vocab_for(ann[uid])
        n_frames = VIDEO_S // args.frame_stride
        sys_p = system_prompt_span(vocab, SPAN, VIDEO_S, n_frames=n_frames)

        if args.mock_oracle:
            def predict(s0, hist, before, _tl=tl):
                return clip_events(_tl, s0, SPAN), 0.0
        else:
            def predict(s0, hist, before, _tl=tl, _sys=sys_p, _vocab=vocab, _uid=uid):
                user = user_prompt_span(hist, before, VIDEO_S)
                messages = [{"role": "system", "content": _sys},
                            {"role": "user", "content": user}]
                frames = [str(frame_path(_uid, tt))
                          for tt in range(s0 + TEXT_S, s0 + SPAN, args.frame_stride)]
                prompt = apply_chat_template(processor, config, messages, num_images=len(frames))
                t0 = time.perf_counter()
                out = generate(model, processor, prompt, image=frames,
                               max_tokens=args.max_tokens, temperature=0.0, verbose=False)
                latency = (time.perf_counter() - t0) * 1000
                text = out.text if hasattr(out, "text") else str(out)
                return parse_events(text, set(_vocab), SPAN)["events"], latency

        state = track_video(uid, tl, predict, per_window)
        s = score_video(state, tl)
        prec = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0.0
        rec = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0.0
        per_video.append({
            "uid": uid[:8], "duration_s": len(tl),
            "secondly_acc": s["sec_ok"] / s["sec_all"],
            "event_precision": prec, "event_recall": rec,
            "event_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        })
        print(f"{uid[:8]} ({len(tl)}s): 秒正解率 {per_video[-1]['secondly_acc']*100:.1f}% "
              f"eventF1 {per_video[-1]['event_f1']*100:.1f}%", flush=True)

    n = len(per_video)
    lat = sorted(x["latency_ms"] for x in per_window)
    agg = {
        "mode": "mock-oracle" if args.mock_oracle else (args.adapter or "zero-shot"),
        "n_videos": n, "n_windows": len(per_window),
        "secondly_acc_mean": sum(v["secondly_acc"] for v in per_video) / n,
        "event_f1_mean": sum(v["event_f1"] for v in per_video) / n,
        "latency_ms_p50": lat[len(lat) // 2] if lat else 0.0,
        "deadline_ok_rate": (sum(1 for x in lat if x <= VIDEO_S * 1000) / len(lat)) if lat else None,
        "per_video": per_video,
    }
    print(json.dumps({k: v for k, v in agg.items() if k != "per_video"},
                     ensure_ascii=False, indent=2))
    if args.out:
        agg["per_window"] = per_window
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(agg, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
