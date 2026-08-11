"""デモ用の推論記録: 連続窓をストリーミング推論し、トークン到着時刻を実測で残す。

出力(JSONL): 1行目にメタ(モデルロード時間=コールドスタート)、以降は窓ごとに
  {"w_ix", "w_start", "gold", "pred", "verdict", "ttft_ms", "total_ms",
   "tokens": [{"t_ms", "text"}, ...], "raw"}
verdict: "ok"(全イベント一致) / "partial"(一部一致) / "ng"

v2窓モード(既定): 5秒窓×5フレームを毎窓同一プロンプトで解析。
v3スパンモード(--span 30): 10秒ごとに「過去20秒=履歴テキスト(オラクル) +
直近10秒=5フレーム」でスパン解析。記録するgold/pred/verdictは動画区間
[20,30)を窓相対(0-10)に直したもの(レンダラは窓モードと同じ扱いでよい)。

使い方:
    python scripts/demo_record.py --uid 80111886 --start 30 --windows 10 \
        --adapter training_runs/window-v2 --out runs/demo/after.jsonl
    python scripts/demo_record.py --uid 97c1c805 --start 490 --windows 3 \
        --span 30 --window 10 --adapter training_runs/span-v3 \
        --max-pixels 112896 --out runs/demo/v3.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtva.goalstep import ROOT, build_timeline, frame_path, load_annotations, vocab_for
from rtva.imaging import set_max_image_pixels
from rtva.metrics import event_match
from rtva.parsing import parse_events
from rtva.prompts import USER_PLAIN, system_prompt, system_prompt_span, user_prompt_span
from rtva.windows import clip_events


def span_history(gold_span: list[dict], text_s: int) -> list[dict]:
    """オラクル履歴: スパン前半のGTイベント(継続中は end=text_s に切る)。"""
    return [{"start_s": e["start_s"], "end_s": min(e["end_s"], text_s), "step": e["step"]}
            for e in gold_span if e["start_s"] < text_s]


def before_label(tl: list, s0: int) -> str | None:
    for t in range(s0 - 1, max(-1, s0 - 31), -1):
        if 0 <= t < len(tl) and tl[t]:
            return tl[t]
    return None


def to_video_region(events: list[dict], text_s: int, w: int) -> list[dict]:
    """スパン座標のイベント → 動画区間[text_s, text_s+w)を窓相対0-wに。"""
    out = []
    for e in events:
        a, b = max(0, e["start_s"] - text_s), min(w, e["end_s"] - text_s)
        if a < b:
            out.append({"start_s": a, "end_s": b, "step": e["step"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", required=True, help="動画uid(接頭辞可)")
    ap.add_argument("--start", type=int, default=0, help="開始秒(W の倍数推奨)")
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--span", type=int, default=0,
                    help="v3スパンモード: スパン長(秒)。0なら従来の窓モード")
    ap.add_argument("--frame-stride", type=int, default=2, help="スパンモードのフレーム間隔")
    ap.add_argument("--max-pixels", type=int, default=448 * 448,
                    help="学習時の --max-image-pixels と一致させること")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    w = args.window
    text_s = args.span - w  # スパンモードのみ使用
    if args.span:
        assert args.start >= text_s, f"スパンモードは --start >= {text_s} が必要(履歴分)"

    from mlx_vlm import load, stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    t_load0 = time.perf_counter()
    model, processor = load(
        "mlx-community/Qwen3.5-0.8B-MLX-4bit",
        adapter_path=args.adapter,
        processor_config={"trust_remote_code": True},
    )
    set_max_image_pixels(processor, args.max_pixels)
    config = model.config.__dict__
    load_ms = (time.perf_counter() - t_load0) * 1000

    ann = load_annotations()
    uid = next(u for u in ann if u.startswith(args.uid))
    tl = build_timeline(ann[uid])
    vocab = vocab_for(ann[uid])
    n_frames = w // args.frame_stride if args.span else w
    sys_p = (system_prompt_span(vocab, args.span, w, n_frames=n_frames)
             if args.span else system_prompt(vocab, w))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = out_path.open("w")
    f.write(json.dumps({"meta": {"uid": uid, "start": args.start, "window": w,
                                 "span": args.span or None,
                                 "adapter": args.adapter or "zero-shot",
                                 "model_load_ms": load_ms}}, ensure_ascii=False) + "\n")

    for i in range(args.windows):
        w_start = args.start + i * w  # 窓=スパンモードでは動画区間[w_start, w_start+w)
        step = args.frame_stride if args.span else 1
        frames = [str(frame_path(uid, t)) for t in range(w_start, w_start + w, step)]
        if not all(Path(p).exists() for p in frames):
            print(f"窓{i}: フレーム不足でスキップ (t={w_start})")
            continue
        if args.span:
            s0 = w_start - text_s
            gold_span = clip_events(tl, s0, args.span)
            user = user_prompt_span(span_history(gold_span, text_s), before_label(tl, s0), w)
            gold = to_video_region(gold_span, text_s, w)
        else:
            user = USER_PLAIN
            gold = clip_events(tl, w_start, w)
        messages = [{"role": "system", "content": sys_p}, {"role": "user", "content": user}]
        prompt = apply_chat_template(processor, config, messages, num_images=len(frames))
        tokens, text = [], ""
        t0 = time.perf_counter()
        for chunk in stream_generate(model, processor, prompt, image=frames,
                                     max_tokens=args.max_tokens, temperature=0.0):
            seg = chunk.text if hasattr(chunk, "text") else str(chunk)
            tokens.append({"t_ms": (time.perf_counter() - t0) * 1000, "text": seg})
            text += seg
        total_ms = (time.perf_counter() - t0) * 1000
        parsed = parse_events(text, set(vocab), args.span or w)
        pred = (to_video_region(parsed["events"], text_s, w) if args.span
                else parsed["events"])
        m = event_match(pred, gold)
        verdict = ("ok" if m["fn"] == 0 and m["fp"] == 0 and (m["tp"] > 0 or not gold)
                   else "partial" if m["tp"] > 0 else "ng")
        rec = {"w_ix": i, "w_start": w_start, "gold": gold, "pred": pred,
               "verdict": verdict, "ttft_ms": tokens[0]["t_ms"] if tokens else None,
               "total_ms": total_ms, "tokens": tokens, "raw": text}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"窓{i} t={w_start}-{w_start+w}s {verdict} TTFT {rec['ttft_ms']:.0f}ms 計{total_ms:.0f}ms")
    f.close()
    print(f"saved {out_path} (model_load {load_ms:.0f}ms)")


if __name__ == "__main__":
    main()
