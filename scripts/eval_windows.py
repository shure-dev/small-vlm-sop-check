"""窓ベース評価: validation_eval.jsonl(非重複・全窓)をモデルに解かせて採点する。

ステートレスなので zero-shot / SFT後 で完全に同一条件。指標:
- 形式遵守率 / 秒単位正解率 / イベントP・R・F1(ラベル一致+tIoU>=0.5) / 境界MAE
- レイテンシ p50/p95 と締切(W秒)遵守率 — 「W秒の動画をW秒以内に」の実証

使い方:
    python scripts/eval_windows.py --out runs/eval-zeroshot-window.json          # zero-shot
    python scripts/eval_windows.py --adapter training_runs/window-v1 --out ...   # SFT後
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtva.goalstep import ROOT
from rtva.imaging import set_max_image_pixels
from rtva.metrics import event_match, secondly_accuracy, summarize
from rtva.parsing import parse_events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="アダプタdir(未指定ならzero-shot)")
    ap.add_argument("--data", default="data/goalstep/sft_window/validation_eval.jsonl")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--max-pixels", type=int, default=448 * 448,
                    help="画像トークン数の制御。学習時の --max-image-pixels と一致させること")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(
        "mlx-community/Qwen3.5-0.8B-MLX-4bit",
        adapter_path=args.adapter,
        processor_config={"trust_remote_code": True},
    )
    set_max_image_pixels(processor, args.max_pixels)  # 学習と同一にする(回避5)
    config = model.config.__dict__

    rows = [json.loads(l) for l in open(ROOT / args.data)]
    if args.max_examples:
        rows = rows[: args.max_examples]

    per_window, wall_ms, errors_sample = [], [], []
    for i, r in enumerate(rows):
        sys_p, usr, gold_raw = (m["content"] for m in r["messages"])
        gold = json.loads(gold_raw)["events"]
        vocab = {l[2:].strip() for l in sys_p.splitlines() if l.startswith("- ")}
        messages = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr}]
        prompt = apply_chat_template(processor, config, messages, num_images=len(r["images"]))
        t0 = time.perf_counter()
        out = generate(model, processor, prompt,
                       image=[str(ROOT / p) for p in r["images"]],
                       max_tokens=args.max_tokens, temperature=0.0, verbose=False)
        wall_ms.append((time.perf_counter() - t0) * 1000)
        text = out.text if hasattr(out, "text") else str(out)
        parsed = parse_events(text, vocab, args.window)

        sec_ok, sec_all = secondly_accuracy(parsed["events"], gold, args.window)
        m = event_match(parsed["events"], gold)
        per_window.append({"uid": r["uid"][:8], "w_start": r["w_start"],
                           "category": r.get("category"),
                           "sec_ok": sec_ok, "sec_all": sec_all,
                           "format_ok": parsed["format_ok"],
                           "gold": gold, "pred": parsed["events"],
                           "latency_ms": wall_ms[-1], **m})
        if (m["fn"] or not parsed["format_ok"]) and len(errors_sample) < 6:
            errors_sample.append({"uid": r["uid"][:8], "w_start": r["w_start"],
                                  "gold": gold, "pred": parsed["events"], "raw": text[:120]})
        if (i + 1) % 100 == 0:
            interim = summarize(per_window)
            print(f"{i+1}/{len(rows)} 秒正解率 {interim['secondly_acc']*100:.0f}% "
                  f"eventF1 {interim['event_f1']*100:.0f}%", flush=True)

    result = summarize(per_window)
    ms = sorted(wall_ms)
    result.update({
        "adapter": args.adapter or "zero-shot",
        "latency_ms_p50": ms[len(ms) // 2],
        "latency_ms_p95": ms[int(len(ms) * 0.95)],
        "deadline_ok_rate": sum(1 for x in ms if x <= args.window * 1000) / len(ms),
        "errors_sample": errors_sample,
        "per_window": per_window,  # エラー型分析(前後比較・Vision凍結の妥当性判定)に使う
    })
    print(json.dumps({k: v for k, v in result.items() if k != "per_window"},
                     ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
