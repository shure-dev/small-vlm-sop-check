"""CLIエントリポイント。

  sop-check run     --sop SOP.yaml --video VIDEO.mp4 --out-dir out/
  sop-check observe --sop SOP.yaml --frames-dir DIR --out answer_log.json
  sop-check detect  --sop SOP.yaml --answer-log answer_log.json
  sop-check eval    --sop SOP.yaml --answer-log answer_log.json
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from pathlib import Path

from .core.events import Run, detect_events
from .core.sop import load_answer_log, load_sop

# 動作確認済みモデル(alias -> (mlx-community等のID, 短い実測メモ))。
# ここに無いモデルも --model にフルIDを直接渡せば使える。
# 実測の所見は experiments 側で確認したもの。
MODELS = {
    "qwen3-2b":   ("mlx-community/Qwen3-VL-2B-Instruct-4bit",  "Qwen3-VL 2B。軽量だがJSONが崩れやすい(mlx-vlm 0.6.3実測: 半数のフレームで形式崩壊)"),
    "qwen3-4b":   ("mlx-community/Qwen3-VL-4B-Instruct-4bit",  "Qwen3-VL 4B。既定(konroでPASS)"),
    "qwen2.5-3b": ("mlx-community/Qwen2.5-VL-3B-Instruct-4bit", "Qwen2.5-VL 3B"),
    "internvl3-2b": ("mlx-community/InternVL3-2B-4bit",        "InternVL3 2B。pointing等でyesを出しすぎる傾向"),
    "gemma4-e2b": ("mlx-community/gemma-4-e2b-it-4bit",        "Gemma4 E2B。形式はOKだがロードが遅い"),
    "minicpm-4.6": ("mlx-community/MiniCPM-V-4.6-4bit",        "MiniCPM-V 4.6 1.3B(思考モデル。prefill既定で全フレーム回答)"),
    "qwen3.5-4b": ("mlx-community/Qwen3.5-4B-MLX-4bit",        "Qwen3.5 4B(早期fusionのネイティブVLM)。batteryは完璧だがpoint2を取りこぼす"),
    "qwen3.5-2b": ("mlx-community/Qwen3.5-2B-MLX-4bit",        "Qwen3.5 2B。grill/batteryを過検出"),
    "qwen3.5-0.8b": ("mlx-community/Qwen3.5-0.8B-MLX-4bit",    "Qwen3.5 0.8B。超軽量。形式は安定"),
    "lfm2.5-1.6b": ("mlx-community/LFM2.5-VL-1.6B-4bit",       "LFM2.5-VL 1.6B。要mlx-vlm>=0.6.4(0.6.3はlfm2_vl実装バグでロード不可)"),
}
def resolve_model(key: str) -> str:
    """エイリアス or フルモデルIDのどちらを渡されてもmlx-vlmが読めるIDに解決する。"""
    entry = MODELS.get(key)
    return entry[0] if entry else key  # 未知のキーはフルIDとみなす


# 思考モード指定 -> Observerに渡すenable_thinking値。
THINKING = {"auto": None, "on": True, "off": False}


def _print_result(sop_name: str, events: dict[str, list[Run]]) -> None:
    print(f"\nSOP: {sop_name}")
    print(f"{'event':14s} {'status':13s} {'t(s)':>6s}  span(idx)")
    for name, runs in events.items():
        if not runs:
            print(f"{name:14s} {'NOT_DETECTED':13s} {'  -':>6s}")
            continue
        for k, run in enumerate(runs):
            label = name if len(runs) == 1 else f"{name}[{k + 1}]"
            print(f"{label:14s} {'detected':13s} {run.t:>6.1f}  {run.start_idx}-{run.end_idx}")
    n_det = sum(1 for runs in events.values() if runs)
    print(f"\n検出: {n_det}/{len(events)} イベント\n")


def _run_observer(sop, meta_or_paths, model_key, out_path, max_tokens=200,
                  thinking="auto", prefill='{"', backend="mlx"):
    """meta_or_pathsは[{"idx","t","path"}] または [path,...](idxはenumerateで振る)。"""
    from .inference.observe import Observer, TransformersObserver

    domain_hint = sop["sop"].get("domain_hint", "これは作業動画の1フレームです")
    model_name = resolve_model(model_key)  # エイリアス or フルモデルIDのどちらでも受ける
    cls = TransformersObserver if backend == "transformers" else Observer
    obs = cls(model=model_name, questions=sop["events"],
              enable_thinking=THINKING[thinking])

    if meta_or_paths and isinstance(meta_or_paths[0], str):
        meta = [{"idx": i, "t": round(i, 2), "path": p} for i, p in enumerate(meta_or_paths)]
    else:
        meta = meta_or_paths

    results = json.load(open(out_path)) if os.path.exists(out_path) else []
    done_idx = {r["idx"] for r in results}

    for m in meta:
        if m["idx"] in done_idx:
            continue
        rec = obs.ask(m["path"], t=m["t"], domain_hint=domain_hint,
                      max_tokens=max_tokens, prefill=prefill)
        results.append({"idx": m["idx"], "t": m["t"], "raw": rec["raw"],
                         "confidence": rec["confidence"], **rec["mem"]})
        conf_str = " ".join(f"{k}={v['argmax']}({v['probs'].get(v['argmax'], 0):.2f})"
                             for k, v in rec["confidence"].items())
        print(f"  t={m['t']:>5}s: {conf_str}", flush=True)
        results.sort(key=lambda r: r["idx"])
        json.dump(results, open(out_path, "w"), ensure_ascii=False, indent=2)
    return results


def cmd_run(args):
    """動画 -> フレーム抽出 -> VLMの回答収集 -> 区間検出 を1コマンドで実行する(Macで完結)。"""
    from .inference.extract import extract_frames

    sop = load_sop(args.sop)
    os.makedirs(args.out_dir, exist_ok=True)
    frames_dir = os.path.join(args.out_dir, "frames")
    answer_log_path = os.path.join(args.out_dir, "answer_log.json")

    print(f"[run] 1/3 動画からフレーム抽出中... ({args.video})")
    meta = extract_frames(args.video, frames_dir, fps=args.fps)
    print(f"[run]   {len(meta)}フレーム -> {frames_dir}")

    print(f"[run] 2/3 VLMがフレームごとの質問に回答中... (model={args.model})")
    _run_observer(sop, meta, args.model, answer_log_path,
                  max_tokens=args.max_tokens, thinking=args.thinking, prefill=args.prefill,
                  backend=args.backend)
    print(f"[run]   回答ログ -> {answer_log_path}")

    print("[run] 3/3 区間検出中...")
    frames = load_answer_log(answer_log_path)
    events = detect_events(sop["events"], frames, sop.get("defaults"))
    _print_result(sop["sop"]["name"], events)
    from .core.temporal import frame_answers_to_spans, prediction_document
    prediction = prediction_document(
        args.run_id or Path(args.out_dir).name,
        sop["sop"]["id"],
        "frame_classification",
        frame_answers_to_spans(frames, sop),
    )
    prediction_path = Path(args.out_dir) / "prediction.json"
    prediction_path.write_text(json.dumps(prediction, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    print(f"[run]   秒区間prediction -> {prediction_path}")


def cmd_observe(args):
    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")))
    if not frame_paths:
        print(f"[observe] {args.frames_dir} に.jpgが見つかりません", file=sys.stderr)
        sys.exit(1)
    sop = load_sop(args.sop)
    meta = [{"idx": i, "t": round(i / args.fps, 2), "path": p} for i, p in enumerate(frame_paths)]
    _run_observer(sop, meta, args.model, args.out,
                  max_tokens=args.max_tokens, thinking=args.thinking, prefill=args.prefill,
                  backend=args.backend)
    print(f"[observe] saved -> {args.out}")


def cmd_detect(args):
    sop = load_sop(args.sop)
    frames = load_answer_log(args.answer_log)
    events = detect_events(sop["events"], frames, sop.get("defaults"))
    _print_result(sop["sop"]["name"], events)
    if args.out:
        from .core.temporal import frame_answers_to_spans, prediction_document
        prediction = prediction_document(
            args.run_id or Path(args.answer_log).stem,
            args.unit_id or sop["sop"]["id"],
            "frame_classification",
            frame_answers_to_spans(frames, sop),
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(prediction, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"[detect] prediction -> {out}")


def cmd_eval(args):
    """共通の秒区間annotationとpredictionをTemporal IoUで評価する。"""
    from .core.temporal import (
        evaluate_temporal,
        format_temporal_report,
        load_annotation,
        load_json,
        load_prediction,
    )

    annotation_doc = load_json(args.ground_truth)
    prediction_doc = load_json(args.prediction)
    try:
        annotation = load_annotation(annotation_doc)
        prediction = load_prediction(prediction_doc)
    except ValueError as exc:
        raise SystemExit(f"[eval] {exc}") from exc

    if annotation_doc["unit_id"] != prediction_doc["unit_id"]:
        raise SystemExit("[eval] annotationとpredictionのunit_idが一致しません")

    result = evaluate_temporal(annotation, prediction)
    result["inputs"] = {
        "ground_truth": str(Path(args.ground_truth).resolve()),
        "prediction": str(Path(args.prediction).resolve()),
    }
    print(format_temporal_report(result))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"[eval] metrics -> {args.out}")


def cmd_models(args):
    """--model に使える動作確認済みエイリアスと実測メモを一覧する。"""
    print("動作確認済みモデル(--model にエイリアス or フルIDを渡せる):\n")
    for alias, (mid, note) in MODELS.items():
        print(f"  {alias:13s} {mid}")
        print(f"  {'':13s} {note}\n")


def _add_model_args(p):
    """observe/run 共通のモデル関連オプション。"""
    p.add_argument("--model", default="qwen3-4b", help="'qwen3-4b'等のエイリアス or HF/mlx-communityのフルID")
    p.add_argument("--max-tokens", type=int, default=200,
                   help="1フレームあたりの最大生成トークン(既定: 200)。思考モデルは1024程度に上げる")
    p.add_argument("--thinking", choices=["auto", "on", "off"], default="auto",
                   help="思考モードの明示指定(既定: auto=モデル任せ)。テンプレートが対応する場合のみ有効")
    p.add_argument("--prefill", default='{"',
                   help="アシスタント応答の先頭に差し込む文字列(既定: '{\"')。JSONを最初のキーの"
                        "途中まで固定し、Molmoの空応答やMiniCPMの思考/エコーを防ぐ。思考させたい時は '' で無効化")
    p.add_argument("--backend", choices=["mlx", "transformers"], default="mlx",
                   help="推論バックエンド(既定: mlx)。transformersは要torch。mlx変換で視覚入力が"
                        "壊れるモデル(例: SmolVLM2)を公式実装で動かすための代替経路")


def main():
    ap = argparse.ArgumentParser(prog="sop-check")
    sub = ap.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="動画→抽出→VLMの回答収集→区間検出を1コマンドで実行")
    p_run.add_argument("--sop", required=True)
    p_run.add_argument("--video", required=True)
    p_run.add_argument("--fps", type=float, default=1.0)
    p_run.add_argument("--out-dir", required=True)
    p_run.add_argument("--run-id", default=None)
    _add_model_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_obs = sub.add_parser("observe", help="Phase1のみ: 抽出済みフレームの質問にVLMが回答(信頼度付き)")
    p_obs.add_argument("--sop", required=True)
    p_obs.add_argument("--frames-dir", required=True)
    p_obs.add_argument("--fps", type=float, default=1.0)
    p_obs.add_argument("--out", required=True)
    _add_model_args(p_obs)
    p_obs.set_defaults(func=cmd_observe)

    p_detect = sub.add_parser("detect", help="Phase2のみ: 回答ログからイベント区間を検出")
    p_detect.add_argument("--sop", required=True)
    p_detect.add_argument("--answer-log", required=True)
    p_detect.add_argument("--out", default=None, help="共通の秒区間prediction JSON")
    p_detect.add_argument("--run-id", default=None)
    p_detect.add_argument("--unit-id", default=None)
    p_detect.set_defaults(func=cmd_detect)

    p_eval = sub.add_parser("eval", help="秒単位の正解区間と予測区間をTemporal IoUで評価")
    p_eval.add_argument("--ground-truth", required=True)
    p_eval.add_argument("--prediction", required=True)
    p_eval.add_argument("--out", default=None, help="評価結果をJSONでも保存する場合の出力先")
    p_eval.set_defaults(func=cmd_eval)

    p_models = sub.add_parser("models", help="--model に使える動作確認済みエイリアス一覧")
    p_models.set_defaults(func=cmd_models)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
