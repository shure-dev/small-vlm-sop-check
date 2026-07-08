"""CLIエントリポイント。

  python src/cli.py run     --sop SOP.yaml --video VIDEO.mp4 --out-dir out/   # 動画→抽出→観察→判定を1コマンドで
  python src/cli.py observe --sop SOP.yaml --frames-dir DIR --out answer_log.json  # 観察だけ(フレーム済みの場合)
  python src/cli.py judge   --sop SOP.yaml --answer-log answer_log.json               # 判定だけ(観察済みの場合)
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

from sop import load_sop, load_answer_log
from judge import judge, JudgeResult, check_expectation

# 動作確認済みモデル(alias -> (mlx-community等のID, 短い実測メモ))。
# ここに無いモデルも --model にフルIDを直接渡せば使える。
# 実測の所見は experiments 側で確認したもの。「基準」= konro sop.yaml で総合PASSする。
MODELS = {
    "qwen3-2b":   ("mlx-community/Qwen3-VL-2B-Instruct-4bit",  "Qwen3-VL 2B。軽量"),
    "qwen3-4b":   ("mlx-community/Qwen3-VL-4B-Instruct-4bit",  "Qwen3-VL 4B。既定・基準(konroでPASS)"),
    "qwen2.5-3b": ("mlx-community/Qwen2.5-VL-3B-Instruct-4bit", "Qwen2.5-VL 3B"),
    "internvl3-2b": ("mlx-community/InternVL3-2B-4bit",        "InternVL3 2B。pointing等でyesを出しすぎる傾向"),
    "gemma4-e2b": ("mlx-community/gemma-4-e2b-it-4bit",        "Gemma4 E2B。形式はOKだがロードが遅い"),
    "minicpm-4.6": ("mlx-community/MiniCPM-V-4.6-4bit",        "MiniCPM-V 4.6 1.3B(思考モデル。prefill既定で全フレーム回答)"),
    "molmo-7b":   ("mlx-community/Molmo-7B-D-0924-4bit",       "Molmo 7B(prefill無しだと空応答が多い)"),
    "cosmos-7b":  ("mlx-community/Cosmos-Reason1-7B-4bit",     "Cosmos-Reason1 7B(NVIDIA物理推論。思考モデル)"),
}
# 旧来のエイリアス(README/CLAUDE.mdが参照)を後方互換で維持。
LEGACY_ALIASES = {"2b": "qwen3-2b", "4b": "qwen3-4b"}


def resolve_model(key: str) -> str:
    """エイリアス or フルモデルIDのどちらを渡されてもmlx-vlmが読めるIDに解決する。"""
    key = LEGACY_ALIASES.get(key, key)
    entry = MODELS.get(key)
    return entry[0] if entry else key  # 未知のキーはフルIDとみなす


# 思考モード指定 -> Observerに渡すenable_thinking値。
THINKING = {"auto": None, "on": True, "off": False}


def _print_result(sop_name: str, result: JudgeResult) -> None:
    print(f"\nSOP: {sop_name}")
    print(f"{'event':14s} {'status':13s} {'t(s)':>6s}  span(idx)")
    for name, run in result.events.items():
        if run:
            print(f"{name:14s} {'done':13s} {run.t:>6.1f}  {run.start_idx}-{run.end_idx}")
        else:
            print(f"{name:14s} {'NOT_DETECTED':13s} {'  -':>6s}")
    print(f"\ncoverage = {result.coverage:.0%}")
    if result.violations:
        print("違反:")
        for v in result.violations:
            print(f"  - {v}")
    else:
        print("違反: なし")
    print(f"\n>>> 総合判定: {result.verdict} <<<\n")


def _print_expectation(sop_def, result: JudgeResult) -> None:
    """SOPに expect(正解)があれば、verdict と『なぜ違反か(理由)』を当てられたかを表示。"""
    ev = check_expectation(sop_def, result)
    if ev is None:
        return
    parts = [f"verdict {'✓' if ev['verdict_ok'] else '✗'}"]
    for r in ev["reasons"]:
        target = r.get("relation") or r.get("event")
        parts.append(f"理由「{target}」({r['kind']}) {'✓当てた' if r['caught'] else '✗外した'}")
    mark = "✓" if ev["localized"] else "✗"
    print(f"[正解照合] {'  /  '.join(parts)}  =>  箇所特定 {mark}\n")


def _run_observer(sop, meta_or_paths, model_key, out_path, max_tokens=200,
                  thinking="auto", prefill='{"'):
    """meta_or_pathsは[{"idx","t","path"}] または [path,...](idxはenumerateで振る)。"""
    from observe import Observer

    domain_hint = sop["sop"].get("domain_hint", "これは作業動画の1フレームです")
    model_name = resolve_model(model_key)  # エイリアス or フルモデルIDのどちらでも受ける
    obs = Observer(model=model_name, questions=sop["questions"],
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
    """動画 -> フレーム抽出 -> VLM観察 -> 判定 を1コマンドで実行する(Macで完結)。"""
    from extract import extract_frames

    sop = load_sop(args.sop)
    os.makedirs(args.out_dir, exist_ok=True)
    frames_dir = os.path.join(args.out_dir, "frames")
    answer_log_path = os.path.join(args.out_dir, "answer_log.json")

    print(f"[run] 1/3 動画からフレーム抽出中... ({args.video})")
    meta = extract_frames(args.video, frames_dir, fps=args.fps)
    print(f"[run]   {len(meta)}フレーム -> {frames_dir}")

    print(f"[run] 2/3 VLMで観察中... (model={args.model})")
    _run_observer(sop, meta, args.model, answer_log_path,
                  max_tokens=args.max_tokens, thinking=args.thinking, prefill=args.prefill)
    print(f"[run]   観察ログ -> {answer_log_path}")

    print("[run] 3/3 判定中...")
    result = judge(sop, load_answer_log(answer_log_path))
    _print_result(sop["sop"]["name"], result)
    _print_expectation(sop, result)


def cmd_observe(args):
    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")))
    if not frame_paths:
        print(f"[observe] {args.frames_dir} に.jpgが見つかりません", file=sys.stderr)
        sys.exit(1)
    sop = load_sop(args.sop)
    meta = [{"idx": i, "t": round(i / args.fps, 2), "path": p} for i, p in enumerate(frame_paths)]
    _run_observer(sop, meta, args.model, args.out,
                  max_tokens=args.max_tokens, thinking=args.thinking, prefill=args.prefill)
    print(f"[observe] saved -> {args.out}")


def cmd_judge(args):
    sop = load_sop(args.sop)
    result = judge(sop, load_answer_log(args.answer_log))
    _print_result(sop["sop"]["name"], result)
    _print_expectation(sop, result)


def cmd_models(args):
    """--model に使える動作確認済みエイリアスと実測メモを一覧する。"""
    print("動作確認済みモデル(--model にエイリアス or フルIDを渡せる):\n")
    for alias, (mid, note) in MODELS.items():
        print(f"  {alias:13s} {mid}")
        print(f"  {'':13s} {note}\n")
    print(f"後方互換エイリアス: " + ", ".join(f"{k}={v}" for k, v in LEGACY_ALIASES.items()))


def _add_model_args(p):
    """observe/run 共通のモデル関連オプション。"""
    p.add_argument("--model", default="4b", help="'qwen3-4b'等のエイリアス or HF/mlx-communityのフルID(既定: 4b)")
    p.add_argument("--max-tokens", type=int, default=200,
                   help="1フレームあたりの最大生成トークン(既定: 200)。思考モデルは1024程度に上げる")
    p.add_argument("--thinking", choices=["auto", "on", "off"], default="auto",
                   help="思考モードの明示指定(既定: auto=モデル任せ)。テンプレートが対応する場合のみ有効")
    p.add_argument("--prefill", default='{"',
                   help="アシスタント応答の先頭に差し込む文字列(既定: '{\"')。JSONを最初のキーの"
                        "途中まで固定し、Molmoの空応答やMiniCPMの思考/エコーを防ぐ。思考させたい時は '' で無効化")


def main():
    ap = argparse.ArgumentParser(prog="python src/cli.py")
    sub = ap.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="動画→抽出→観察→判定を1コマンドで実行")
    p_run.add_argument("--sop", required=True)
    p_run.add_argument("--video", required=True)
    p_run.add_argument("--fps", type=float, default=1.0)
    p_run.add_argument("--out-dir", required=True)
    _add_model_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_obs = sub.add_parser("observe", help="Phase1のみ: 抽出済みフレームをVLMで観察(信頼度付き)")
    p_obs.add_argument("--sop", required=True)
    p_obs.add_argument("--frames-dir", required=True)
    p_obs.add_argument("--fps", type=float, default=1.0)
    p_obs.add_argument("--out", required=True)
    _add_model_args(p_obs)
    p_obs.set_defaults(func=cmd_observe)

    p_judge = sub.add_parser("judge", help="Phase2のみ: 観察ログをSOPと突き合わせて判定")
    p_judge.add_argument("--sop", required=True)
    p_judge.add_argument("--answer-log", required=True)
    p_judge.set_defaults(func=cmd_judge)

    p_models = sub.add_parser("models", help="--model に使える動作確認済みエイリアス一覧")
    p_models.set_defaults(func=cmd_models)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
