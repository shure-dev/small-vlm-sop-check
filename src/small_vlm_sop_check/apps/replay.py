"""observe/judge の結果を、フレーム画像と一緒に再生できる1枚のHTMLにまとめる。

なぜ作ったか:
  ターミナルの表だけでは「結局PASSなのかFAILなのか」「VLMは本当は何と答えたのか」が
  パッと見で分かりにくい。この使い捨てツールは、動画のように再生しながら
  「今どのフレームで」「VLMは各質問にyes/no/unclearのどれと答え」「どのイベントが
  検出/進行中で」「最終的にPASS/FAILか」を1画面で見られるようにする。

出力は依存ファイルの一切ないHTML1枚(フレーム画像もbase64で埋め込み済み)。
ダブルクリックで開くだけで動く。サーバ不要・fetch不要。

使い方:
  sop-replay
  # デフォルトで同梱の models/ 配下(複数モデルの回答ログ)を読み、ヘッダのプルダウンで
  # モデルを切り替えられる1枚HTMLを作る。同じ動画・同じSOPをモデル別に見比べられる。

  # 別の判定(誤った手順など)を見たい場合はSOPを差し替える:
  sop-replay \
    --sop datasets/konro_inspection/sops/konro_inspection/wrong_order.yaml \
    --out out/replay_wrong_order.html

  # 単一の回答ログだけを見たい場合(モデル切替なし):
  sop-replay \
    --answer-log datasets/konro_inspection/fixtures/reference_outputs/answer_log.json
"""
from __future__ import annotations
import argparse
import base64
import json
from pathlib import Path

from ..core.judge import judge, parse_clauses
from ..core.sop import load_sop
from .resources import repository_root, template_text


ROOT = repository_root()
DEMO_ROOT = ROOT / "datasets" / "konro_inspection"
DEFAULT_SOP = DEMO_ROOT / "sops" / "konro_inspection" / "correct.yaml"
DEFAULT_FRAMES = DEMO_ROOT / "units" / "konro_inspection" / "frames"
DEFAULT_MODELS = DEMO_ROOT / "fixtures" / "reference_outputs" / "models"
DEFAULT_GT = DEMO_ROOT / "annotations" / "human-v001" / "konro_inspection.json"


# モデル切替プルダウンの並び順(ベンチマーク順)。ここに無いモデルは末尾にソートして追加。
MODEL_ORDER = ["qwen3-4b", "gemma4-e2b", "cosmos-7b", "qwen2.5-3b",
               "minicpm-4.6", "internvl3-2b", "molmo-7b"]


def _confidence_to_answers(confidence: dict) -> dict[str, str]:
    return {qid: c["argmax"] for qid, c in confidence.items()}


def build_frames_meta(raw_log: list, frames_dir: Path) -> tuple[list, list]:
    """全モデル共通の要素(フレーム画像・時刻)を作る。画像は1度だけ埋め込む。"""
    images, times = [], []
    for r in sorted(raw_log, key=lambda x: x["idx"]):
        img_path = frames_dir / f"f{r['idx']:03d}.jpg"
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii") if img_path.exists() else ""
        images.append(f"data:image/jpeg;base64,{b64}" if b64 else "")
        times.append(r["t"])
    return images, times


def build_model_data(sop_def: dict, raw_log: list) -> dict:
    """1モデルぶんの回答・判定結果(画像は含めない。フレーム位置で共有画像と対応)。"""
    log = sorted(raw_log, key=lambda x: x["idx"])
    frames = [{
        "raw": r.get("raw", ""),
        "answers": _confidence_to_answers(r.get("confidence", {})),
        "probs": {qid: c["probs"] for qid, c in r.get("confidence", {}).items()},
    } for r in log]

    judge_frames = [{"idx": r["idx"], "t": r["t"],
                     "answers": _confidence_to_answers(r.get("confidence", {}))} for r in log]
    result = judge(sop_def, judge_frames)

    events = {}
    for name, spec in sop_def["events"].items():
        run = result.events.get(name)
        events[name] = {
            "evidence": spec if isinstance(spec, str) else spec["evidence"],
            "start_idx": run.start_idx if run else None,
            "end_idx": run.end_idx if run else None,
            "t": run.t if run else None,
            # 実際に一致したフレーム(max_gapの橋渡しを含まない)。帯の描画とtIoUはこちらを使う
            "idxs": list(run.idxs) if run else None,
        }
    return {
        "verdict": result.verdict,
        "coverage": result.coverage,
        "violations": result.violations,
        "events": events,
        "frames": frames,
    }


def _ordered_model_names(names: list[str]) -> list[str]:
    known = [m for m in MODEL_ORDER if m in names]
    rest = sorted(n for n in names if n not in MODEL_ORDER)
    return known + rest


def load_gt_spans(gt_path: Path | None) -> dict | None:
    """ground_truth.json(あれば)から {event: {start_idx,end_idx}|null} を取り出す。
    正解区間はビューア上で検出区間と重ねて表示し、tIoUも出す。"""
    if gt_path is None or not gt_path.exists():
        return None
    return json.loads(gt_path.read_text(encoding="utf-8"))["events"]


def build_data(sop_path: Path, frames_dir: Path,
               model_logs: dict[str, Path], gt_path: Path | None = None) -> dict:
    """model_logs = {表示名: 回答ログのパス}。複数渡すとプルダウンで切り替えられる。"""
    sop_def = load_sop(sop_path)
    order = _ordered_model_names(list(model_logs))

    logs = {name: json.loads(model_logs[name].read_text(encoding="utf-8")) for name in order}
    images, times = build_frames_meta(logs[order[0]], frames_dir)
    models = {name: build_model_data(sop_def, logs[name]) for name in order}

    return {
        "sop": {"id": sop_def["sop"]["id"], "name": sop_def["sop"]["name"]},
        "questions": sop_def["questions"],
        "relations": sop_def.get("relations", []),
        "n_frames": len(images),
        "images": images,
        "times": times,
        "model_order": order,
        "models": models,
        "gt": load_gt_spans(gt_path),
    }


def _collect_model_logs(args) -> dict[str, Path]:
    """--answer-log 指定時はその1本、無指定なら --models-dir 配下の *.json を全部。"""
    if args.answer_log:
        p = Path(args.answer_log)
        return {p.stem: p}
    models_dir = Path(args.models_dir)
    logs = {p.stem: p for p in sorted(models_dir.glob("*.json"))}
    if not logs:
        raise SystemExit(f"[replay_viewer] {models_dir} に *.json が見つかりません")
    return logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sop", default=str(DEFAULT_SOP))
    ap.add_argument("--frames-dir", default=str(DEFAULT_FRAMES))
    ap.add_argument("--models-dir", default=str(DEFAULT_MODELS),
                    help="モデル別回答ログ(<表示名>.json)を置いたディレクトリ。プルダウンで切替")
    ap.add_argument("--answer-log", default=None,
                    help="単一の回答ログだけを見る場合に指定(モデル切替なし)")
    ap.add_argument("--ground-truth", default=None,
                    help="ground_truth.json のパス(既定: SOPと同じディレクトリにあれば自動で重ねる)")
    ap.add_argument("--out", default=str(ROOT / "out" / "replay.html"))
    args = ap.parse_args()

    if args.ground_truth:
        gt_path = Path(args.ground_truth)
    elif Path(args.sop).resolve() == DEFAULT_SOP.resolve():
        gt_path = DEFAULT_GT
    else:
        gt_path = Path(args.sop).parent / "ground_truth.json"
    data = build_data(Path(args.sop), Path(args.frames_dir), _collect_model_logs(args),
                      gt_path=gt_path)

    template = template_text("replay.html")
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")  # </script>混入対策
    html = template.replace('"__REPLAY_DATA__"', data_json)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    verdicts = ", ".join(f"{n}={data['models'][n]['verdict']}" for n in data["model_order"])
    print(f"[replay_viewer] {out_path} を書き出しました "
          f"({data['n_frames']}フレーム, {len(data['model_order'])}モデル: {verdicts})")


if __name__ == "__main__":
    main()
