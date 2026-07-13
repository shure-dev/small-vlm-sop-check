"""observe の結果を、フレーム画像と一緒に再生できる1枚のHTMLにまとめる。

なぜ作ったか:
  ターミナルの表だけでは「どの区間が検出されたのか」「VLMは本当は何と答えたのか」が
  パッと見で分かりにくい。この使い捨てツールは、動画のように再生しながら
  「今どのフレームで」「VLMは各質問にyes/no/unclearのどれと答え」「どのイベントが
  検出/進行中か」を1画面で見られるようにする。

出力は依存ファイルの一切ないHTML1枚(フレーム画像もbase64で埋め込み済み)。
ダブルクリックで開くだけで動く。サーバ不要・fetch不要。開くと自動で再生ループする。
ヘッダのプルダウンで「データセット → サンプル(unit) → モデル」を切り替えられる
(選択肢が1つしかない階層は自動で隠す)。

使い方:
  sop-replay
  # 見つかるものを全部入れた1枚HTMLを作る:
  #  - Konro Inspection: 同梱fixturesの15モデル+人手GT
  #  - Factory Ego: runs/ の全prediction run(Fable/Opus/ローカルモデル)。
  #    gated媒体なのでframes未取得のunitはスキップ、フレームは幅720pxに縮小埋め込み

  # 単一の回答ログだけを見たい場合(モデル切替なし):
  sop-replay \
    --answer-log datasets/konro_inspection/fixtures/reference_outputs/answer_log.json

  # Factory Egoのrunだけを見たい場合:
  sop-replay --runs-dir runs --out out/replay_factory_ego.html
"""
from __future__ import annotations
import argparse
import base64
import io
import json
from pathlib import Path

import yaml

from ..core.events import detect_events
from ..core.sop import load_sop
from .resources import repository_root, template_text


ROOT = repository_root()
DEMO_ROOT = ROOT / "datasets" / "konro_inspection"
DEFAULT_SOP = DEMO_ROOT / "sops" / "konro_inspection" / "konro_inspection.yaml"
DEFAULT_FRAMES = DEMO_ROOT / "units" / "konro_inspection" / "frames"
DEFAULT_MODELS = DEMO_ROOT / "fixtures" / "reference_outputs" / "models"
DEFAULT_GT = DEMO_ROOT / "annotations" / "human-v001" / "konro_inspection.json"
DEFAULT_RUNS = ROOT / "runs"
DEFAULT_FACTORY = ROOT / "datasets" / "factory_ego"

# runs閲覧の既定縮小幅。1920x1080を8unit分そのまま埋めるとHTMLが50MB超になる。
RUNS_MAX_WIDTH = 720

# モデル切替プルダウンの並び順(ベンチマーク順)。ここに無いモデルは末尾にソートして追加。
MODEL_ORDER = ["qwen3-4b", "gemma4-e2b", "cosmos-7b", "qwen2.5-3b",
               "minicpm-4.6", "internvl3-2b", "molmo-7b"]


def _confidence_to_answers(confidence: dict) -> dict[str, str]:
    return {qid: c["argmax"] for qid, c in confidence.items()}


def encode_image(path: Path, max_width: int | None) -> str:
    """フレーム1枚をdata URIにする。max_width指定時はPILで縮小してから埋め込む。
    PILが無い環境では縮小せず原寸のまま埋め込む(動くがHTMLが大きくなる)。"""
    data = path.read_bytes()
    if max_width:
        try:
            from PIL import Image
        except ImportError:
            return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
        img = Image.open(io.BytesIO(data))
        if img.width > max_width:
            img = img.resize((max_width, round(img.height * max_width / img.width)))
            buf = io.BytesIO()
            img.convert("RGB").save(buf, "JPEG", quality=78)
            data = buf.getvalue()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def build_frames_meta(frames_dir: Path, times: list[float],
                      max_width: int | None = None) -> tuple[list, list]:
    """全モデル共通の要素(フレーム画像・時刻)を作る。画像はunitごとに1度だけ埋め込む。
    フレームはファイル名順(f000.jpg / f0000.jpg どちらの桁数でもよい)。"""
    files = sorted(frames_dir.glob("f*.jpg"))
    images = [encode_image(p, max_width) for p in files]
    times = [times[i] if i < len(times) else float(i) for i in range(len(files))]
    return images, times


def _yes_spans(answers_by_idx: list[dict], event_id: str) -> list[dict]:
    """回答が連続で "yes" になる区間を、すべて列挙する(1フレームでも拾う)。

    detect_eventsのmin_frames/max_gapを通す前の生の出現。ビューアでは
    短い動作も落とさず全部の帯を見せる。
    """
    hits = [i for i, ans in enumerate(answers_by_idx)
            if ans and ans.get(event_id) == "yes"]
    spans, start, prev = [], None, None
    for i in hits:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            spans.append({"start": start, "end": prev})
            start = prev = i
    if start is not None:
        spans.append({"start": start, "end": prev})
    return spans


def build_model_data(sop_def: dict, raw_log: list, n_frames: int) -> dict:
    """1モデルぶんの回答・検出結果(画像は含めない。フレーム位置で共有画像と対応)。

    回答ログがunitより短い場合(例: Opus runは先頭10フレームのみ)は、
    足りないフレームを「回答なし」として埋める。区間検出は実在する回答だけで行う。
    """
    log = sorted(raw_log, key=lambda x: x["idx"])
    by_idx = {r["idx"]: r for r in log}
    frames = []
    for i in range(n_frames):
        r = by_idx.get(i)
        if r is None:
            frames.append({"raw": "(このrunはこのフレームの回答を含まない)",
                           "answers": {}, "probs": {}})
            continue
        frames.append({
            "raw": r.get("raw", ""),
            "answers": _confidence_to_answers(r.get("confidence", {})),
            "probs": {qid: c["probs"] for qid, c in r.get("confidence", {}).items()},
        })

    det_frames = [{"idx": r["idx"], "t": r["t"],
                   "answers": _confidence_to_answers(r.get("confidence", {}))} for r in log]
    detected = detect_events(sop_def["events"], det_frames, sop_def.get("defaults"))

    answers_by_idx = [f["answers"] for f in frames]
    events = {}
    for ev_def in sop_def["events"]:
        name = ev_def["id"]
        runs = detected.get(name, [])
        # 実際に一致したフレーム(max_gapの橋渡しを含まない)の全区間ぶんの和集合。tIoUに使う
        idxs = [i for run in runs
                for i in (run.idxs if run.idxs else range(run.start_idx, run.end_idx + 1))]
        events[name] = {
            "label": ev_def.get("ask") or name,
            "question": ev_def.get("ask", ""),
            "idxs": idxs or None,
            # 全出現区間(同じ動作が複数回起きたら各回を別区間として出す)。
            # min_frames/max_gapを通した代表検出はdetect_eventsが別途担う。
            "spans": _yes_spans(answers_by_idx, name),
        }
    return {
        "events": events,
        "frames": frames,
    }


def _ordered_model_names(names: list[str]) -> list[str]:
    known = [m for m in MODEL_ORDER if m in names]
    rest = sorted(n for n in names if n not in MODEL_ORDER)
    return known + rest


def load_gt_spans(gt_path: Path | None) -> dict | None:
    """ground_truth.json(あれば)から {event: [{start_idx,end_idx},...]|null} を取り出す。
    旧v0.1(単一区間dict)はリストへ正規化する。正解区間はビューア上で検出区間と
    重ねて表示し、tIoUも出す。"""
    if gt_path is None or not gt_path.exists():
        return None
    events = json.loads(gt_path.read_text(encoding="utf-8"))["events"]
    return {k: ([v] if isinstance(v, dict) else v) for k, v in events.items()}


def build_unit_data(sop_def: dict, frames_dir: Path, model_rows: dict[str, list],
                    model_order: list[str], gt_path: Path | None,
                    max_width: int | None = None, label: str | None = None) -> dict:
    """1 unit(サンプル)ぶんのビューアデータ(sop・画像・モデル別の回答/検出)を組み立てる。"""
    first_rows = model_rows[model_order[0]]
    times = [r["t"] for r in sorted(first_rows, key=lambda x: x["idx"])]
    images, times = build_frames_meta(frames_dir, times, max_width)
    n_frames = len(images)
    models = {name: build_model_data(sop_def, model_rows[name], n_frames)
              for name in model_order}
    return {
        "label": label or sop_def["sop"]["name"],
        "sop": {"id": sop_def["sop"]["id"], "name": sop_def["sop"]["name"]},
        "questions": sop_def["events"],   # イベント=質問(id/ask/values)。表示パネル用
        "n_frames": n_frames,
        "images": images,
        "times": times,
        "model_order": model_order,
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


def build_single_dataset(args, dataset_name: str) -> dict:
    """SOP+frames+回答ログ(answer_log形式)で、1 unitだけのデータセットを組み立てる。"""
    if args.ground_truth:
        gt_path = Path(args.ground_truth)
    elif Path(args.sop).resolve() == DEFAULT_SOP.resolve():
        gt_path = DEFAULT_GT
    else:
        gt_path = Path(args.sop).parent / "ground_truth.json"

    sop_def = load_sop(args.sop)
    model_logs = _collect_model_logs(args)
    order = _ordered_model_names(list(model_logs))
    model_rows = {name: json.loads(model_logs[name].read_text(encoding="utf-8"))
                  for name in order}
    unit = build_unit_data(sop_def, Path(args.frames_dir), model_rows, order, gt_path,
                           max_width=args.max_width)
    unit_id = sop_def["sop"]["id"]
    return {"name": dataset_name, "unit_order": [unit_id], "units": {unit_id: unit}}


def _rows_from_prediction(prediction: dict, raw_path: Path | None) -> list[dict]:
    """prediction schema(frames[].answers 必須、confidence任意)を回答ログの行形式へ変換。

    - confidenceが無い/欠けている質問はanswersから合成する(probsは空にして%表示を出さない)
    - raw文字列はローカルrunのraw/<unit>.json(answer_log形式)にあれば重ねる。
      Fable等のrawは形式が異なるため、無ければanswersのJSONを表示に使う
    """
    raw_texts: dict[int, str] = {}
    if raw_path and raw_path.exists():
        raw_doc = json.loads(raw_path.read_text(encoding="utf-8"))
        if isinstance(raw_doc, list):
            raw_texts = {r["idx"]: r.get("raw", "") for r in raw_doc}

    rows = []
    for f in prediction["frames"]:
        confidence = dict(f.get("confidence") or {})
        for qid, value in f["answers"].items():
            if qid not in confidence:
                confidence[qid] = {"probs": {}, "argmax": value}
        rows.append({"idx": f["idx"], "t": f["t"],
                     "raw": raw_texts.get(f["idx"]) or json.dumps(f["answers"], ensure_ascii=False),
                     "confidence": confidence})
    return rows


def build_runs_dataset(runs_dir: Path, dataset_root: Path,
                       max_width: int | None) -> dict | None:
    """runs/ の全prediction runを、サンプル切替+モデル切替できるデータセットにまとめる。
    表示できるunitが1つもなければNoneを返す(gated媒体が未取得のクローン等)。"""
    runs = []
    if runs_dir.is_dir():
        for run_dir in sorted(p for p in runs_dir.iterdir() if (p / "run.yaml").is_file()):
            run = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
            runs.append({"dir": run_dir, "name": run["model"]["name"],
                         "role": run["model"]["role"], "units": set(run["target_units"])})
    if not runs:
        return None
    # referenceの大型モデルを先頭に、あとは名前順
    runs.sort(key=lambda r: (r["role"] != "large_model_reference_prediction", r["name"]))

    unit_ids = sorted(set().union(*(r["units"] for r in runs)))
    units, skipped = {}, []
    for unit_id in unit_ids:
        unit_dir = dataset_root / "units" / unit_id
        frames_dir = unit_dir / "frames"
        if not any(frames_dir.glob("f*.jpg")):
            skipped.append(unit_id)  # gated媒体が未取得
            continue
        meta = json.loads((unit_dir / "meta.json").read_text(encoding="utf-8"))
        sop_def = load_sop((unit_dir / meta["sop_ref"]["path"]).resolve())
        model_rows, order = {}, []
        for r in runs:
            if unit_id not in r["units"]:
                continue
            prediction = json.loads(
                (r["dir"] / "predictions" / f"{unit_id}.json").read_text(encoding="utf-8"))
            model_rows[r["name"]] = _rows_from_prediction(
                prediction, r["dir"] / "raw" / f"{unit_id}.json")
            order.append(r["name"])
        gt_path = dataset_root / "annotations" / "human-v001" / f"{unit_id}.json"
        units[unit_id] = build_unit_data(sop_def, frames_dir, model_rows, order,
                                         gt_path if gt_path.exists() else None,
                                         max_width=max_width, label=unit_id)
    if skipped:
        print(f"[replay_viewer] frames未取得のためスキップ: {', '.join(skipped)}")
    if not units:
        return None
    dataset_id = yaml.safe_load((dataset_root / "dataset.yaml").read_text(encoding="utf-8")) \
        .get("dataset_id", dataset_root.name) if (dataset_root / "dataset.yaml").exists() \
        else dataset_root.name
    name = dataset_id.replace("_", " ").title()  # factory_ego -> Factory Ego
    return {"name": name, "unit_order": [u for u in unit_ids if u in units], "units": units}


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
    ap.add_argument("--runs-dir", default=None,
                    help="runs/ を指定するとprediction run閲覧だけの単一データセットモードになる")
    ap.add_argument("--dataset-root", default=str(DEFAULT_FACTORY),
                    help="run閲覧で参照するdataset(既定: datasets/factory_ego)")
    ap.add_argument("--max-width", type=int, default=None,
                    help=f"埋め込み画像の最大幅px(要PIL)。run閲覧の既定は{RUNS_MAX_WIDTH}")
    ap.add_argument("--out", default=str(ROOT / "out" / "replay.html"))
    args = ap.parse_args()

    custom_single = (args.answer_log or args.ground_truth
                     or Path(args.sop).resolve() != DEFAULT_SOP.resolve()
                     or Path(args.models_dir).resolve() != DEFAULT_MODELS.resolve()
                     or Path(args.frames_dir).resolve() != DEFAULT_FRAMES.resolve())

    datasets: dict[str, dict] = {}
    if args.runs_dir:
        entry = build_runs_dataset(Path(args.runs_dir), Path(args.dataset_root),
                                   args.max_width or RUNS_MAX_WIDTH)
        if entry is None:
            raise SystemExit("[replay_viewer] 表示できるrunがありません(gated媒体をfetchしてください)")
        datasets["runs"] = entry
    elif custom_single:
        datasets["custom"] = build_single_dataset(args, dataset_name=Path(args.sop).stem)
    else:
        # 既定: 見つかるデータセットを全部入れる(Konro fixtures + Factory Ego runs)
        datasets["konro_inspection"] = build_single_dataset(args, dataset_name="Konro Inspection")
        entry = build_runs_dataset(DEFAULT_RUNS, DEFAULT_FACTORY,
                                   args.max_width or RUNS_MAX_WIDTH)
        if entry is not None:
            datasets["factory_ego"] = entry

    data = {"dataset_order": list(datasets), "datasets": datasets}

    template = template_text("replay.html")
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")  # </script>混入対策
    html = template.replace('"__REPLAY_DATA__"', data_json)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    n_units = sum(len(d["unit_order"]) for d in datasets.values())
    n_models = len({m for d in datasets.values()
                    for u in d["units"].values() for m in u["model_order"]})
    print(f"[replay_viewer] {out_path} を書き出しました "
          f"({len(datasets)}データセット, {n_units}サンプル, {n_models}モデル, "
          f"{out_path.stat().st_size / 1e6:.1f}MB)")


if __name__ == "__main__":
    main()
