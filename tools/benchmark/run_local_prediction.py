#!/usr/bin/env python3
"""ローカルmlx-vlmモデルでFactory Egoの不変prediction runを新規作成する。

既存runと同一の形式で、
  runs/<run_id>/raw/<unit>.json          … 回答ログ(信頼度・リソース込み)
  runs/<run_id>/predictions/<unit>.json  … prediction schema準拠の正規化予測
  runs/<run_id>/run.yaml                 … run記述(モデルrevision・推論条件を固定)
  runs/<run_id>/inputs.lock.json         … 入力(SOP・frames manifest・meta)のSHA固定
を書き、runs/index.jsonl に追記する。

安全条件:
  - run.yaml が既にあるrun IDは不変とみなし拒否する
  - raw はフレームごとに逐次保存するので、途中で落ちても再実行で再開できる
  - predictions・run.yaml・lock・index は全unit完了後にまとめて確定する
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


class TimeBudgetExceeded(Exception):
    """--max-seconds に達したのでフレーム境界でクリーン終了する(rawは保存済み・run.yamlは書かない)。"""

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from small_vlm_sop_check.cli import resolve_model  # noqa: E402
from small_vlm_sop_check.core.sop import load_sop  # noqa: E402
from small_vlm_sop_check.core.temporal import (  # noqa: E402
    frame_answers_to_spans,
    prediction_document,
)

DATASET_ID = "factory_ego"
SPLIT_ID = "development"
DATASET_ROOT = ROOT / "datasets" / "factory_ego"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def hf_snapshot_revision(model_id: str) -> str:
    """HFキャッシュからロード対象snapshotのcommit hashを読む(モデルrevisionの固定用)。"""
    cache = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + model_id.replace("/", "--"))
    ref = cache / "refs" / "main"
    if ref.is_file():
        return ref.read_text(encoding="utf-8").strip()
    snapshots = sorted((cache / "snapshots").glob("*")) if (cache / "snapshots").is_dir() else []
    return snapshots[-1].name if snapshots else "unknown"


def discover_units(subset: str | None = None) -> list[str]:
    split = json.loads((DATASET_ROOT / "splits" / f"{SPLIT_ID}.json").read_text(encoding="utf-8"))
    split_units = set(split["assignments"]["dev_seen"])
    if subset is None:
        return sorted(split_units)
    subset_path = DATASET_ROOT / "subsets" / f"{subset}.json"
    if not subset_path.is_file():
        raise SystemExit(f"subset定義がありません: {subset_path}")
    subset_units = json.loads(subset_path.read_text(encoding="utf-8"))["unit_ids"]
    unknown = set(subset_units) - split_units
    if unknown:
        raise SystemExit(f"subsetにsplit外のunitがあります: {sorted(unknown)}")
    return sorted(subset_units)


def unit_paths(unit_id: str) -> dict[str, Path]:
    unit_dir = DATASET_ROOT / "units" / unit_id
    meta = json.loads((unit_dir / "meta.json").read_text(encoding="utf-8"))
    sop_path = (unit_dir / meta["sop_ref"]["path"]).resolve()
    manifest_path = unit_dir / meta["media"]["sha256_manifest"]
    frames_path = ROOT / "data" / DATASET_ID / "units" / unit_id / meta["media"]["path"]
    return {"dir": unit_dir, "meta": unit_dir / "meta.json", "sop": sop_path,
            "manifest": manifest_path, "frames": frames_path}


def unit_fps(unit_id: str) -> float:
    """unitの抽出fpsをmeta.jsonから読む(タイムスタンプ t=idx/fps の算出用)。"""
    meta = json.loads((unit_paths(unit_id)["meta"]).read_text(encoding="utf-8"))
    return float(meta.get("sampling", {}).get("fps", 1.0))


def observe_unit(observer, sop: dict, frames_dir: Path, raw_path: Path,
                 max_tokens: int, prefill: str, fps: float, deadline: float | None = None) -> list[dict]:
    """1 unitぶんの回答収集。rawへフレームごとに逐次保存し、再実行時は途中から再開。

    deadline(time.monotonic基準)を渡すと、各フレームを処理する前に超過を確認し、
    超過していればフレーム境界でTimeBudgetExceededを投げる(rawは保存済みで安全に再開できる)。
    """
    frame_files = sorted(frames_dir.glob("f*.jpg"))
    if not frame_files:
        raise SystemExit(f"framesがありません(gated媒体を先にfetch): {frames_dir}")
    domain_hint = sop["sop"].get("domain_hint", "これは作業動画の1フレームです")

    results = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else []
    done = {r["idx"] for r in results}
    for idx, path in enumerate(frame_files):
        if idx in done:
            continue
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeBudgetExceeded
        t = round(idx / fps, 2)  # sampling fpsはunitのmeta由来(新データセットは2fps)
        rec = observer.ask(str(path), t=t, domain_hint=domain_hint,
                           max_tokens=max_tokens, prefill=prefill)
        results.append({"idx": idx, "t": t, "raw": rec["raw"],
                        "confidence": rec["confidence"], **rec["mem"]})
        results.sort(key=lambda r: r["idx"])
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        conf = " ".join(f"{k}={v['argmax']}" for k, v in rec["confidence"].items())
        print(f"  [{frames_dir.parent.name}] t={t:>4.0f}s: {conf}", flush=True)
    return results


def normalize(run_id: str, unit_id: str, rows: list[dict], sop: dict) -> dict:
    """rawの信頼度argmaxを共通の秒区間predictionへ変換する。

    出力が崩壊して回答が取れなかった質問は "unclear" で埋める(全質問のキーを常に持つ)。
    実測: Qwen2.5-VL-3Bはロード直後の最初の推論でだけ '!' の羅列に退化することがある。
    """
    question_ids = [event["id"] for event in sop["events"]]
    frames = []
    for row in rows:
        answers = {qid: "unclear" for qid in question_ids}
        for qid, conf in row["confidence"].items():
            value = conf.get("argmax", "unclear")
            answers[qid] = value if value in {"yes", "no", "unclear"} else "unclear"
        frames.append({"idx": row["idx"], "t": row["t"], "answers": answers})
    return prediction_document(run_id, unit_id, "frame_classification",
                               frame_answers_to_spans(frames, sop))


def build_inputs_lock(units: list[str]) -> dict:
    lock: dict = {"units": {}}
    for unit_id in units:
        paths = unit_paths(unit_id)
        lock["units"][unit_id] = {
            "dataset_id": DATASET_ID,
            "frames_manifest_sha256": sha256_file(paths["manifest"]),
            "sop_id": unit_id,
            "sop_sha256": sha256_file(paths["sop"]),
            "split_id": SPLIT_ID,
            "unit_meta_sha256": sha256_file(paths["meta"]),
        }
    return lock


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="cli.MODELSのエイリアス or HF/mlx-communityフルID")
    ap.add_argument("--model-name", required=True, help="run.yaml/indexに載せる表示名(例: 'Qwen2.5-VL-3B-Instruct 4-bit')")
    ap.add_argument("--run-id", required=True, help="例: <日付>-factory_ego-qwen2.5-3b-baseline-r1")
    ap.add_argument("--role", default="local_small_vlm_baseline")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--prefill", default='{"')
    ap.add_argument("--subset", default=None,
                    help="datasets/factory_ego/subsets/<name>.json のunit_idsに対象を制限(既定: split全unit)")
    ap.add_argument("--backend", choices=["mlx", "transformers"], default="mlx",
                    help="推論バックエンド(既定: mlx)。transformersは要torch(SmolVLM2等)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="この秒数に達したらフレーム境界でクリーン終了する(rawは保存済み。再実行で続きから)。"
                         "長時間ジョブを短いチャンクに分割するための安全弁")
    args = ap.parse_args()

    run_dir = ROOT / "runs" / args.run_id
    if (run_dir / "run.yaml").exists():
        raise SystemExit(f"既存runは不変です。上書きしません: {run_dir}")

    units = discover_units(args.subset)
    model_id = resolve_model(args.model)
    model_revision = hf_snapshot_revision(model_id)
    print(f"[run] {args.run_id}: model={model_id} rev={model_revision} "
          f"units={len(units)} backend={args.backend}", flush=True)

    if args.backend == "transformers":
        from small_vlm_sop_check.inference.observe import TransformersObserver as ObserverCls
    else:
        from small_vlm_sop_check.inference.observe import Observer as ObserverCls

    first_sop = load_sop(unit_paths(units[0])["sop"])
    observer = ObserverCls(model=model_id, questions=first_sop["events"])

    deadline = time.monotonic() + args.max_seconds if args.max_seconds else None
    predictions: dict[str, dict] = {}
    try:
        for unit_id in units:
            paths = unit_paths(unit_id)
            sop = load_sop(paths["sop"])
            observer.set_questions(sop["events"])
            print(f"[run] unit {unit_id} ({len(sop['events'])} events)", flush=True)
            rows = observe_unit(observer, sop, paths["frames"], run_dir / "raw" / f"{unit_id}.json",
                                args.max_tokens, args.prefill, unit_fps(unit_id), deadline)
            predictions[unit_id] = normalize(args.run_id, unit_id, rows, sop)
    except TimeBudgetExceeded:
        done = sorted((run_dir / "raw").glob("*.json"))
        print(f"[run] --max-seconds={args.max_seconds}s に到達。rawは保存済み(raw {len(done)}ファイル)。"
              f"同じコマンドを再実行すれば続きから完了します。run.yamlは未作成。", flush=True)
        sys.exit(0)

    for unit_id, prediction in predictions.items():
        out = run_dir / "predictions" / f"{unit_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(prediction, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")

    lock = build_inputs_lock(units)
    (run_dir / "inputs.lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sampling_fps = sorted({unit_fps(u) for u in units})
    inference = {
        "backend": args.backend,
        "prefill": args.prefill,
        "sampling_fps": sampling_fps[0] if len(sampling_fps) == 1 else sampling_fps,
        "max_tokens": args.max_tokens,
        "prompt_builder": "small_vlm_sop_check.inference.observe.build_prompt",
        "frame_input": "single (最新1フレームのみ。時間文脈なし)",
    }
    if args.backend == "transformers":
        import torch
        import transformers
        # torch.__version__ はTorchVersion(str subclass)でyaml.safe_dumpが表現できないためstr()化
        inference["torch_version"] = str(torch.__version__)
        inference["transformers_version"] = str(transformers.__version__)
        inference["dtype"] = "float32"
    else:
        import mlx_vlm
        inference["mlx_vlm_version"] = mlx_vlm.__version__

    run_doc = {
        "run_id": args.run_id,
        "kind": "prediction",
        "status": "complete",
        "immutable": True,
        "created_at": datetime.date.today().isoformat(),
        "model": {"name": args.model_name, "role": args.role,
                  "id": model_id, "revision": model_revision},
        "dataset": {"id": DATASET_ID, "split": SPLIT_ID},
        "subset": args.subset,
        "target_units": units,
        "ground_truth_used": False,
        "metrics": None,
        "inference_code_revision": git_revision(),
        "notes": [
            "Generated by tools/benchmark/run_local_prediction.py.",
            "単一フレーム入力(最新1フレームのみ)。参照のOpus online5(直近5フレーム因果窓)とは方式が異なる。",
        ],
        "inference": inference,
    }
    (run_dir / "run.yaml").write_text(
        yaml.safe_dump(run_doc, allow_unicode=True, sort_keys=False), encoding="utf-8")

    index_path = ROOT / "runs" / "index.jsonl"
    entry = {"dataset": DATASET_ID, "formal_accuracy": None, "kind": "prediction",
             "model": args.model_name, "role": args.role, "run_id": args.run_id,
             "split": SPLIT_ID, "unit_count": len(units)}
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"[run] 完了: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
