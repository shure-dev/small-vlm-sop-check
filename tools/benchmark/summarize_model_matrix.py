#!/usr/bin/env python3
"""model matrixのfull評価とスモーク失敗を1つの公開artifactへ固定する。"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def concise_failure(key: str, result: dict[str, Any]) -> str:
    response = result.get("response")
    if key == "minicpm-v-4.6" and response:
        return f"20秒の範囲外を返してoutput contract不成立: {response.strip()}"
    tail = result.get("log_tail") or ""
    known = {
        "internvl3-2b": "mlx-vlmの動画processorでimage_features未初期化",
        "lfm2.5-vl-1.6b": "MLX重みと実装の不一致（projector layer norm不足）",
        "smolvlm2-256m": "MLX動画入力のimage token数と抽出40 frameが不一致",
        "smolvlm2-500m": "MLX動画入力のimage token数と抽出40 frameが不一致",
        "smolvlm2-2.2b": "MLX動画入力のimage token数と抽出40 frameが不一致",
        "fastvlm-0.5b": "変換repositoryのprocessor依存ファイル不足",
        "fastvlm-1.5b": "MLX重みとprojector実装の不一致",
        "perception-lm-1b": "変換snapshotにconfig.jsonがなくロード不可",
    }
    return known.get(key, tail[-500:].strip() or "runtime failure")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-status", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.smoke_status = args.smoke_status.resolve()
    args.out = args.out.resolve()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    smoke = json.loads(args.smoke_status.read_text(encoding="utf-8"))
    model_config = {model["key"]: model for model in config["models"]}
    results = []
    for evaluation_path in sorted((ROOT / "evaluations").glob("*-full20-r1.json")):
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        run_id = evaluation["prediction_run_id"]
        key = run_id.removeprefix("20260718-factory_ego-").removesuffix("-full20-r1")
        if key not in model_config and key != "marlin-2b":
            continue
        run_path = ROOT / "runs" / run_id / "run.yaml"
        results.append({
            "key": key,
            "model": evaluation["model"],
            "run_id": run_id,
            "evaluation": str(evaluation_path.relative_to(ROOT)),
            "evaluation_sha256": sha256_file(evaluation_path),
            "run_manifest_sha256": sha256_file(run_path),
            "metrics": evaluation["metrics"],
        })

    failures = []
    excluded = []
    for key, model in model_config.items():
        if model.get("full_enabled") is False:
            excluded.append({
                "key": key, "model": model["name"],
                "reason": model["full_exclusion_reason"],
            })
            continue
        result = smoke.get("models", {}).get(key)
        if result and result.get("status") != "passed":
            failures.append({
                "key": key, "model": model["name"],
                "status": result["status"],
                "reason": concise_failure(key, result),
            })

    artifact = {
        "benchmark_id": config["benchmark_id"],
        "kind": "development_model_matrix",
        "created_at": datetime.date.today().isoformat(),
        "formal_accuracy": False,
        "dataset": {
            "id": config["dataset"], "split": config["split"],
            "unit_count": config["unit_count"],
            "event_count": config["event_query_count"],
            "reference_occurrences": 81,
        },
        "task": config["task"],
        "config": str(args.config.relative_to(ROOT)),
        "config_sha256": sha256_file(args.config),
        "results": sorted(
            results, key=lambda row: row["metrics"]["mean_tiou"], reverse=True
        ),
        "smoke_failures": failures,
        "excluded_from_full_run": excluded,
        "limitations": [
            "All clips belong to the development split; this is not held-out accuracy.",
            "Each generative VLM was asked for at most one span per event while GT may contain repetitions.",
            "A smoke failure records runtime compatibility, not an intrinsic model-quality score.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[matrix-summary] results={len(results)} failures={len(failures)}: {args.out}")


if __name__ == "__main__":
    main()
