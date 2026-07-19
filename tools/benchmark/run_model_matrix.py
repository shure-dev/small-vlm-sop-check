#!/usr/bin/env python3
"""モデルmatrixを1プロセスずつ実行し、スモーク結果を機械可読に保存する。"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools" / "benchmark" / "run_qwen_video_prediction.py"
DEFAULT_CONFIG = ROOT / "configs" / "benchmark" / "factory_ego_model_matrix_v1.yaml"


def load_parser_module():
    spec = importlib.util.spec_from_file_location("video_prediction_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"runnerをロードできません: {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_models(config: dict[str, Any], keys: list[str]) -> list[dict[str, Any]]:
    models = config["models"]
    if not keys:
        return models
    by_key = {model["key"]: model for model in models}
    unknown = sorted(set(keys) - set(by_key))
    if unknown:
        raise SystemExit(f"未登録model key: {', '.join(unknown)}")
    return [by_key[key] for key in keys]


def command_for(
    python: Path,
    config: dict[str, Any],
    model: dict[str, Any],
    mode: str,
    run_root: Path,
    run_id: str,
) -> list[str]:
    limits = config["limits"]
    command = [
        str(python), str(RUNNER),
        "--queries", str(ROOT / config["queries"]),
        "--run-id", run_id,
        "--model", model["model"],
        "--model-name", model["name"],
        "--model-parameters-b", str(model["parameters_b"]),
        "--video-dir", str(ROOT / "out" / "model-videos"),
        "--fps", str(limits["video_fps"]),
        "--max-pixels", str(limits["max_pixels"]),
        "--max-tokens", str(limits["max_tokens"]),
        "--run-root", str(run_root),
    ]
    if model.get("upstream_model"):
        command.extend(["--upstream-model", model["upstream_model"]])
    if model.get("suppress_reasoning"):
        command.append("--suppress-reasoning")
    if model.get("json_prefill"):
        command.append("--json-prefill")
    if mode == "smoke":
        command.extend([
            "--skip-index", "--max-units", "1", "--max-events-per-unit", "1",
        ])
    return command


def smoke_output(run_dir: Path) -> tuple[bool, str | None, float | None]:
    raw_paths = sorted((run_dir / "raw").glob("*.json"))
    if not raw_paths:
        return False, None, None
    raw = json.loads(raw_paths[0].read_text(encoding="utf-8"))
    records = list(raw.get("events", {}).values())
    if not records:
        return False, None, None
    response = records[0].get("response", "")
    valid, _ = load_parser_module().parse_response(response)
    peak = records[0].get("peak_memory_gb")
    return valid, response, float(peak) if peak is not None else None


def write_status(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--model", action="append", default=[], dest="models")
    parser.add_argument("--python", type=Path, default=ROOT / ".venv-vlm" / "bin" / "python")
    parser.add_argument("--revision", default="r1", help="run IDの末尾revision")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    models = select_models(config, args.models)
    date = datetime.date.today().strftime("%Y%m%d")
    artifact_root = ROOT / "out" / "benchmarks" / config["benchmark_id"]
    run_root = artifact_root / "smoke-runs" if args.mode == "smoke" else ROOT / "runs"
    log_root = artifact_root / f"{args.mode}-logs"
    status_path = artifact_root / f"{args.mode}-status.json"
    status = {
        "benchmark_id": config["benchmark_id"],
        "config": str(config_path.relative_to(ROOT)),
        "mode": args.mode,
        "updated_at": datetime.datetime.now().astimezone().isoformat(),
        "models": {},
    }
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))

    for index, model in enumerate(models, 1):
        suffix = "smoke1" if args.mode == "smoke" else "full20"
        run_id = f"{date}-factory_ego-{model['key']}-{suffix}-{args.revision}"
        run_dir = run_root / run_id
        if (run_dir / "run.yaml").is_file():
            valid, response, peak = smoke_output(run_dir) if args.mode == "smoke" else (True, None, None)
            status["models"][model["key"]] = {
                "status": "passed" if valid else "failed_output_contract",
                "run_id": run_id,
                "peak_memory_gb": peak,
                "response": response,
                "note": "existing complete run reused",
            }
            write_status(status_path, status)
            continue

        # venv/bin/pythonはsymlinkの場合がある。resolve()するとvenvを外れて依存が消える。
        python = args.python if args.python.is_absolute() else (Path.cwd() / args.python)
        command = command_for(python.absolute(), config, model, args.mode, run_root, run_id)
        log_path = log_root / f"{model['key']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[matrix] {index}/{len(models)} {model['key']} ({args.mode})", flush=True)
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True,
            )
        elapsed = round(time.monotonic() - started, 3)
        valid, response, peak = smoke_output(run_dir) if process.returncode == 0 and args.mode == "smoke" else (process.returncode == 0, None, None)
        log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        result = {
            "status": (
                "passed" if valid else
                "failed_output_contract" if process.returncode == 0 else
                "failed_runtime"
            ),
            "returncode": process.returncode,
            "elapsed_s": elapsed,
            "run_id": run_id,
            "peak_memory_gb": peak,
            "response": response,
            "log": str(log_path.relative_to(ROOT)),
            "log_tail": log_tail,
        }
        status["models"][model["key"]] = result
        status["updated_at"] = datetime.datetime.now().astimezone().isoformat()
        write_status(status_path, status)
        print(f"[matrix] {model['key']}: {result['status']} ({elapsed}s)", flush=True)

    print(f"[matrix] status: {status_path}")


if __name__ == "__main__":
    main()
