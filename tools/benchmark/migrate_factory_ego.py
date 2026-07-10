#!/usr/bin/env python3
"""Legacy Factory Ego assets -> reproducible accuracy-comparison benchmark.

The command is intentionally safe by default: without ``--apply`` it only
validates sources and prints a plan. Existing files are accepted only when
their bytes are identical; a differing destination is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "1.0"
MIGRATION_DATE = "2026-07-10"
DATASET_ID = "factory_ego"
SPLIT_ID = "development-v001"

FABLE_RUN_ID = "20260710-fable5-reference-r1"
OPUS_RUN_ID = "20260710-opus48-reference-r1"
QWEN_RUN_ID = "20260710-qwen3-4b-baseline-r1"


@dataclass(frozen=True)
class UnitSpec:
    unit_id: str
    legacy_dir: str
    letter: str
    clip_number: str
    start: int
    end: int

    @property
    def clip_id(self) -> str:
        return f"factory051_worker001_{self.clip_number}"


UNITS = (
    UnitSpec("f051_w001_assembly", "factory_assembly", "a", "00001", 26, 45),
    UnitSpec("f051_w001_board_cables", "factory_board_cables", "b", "00010", 90, 109),
    UnitSpec("f051_w001_cable_tying", "factory_cable_tying", "c", "00025", 31, 50),
    UnitSpec("f051_w001_part_inspection", "factory_part_inspection", "d", "00040", 10, 29),
    UnitSpec("f051_w001_tray_handoff", "factory_tray_handoff", "e", "00000", 105, 124),
    UnitSpec("f051_w001_assembly_cycle2", "factory_assembly_cycle2", "f", "00001", 46, 65),
    UnitSpec("f051_w001_connector_seated", "factory_connector_seated", "g", "00025", 60, 79),
    UnitSpec("f051_w001_part_pick", "factory_part_pick", "h", "00040", 30, 49),
)


DATASET_README = """# Factory Ego accuracy comparison dataset

全体のフォルダ境界と評価条件は[ベンチマーク文書](../../docs/benchmark/README.md)を参照してください。

Egocentric-10K の `factory051 / worker001` から切り出した、VLMがフレームごとの質問にどれだけ正しく答えられるかを比較するための開発用データです。

## 境界

- `units/`: 出典、区間、ローカル1fpsフレームのSHA manifest
- `sops/`: 比較時に使う質問・イベント仕様。現時点では `provisional`（暫定）
- `annotations/`: 人手で検証した正解だけを置く。現在のFactory Ego 8 unitには人手GTがない
- `splits/`: factory/workerを跨がせない分割。既存8 unitは全て `dev_seen`

Fable、Opus、Qwenなどモデルの出力は事実層には置かず、リポジトリ直下の `runs/` に保存します。モデル間一致は予備比較であり、人手GTができるまで「精度」とは扱いません。

## 出典と配布

- Upstream: `builddotai/Egocentric-10K`
- Upstream license: Apache-2.0
- Access: Hugging Face上で連絡先共有への同意が必要なgated dataset
- Sampling: 1920x1080映像から1fpsで抽出

抽出フレームは公開repositoryへ含めません。upstreamのgated accessを通して取得したローカル媒体を使い、`frames.sha256.json` で同一性を検証します。再配布する場合はupstreamの最新ライセンスだけでなく、アクセス時に同意した条件とプライバシー要件を確認してください。
"""

ANNOTATIONS_README = """# Human annotations

ここには人手で確認・裁定した正解だけを置きます。モデル生成結果は置きません。

Factory Egoの現8 unitには、まだ人手ground truthがありません。未注釈を空JSONやモデル予測で埋めず、人手注釈が完了した時点で `human-v001/<unit_id>.json` を追加します。
"""

EVALUATIONS_README = """# Evaluations

評価条件の正本は[評価ポリシー](../docs/benchmark/evaluation.md)です。

評価はprediction runとは別の、不変なevaluation runとして作成します。入力にはprediction run IDとhuman annotation revisionを必ず固定します。

Factory Egoには現在人手GTがないため、正式な精度指標はまだありません。モデル間一致だけを精度として報告しないでください。
"""

REPORT = """# Factory Ego model comparison

## 現在地

8 unit × 20 framesを対象に、Fable 5、Opus 4.8、Qwen3-VL-4Bの予測を同一形式へ移行しました。

| run | role | unit coverage | formal accuracy |
|---|---|---:|---|
| `20260710-fable5-reference-r1` | large-model reference prediction | 8/8 | 未評価（人手GTなし） |
| `20260710-opus48-reference-r1` | large-model reference prediction | 1/8、10 framesのみ | 未評価（人手GTなし） |
| `20260710-qwen3-4b-baseline-r1` | local small-VLM baseline | 8/8 | 未評価（人手GTなし） |

人手GT作成前に計算できるのは、一致率・回答分布・境界差などの予備比較です。precision、recall、F1、balanced accuracy、tIoUはhuman annotation revisionを入力にしたevaluation runで計算します。
"""


class MigrationError(RuntimeError):
    pass


class Writer:
    def __init__(self, apply: bool, refresh: bool = False):
        self.apply = apply
        self.refresh = refresh
        self.created = 0
        self.identical = 0
        self.refreshed = 0

    def write_bytes(self, path: Path, data: bytes) -> None:
        if path.exists():
            if path.read_bytes() == data:
                self.identical += 1
                return
            if not self.refresh:
                raise MigrationError(f"refusing to overwrite differing file: {path}")
            if self.apply:
                path.write_bytes(data)
            self.refreshed += 1
            return
        if self.apply:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.created += 1

    def write_text(self, path: Path, text: str) -> None:
        self.write_bytes(path, text.encode("utf-8"))

    def write_json(self, path: Path, value: Any, *, jsonl: bool = False) -> None:
        if jsonl:
            text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in value)
        else:
            text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        self.write_text(path, text)

    def write_yaml(self, path: Path, value: Any) -> None:
        self.write_text(path, yaml.safe_dump(value, allow_unicode=True, sort_keys=False))

    def copy(self, source: Path, target: Path) -> None:
        if not source.is_file():
            raise MigrationError(f"missing source file: {source}")
        self.write_bytes(target, source.read_bytes())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_prompt(questions: list[dict[str, Any]], domain_hint: str, t: float) -> str:
    legend_parts: list[str] = []
    schema_parts: list[str] = []
    for question in questions:
        values = [str(v).lower() if isinstance(v, bool) else str(v) for v in question.get("values", ["yes", "no"])]
        legend_parts.append(
            f'- {question["id"]}: {question["ask"]} (answer with {" or ".join(values)})'
        )
        schema_parts.append(f'"{question["id"]}":""')
    legend = "\n".join(legend_parts)
    schema = "{" + ",".join(schema_parts) + "}"
    return (
        f"{domain_hint} (time t={t}s). Report only what you can see (no guessing).\n"
        f"Fields:\n{legend}\n"
        "Fill each JSON value with exactly one allowed word (e.g. yes/no/unclear). "
        "Do NOT repeat the question text as the value:\n" + schema
    )


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalized_sop(source: Path, spec: UnitSpec) -> dict[str, Any]:
    value = load_yaml(source)
    value["sop"]["id"] = spec.unit_id
    value["sop"]["name"] = f"Factory Ego: {spec.unit_id}"
    value["benchmark"] = {
        "schema_version": SCHEMA_VERSION,
        "version": "v001",
        "status": "provisional",
        "purpose": "model_accuracy_comparison",
        "provenance": "migrated from the legacy Qwen comparison SOP; not ground truth",
    }
    return value


def fable_prediction(source: Path, run_id: str, unit_id: str) -> dict[str, Any]:
    raw = load_json(source)
    frames = []
    for frame in raw["frames"]:
        answers = {
            key: ("yes" if value else "no")
            for key, value in frame.items()
            if key not in {"idx", "phase"} and isinstance(value, bool)
        }
        frames.append({"idx": frame["idx"], "t": float(frame["idx"]), "answers": answers})
    metadata = {key: value for key, value in raw.items() if key != "frames"}
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "unit_id": unit_id,
        "prediction_type": "frame_question_answers",
        "frame_count": len(frames),
        "source_format": raw.get("schema_version", "unknown"),
        "question_definitions": raw.get("events_def", {}),
        "frames": frames,
        "legacy_metadata": metadata,
    }


def qwen_prediction(source: Path, unit_id: str) -> dict[str, Any]:
    raw = load_json(source)
    frames = []
    for frame in raw:
        confidence = frame.get("confidence", {})
        answers = {
            question_id: detail["argmax"]
            for question_id, detail in confidence.items()
            if detail.get("argmax") in {"yes", "no", "unclear"}
        }
        resource = {
            key: frame[key]
            for key in ("active_mb", "peak_mb")
            if key in frame
        }
        frames.append(
            {
                "idx": frame["idx"],
                "t": float(frame.get("t", frame["idx"])),
                "answers": answers,
                "confidence": confidence,
                "resource": resource,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": QWEN_RUN_ID,
        "unit_id": unit_id,
        "prediction_type": "frame_question_answers",
        "frame_count": len(frames),
        "answer_source": "candidate-token probabilities; argmax normalized to yes/no",
        "frames": frames,
    }


def make_run(
    run_id: str,
    model_name: str,
    role: str,
    target_units: list[str],
    notes: list[str],
    *,
    model_id: str | None = None,
) -> dict[str, Any]:
    model: dict[str, Any] = {"name": model_name, "role": role}
    if model_id:
        model["id"] = model_id
        model["revision"] = "unpinned_legacy_run"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "kind": "prediction",
        "status": "complete",
        "immutable": True,
        "created_at": MIGRATION_DATE,
        "model": model,
        "dataset": {"id": DATASET_ID, "split": SPLIT_ID},
        "target_units": target_units,
        "ground_truth_used": False,
        "metrics": None,
        "inference_code_revision": "unknown_legacy_scratchpad",
        "notes": notes,
    }


def require_sources(legacy: Path, scratch: Path) -> None:
    missing: list[Path] = []
    for spec in UNITS:
        candidates = (
            legacy / spec.legacy_dir / "frame_annotation_fable5.json",
            scratch / f"frames_{spec.clip_number}",
            scratch / f"vlm_unit_{spec.letter}" / "answer_log.json",
            scratch / f"vlm_unit_{spec.letter}" / "sop.yaml",
        )
        missing.extend(path for path in candidates if not path.exists())
        for source_idx in range(spec.start, spec.end + 1):
            frame = scratch / f"frames_{spec.clip_number}" / f"f{source_idx:04d}.jpg"
            if not frame.is_file():
                missing.append(frame)
    opus = legacy / "factory_assembly" / "frame_annotation.json"
    if not opus.is_file():
        missing.append(opus)
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:20])
        raise MigrationError(f"missing {len(missing)} source path(s):\n{preview}")


def migrate(repo: Path, legacy: Path, scratch: Path, writer: Writer) -> None:
    require_sources(legacy, scratch)
    dataset = repo / "datasets" / DATASET_ID
    unit_ids = [spec.unit_id for spec in UNITS]
    revision = git_revision(repo)

    writer.write_text(dataset / "README.md", DATASET_README)
    writer.write_text(dataset / "annotations" / "README.md", ANNOTATIONS_README)
    writer.write_text(repo / "evaluations" / "README.md", EVALUATIONS_README)
    writer.write_text(repo / "reports" / "model_comparison.md", REPORT)

    dataset_yaml = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "name": "Factory Ego VLM accuracy comparison",
        "purpose": "frame-level VLM observation accuracy comparison",
        "upstream": {
            "id": "builddotai/Egocentric-10K",
            "url": "https://huggingface.co/datasets/builddotai/Egocentric-10K",
            "license": "Apache-2.0",
            "access": "gated_contact_information_required",
        },
        "benchmark_state": {
            "human_ground_truth_available": False,
            "formal_accuracy_available": False,
            "allowed_without_ground_truth": ["agreement", "answer_distribution", "boundary_difference"],
        },
        "unit_count": len(UNITS),
        "media_distribution": "gated_source_frames_not_in_public_repository",
        "migration": {"date": MIGRATION_DATE, "code_revision": revision},
    }
    writer.write_yaml(dataset / "dataset.yaml", dataset_yaml)

    split = {
        "schema_version": SCHEMA_VERSION,
        "split_id": SPLIT_ID,
        "group_by": ["factory_id", "worker_id"],
        "assignments": {"dev_seen": unit_ids, "validation": [], "test": []},
        "groups": {"factory_051/worker_001": {"split": "dev_seen", "units": unit_ids}},
        "policy": {
            "test_must_be_unseen": True,
            "current_units_never_promote_to_test": True,
            "test_predictions_must_not_influence_prompt_selection": True,
        },
        "reason": "All current units share one factory and worker and have already been inspected by multiple models.",
    }
    writer.write_json(dataset / "splits" / f"{SPLIT_ID}.json", split)

    unit_locks: dict[str, Any] = {}
    input_locks: dict[str, Any] = {}
    normalized_sops: dict[str, dict[str, Any]] = {}
    frame_hashes_by_unit: dict[str, dict[str, str]] = {}

    for spec in UNITS:
        unit_dir = dataset / "units" / spec.unit_id
        source_frames = scratch / f"frames_{spec.clip_number}"
        hashes: dict[str, str] = {}
        for output_idx, source_idx in enumerate(range(spec.start, spec.end + 1)):
            source = source_frames / f"f{source_idx:04d}.jpg"
            target_name = f"f{output_idx:04d}.jpg"
            hashes[target_name] = sha256_file(source)
            writer.copy(source, unit_dir / "frames" / target_name)
        frame_hashes_by_unit[spec.unit_id] = hashes
        writer.write_json(unit_dir / "frames.sha256.json", hashes)

        sop = normalized_sop(scratch / f"vlm_unit_{spec.letter}" / "sop.yaml", spec)
        normalized_sops[spec.unit_id] = sop
        sop_path = dataset / "sops" / spec.unit_id / "v001.yaml"
        writer.write_yaml(sop_path, sop)

        source_annotation = load_json(legacy / spec.legacy_dir / "frame_annotation_fable5.json")
        meta = {
            "schema_version": SCHEMA_VERSION,
            "unit_id": spec.unit_id,
            "dataset_id": DATASET_ID,
            "benchmark_status": "dev_seen",
            "source": {
                "dataset": "builddotai/Egocentric-10K",
                "factory_id": "factory_051",
                "worker_id": "worker_001",
                "clip_id": spec.clip_id,
                "start_second": spec.start,
                "end_second": spec.end,
            },
            "sampling": {
                "fps": 1.0,
                "n_frames": len(hashes),
                "source_frame_start": spec.start,
                "source_frame_end": spec.end,
                "output_naming": "f0000.jpg...f0019.jpg",
            },
            "media": {
                "path": "frames",
                "sha256_manifest": "frames.sha256.json",
                "availability": "local_gated_source_not_committed",
            },
            "sop_ref": {
                "id": spec.unit_id,
                "version": "v001",
                "path": f"../../sops/{spec.unit_id}/v001.yaml",
                "status": "provisional",
            },
            "legacy_summary": source_annotation.get("step1_summary"),
            "ground_truth": {"available": False, "required_source": "human"},
        }
        writer.write_json(unit_dir / "meta.json", meta)

        meta_bytes = (json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
        sop_bytes = yaml.safe_dump(sop, allow_unicode=True, sort_keys=False).encode()
        frame_manifest_bytes = (json.dumps(hashes, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
        unit_locks[spec.unit_id] = {
            "meta_sha256": sha256_bytes(meta_bytes),
            "sop_sha256": sha256_bytes(sop_bytes),
            "frames_manifest_sha256": sha256_bytes(frame_manifest_bytes),
            "frame_count": len(hashes),
        }
        input_locks[spec.unit_id] = {
            "dataset_id": DATASET_ID,
            "split_id": SPLIT_ID,
            "unit_meta_sha256": unit_locks[spec.unit_id]["meta_sha256"],
            "sop_id": spec.unit_id,
            "sop_version": "v001",
            "sop_sha256": unit_locks[spec.unit_id]["sop_sha256"],
            "frames_manifest_sha256": unit_locks[spec.unit_id]["frames_manifest_sha256"],
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "split_id": SPLIT_ID,
        "units": unit_locks,
    }
    writer.write_json(dataset / "manifest.lock.json", manifest)

    # Fable: eight model-generated frame annotations, retained as predictions.
    fable_dir = repo / "runs" / FABLE_RUN_ID
    fable_run = make_run(
        FABLE_RUN_ID,
        "Claude Fable 5",
        "large_model_reference_prediction",
        unit_ids,
        [
            "Migrated from frame_annotation_fable5.json files.",
            "These predictions are not human ground truth.",
        ],
    )
    writer.write_yaml(fable_dir / "run.yaml", fable_run)
    writer.write_json(fable_dir / "inputs.lock.json", {"schema_version": SCHEMA_VERSION, "units": input_locks})
    for spec in UNITS:
        source = legacy / spec.legacy_dir / "frame_annotation_fable5.json"
        writer.copy(source, fable_dir / "raw" / f"{spec.unit_id}.json")
        writer.write_json(
            fable_dir / "predictions" / f"{spec.unit_id}.json",
            fable_prediction(source, FABLE_RUN_ID, spec.unit_id),
        )

    # Opus: only the first ten frames of the assembly unit exist.
    opus_spec = UNITS[0]
    opus_source = legacy / opus_spec.legacy_dir / "frame_annotation.json"
    opus_dir = repo / "runs" / OPUS_RUN_ID
    opus_run = make_run(
        OPUS_RUN_ID,
        "Claude Opus 4.8",
        "large_model_reference_prediction",
        [opus_spec.unit_id],
        [
            "Legacy visual prediction covers only frames 0-9 of the assembly unit.",
            "These predictions are not human ground truth.",
        ],
    )
    writer.write_yaml(opus_dir / "run.yaml", opus_run)
    writer.write_json(
        opus_dir / "inputs.lock.json",
        {"schema_version": SCHEMA_VERSION, "units": {opus_spec.unit_id: input_locks[opus_spec.unit_id]}},
    )
    writer.copy(opus_source, opus_dir / "raw" / f"{opus_spec.unit_id}.json")
    writer.write_json(
        opus_dir / "predictions" / f"{opus_spec.unit_id}.json",
        fable_prediction(opus_source, OPUS_RUN_ID, opus_spec.unit_id),
    )

    # Qwen: preserve raw logs, canonical argmax predictions, and exact rendered prompts.
    qwen_dir = repo / "runs" / QWEN_RUN_ID
    qwen_run = make_run(
        QWEN_RUN_ID,
        "Qwen3-VL-4B-Instruct 4-bit",
        "local_small_vlm_baseline",
        unit_ids,
        [
            "Migrated from vlm_unit_a...h answer_log.json files.",
            "Legacy execution did not pin the Hugging Face model revision or mlx-vlm version.",
        ],
        model_id="mlx-community/Qwen3-VL-4B-Instruct-4bit",
    )
    qwen_run["inference"] = {
        "backend": "mlx-vlm",
        "prefill": "{\"",
        "sampling_fps": 1.0,
        "max_tokens": 200,
        "prompt_builder": "src.observe.build_prompt",
    }
    writer.write_yaml(qwen_dir / "run.yaml", qwen_run)
    writer.write_json(qwen_dir / "inputs.lock.json", {"schema_version": SCHEMA_VERSION, "units": input_locks})
    writer.write_text(qwen_dir / "prompt" / "prefill.txt", "{\"")
    for spec in UNITS:
        raw_source = scratch / f"vlm_unit_{spec.letter}" / "answer_log.json"
        writer.copy(raw_source, qwen_dir / "raw" / f"{spec.unit_id}.json")
        writer.write_json(
            qwen_dir / "predictions" / f"{spec.unit_id}.json",
            qwen_prediction(raw_source, spec.unit_id),
        )
        sop = normalized_sops[spec.unit_id]
        rendered = [
            {
                "idx": idx,
                "t": float(idx),
                "prompt": build_prompt(sop["questions"], sop["sop"].get("domain_hint", ""), float(idx)),
            }
            for idx in range(20)
        ]
        writer.write_json(qwen_dir / "prompt" / "rendered" / f"{spec.unit_id}.jsonl", rendered, jsonl=True)

    index = [
        {
            "run_id": FABLE_RUN_ID,
            "kind": "prediction",
            "model": "Claude Fable 5",
            "role": "large_model_reference_prediction",
            "unit_count": 8,
            "formal_accuracy": None,
        },
        {
            "run_id": OPUS_RUN_ID,
            "kind": "prediction",
            "model": "Claude Opus 4.8",
            "role": "large_model_reference_prediction",
            "unit_count": 1,
            "formal_accuracy": None,
        },
        {
            "run_id": QWEN_RUN_ID,
            "kind": "prediction",
            "model": "Qwen3-VL-4B-Instruct 4-bit",
            "role": "local_small_vlm_baseline",
            "unit_count": 8,
            "formal_accuracy": None,
        },
    ]
    writer.write_json(repo / "runs" / "index.jsonl", index, jsonl=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-examples", type=Path, required=True)
    parser.add_argument("--scratchpad", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--apply", action="store_true", help="write files; default is dry-run")
    parser.add_argument(
        "--refresh-generated",
        action="store_true",
        help="replace differing generated files; requires --apply and is only for schema/policy migrations",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.refresh_generated and not args.apply:
        print("ERROR: --refresh-generated requires --apply")
        return 2
    writer = Writer(apply=args.apply, refresh=args.refresh_generated)
    try:
        migrate(args.repo.resolve(), args.legacy_examples.resolve(), args.scratchpad.resolve(), writer)
    except MigrationError as exc:
        print(f"ERROR: {exc}")
        return 1
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(
        f"{mode}: {writer.created} new file(s), {writer.refreshed} refreshed file(s), "
        f"{writer.identical} identical existing file(s)"
    )
    if not args.apply:
        print("No files were written. Re-run with --apply after reviewing the plan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
