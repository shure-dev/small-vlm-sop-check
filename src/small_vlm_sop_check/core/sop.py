"""SOP定義ファイル(YAML)の読み込みと最低限のバリデーション。"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml


REQUIRED_TOP_KEYS = ("sop", "questions", "events")


def load_sop(path: str | Path) -> dict[str, Any]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_TOP_KEYS if k not in doc]
    if missing:
        raise ValueError(f"{path}: 必須キーが不足しています: {missing}")
    if "id" not in doc["sop"] or "name" not in doc["sop"]:
        raise ValueError(f"{path}: sop.id / sop.name は必須です")
    if not doc["questions"]:
        raise ValueError(f"{path}: questions が空です(VLMへのプロンプトを生成できません)")
    if not doc["events"]:
        raise ValueError(f"{path}: events が空です(検出対象がありません)")
    return doc


def load_answer_log(path: str | Path) -> list[dict[str, Any]]:
    """observe が出力したログを読み込み、detect_events が使う形に整形する。"""
    import json
    from ..inference.observe import confidence_to_answers

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = []
    for r in raw:
        frames.append({"idx": r["idx"], "t": r["t"], "answers": confidence_to_answers(r["confidence"])})
    return frames
