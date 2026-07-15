"""SOP定義ファイル(YAML)の読み込み・検証・書き戻し。

SOPフォーマット:
    sop: {id, name, domain_hint?}
    defaults: {min_frames?, max_gap_frames?}    # 任意
    events:
      - id: knob                # 回答ログ・GT・検出結果すべてのキー
        ask: "..."              # VLMへ送る質問(yes/noで答えられる文)
        values: ["yes", "no"]   # 任意(既定 yes/no)。プロンプト生成に使う
        min_frames: 2           # 任意

イベント = 質問。同じ動作が複数回起こる場合は、GT側で
同じイベントidに複数区間を注釈し、検出側も複数区間を返す。

annotation保存層がSOPを編集するための決定論的なdict変換
(set_domain_hint / upsert_event / rename_event / delete_event)と、
原子的なYAML書き出し(save_sop)もここに置く。書き出しはPyYAMLのsafe_dumpを
使うため 'yes'/'no' は自動でクォートされ、YAML 1.1のブール化は起きない。
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml


REQUIRED_TOP_KEYS = ("sop", "events")
DEFAULT_VALUES = ["yes", "no"]


def validate_sop(doc: dict[str, Any], path: str | Path = "<sop>") -> dict[str, Any]:
    """SOP dictが最低限の構造を満たすか確認する(満たさなければValueError)。"""
    missing = [k for k in REQUIRED_TOP_KEYS if k not in doc]
    if missing:
        raise ValueError(f"{path}: 必須キーが不足しています: {missing}")
    if "questions" in doc:
        raise ValueError(
            f"{path}: questionsは未対応です。eventsの各要素にaskを指定してください")
    if "id" not in doc["sop"] or "name" not in doc["sop"]:
        raise ValueError(f"{path}: sop.id / sop.name は必須です")
    events = doc["events"]
    if not isinstance(events, list):
        raise ValueError(f"{path}: events はイベントのリストです")
    seen: set[str] = set()
    for ev in events:
        if not isinstance(ev, dict) or "id" not in ev:
            raise ValueError(f"{path}: 各イベントは id を持つマップです: {ev!r}")
        if not str(ev["id"]).isidentifier():
            raise ValueError(f"{path}: イベントidが不正です(識別子のみ可): {ev['id']!r}")
        if ev["id"] in seen:
            raise ValueError(f"{path}: イベントidが重複しています: {ev['id']}")
        seen.add(ev["id"])
    return doc


def load_sop(path: str | Path) -> dict[str, Any]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_sop(doc, path)
    for ev in doc["events"]:
        ev.setdefault("ask", "")
        ev.setdefault("values", list(DEFAULT_VALUES))
    return doc


def save_sop(path: str | Path, sop_def: dict[str, Any]) -> None:
    """SOP dictを検証してYAMLへ原子的に書き込む(tmpに書いてから置き換え)。

    'yes'/'no' はsafe_dumpが自動でクォートするのでYAML 1.1のブール化は起きない。
    未知キーもそのまま保存される(dictを丸ごとdumpするため)。
    """
    validate_sop(sop_def, path)
    path = Path(path)
    text = yaml.safe_dump(sop_def, sort_keys=False, allow_unicode=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def load_answer_log(path: str | Path) -> list[dict[str, Any]]:
    """observe が出力したログを読み込み、detect_events が使う形に整形する。"""
    import json
    from ..inference.observe import confidence_to_answers

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = []
    for r in raw:
        frames.append({"idx": r["idx"], "t": r["t"], "answers": confidence_to_answers(r["confidence"])})
    return frames


# ---------------------------------------------------------------------------
# SOP編集（UIから独立した決定論的なdict変換）
# ---------------------------------------------------------------------------

def get_event(sop_def: dict[str, Any], event_id: str) -> dict[str, Any] | None:
    return next((ev for ev in sop_def["events"] if ev["id"] == event_id), None)


def set_domain_hint(sop_def: dict[str, Any], hint: str) -> dict[str, Any]:
    """sop.domain_hint(撮影状況などのヒント文)を設定する。空文字ならキーごと削除。"""
    hint = (hint or "").strip()
    if hint:
        sop_def["sop"]["domain_hint"] = hint
    else:
        sop_def["sop"].pop("domain_hint", None)
    return sop_def


def upsert_event(sop_def: dict[str, Any], event_id: str, *,
                 ask: str | None = None,
                 min_frames: int | None = None) -> dict[str, Any]:
    """イベントを追加、または既存イベントの質問文/min_framesを更新する。"""
    if not event_id or not event_id.isidentifier():
        raise ValueError(f"イベントidが不正です(識別子のみ可): {event_id!r}")
    ev = get_event(sop_def, event_id)
    if ev is None:
        ev = {"id": event_id, "ask": ask or "", "values": list(DEFAULT_VALUES)}
        if min_frames is not None:
            ev["min_frames"] = min_frames
        sop_def["events"].append(ev)
    else:
        if ask is not None:
            ev["ask"] = ask
        if min_frames is not None:
            ev["min_frames"] = min_frames
    return sop_def


def rename_event(sop_def: dict[str, Any], old_id: str, new_id: str) -> bool:
    """イベントidを変更する(宣言順は保持)。

    変更したらTrue、old_idが無い/同名ならFalse。衝突・不正idはValueError。
    注意: annotation JSONのキーはここでは触らない（呼び出し側が同期する）。
    既存のprediction runの回答キーは旧idのまま残る(runは不変の歴史記録)。
    """
    if old_id == new_id:
        return False
    if not new_id or not new_id.isidentifier():
        raise ValueError(f"イベントidが不正です(識別子のみ可): {new_id!r}")
    ev = get_event(sop_def, old_id)
    if ev is None:
        return False
    if get_event(sop_def, new_id) is not None:
        raise ValueError(f"イベントidが衝突します: {new_id}")
    ev["id"] = new_id
    return True


def delete_event(sop_def: dict[str, Any], event_id: str) -> bool:
    """イベントを削除する。削除したらTrue、元から無ければFalse。

    最後のイベントも削除できる。未着手の手動annotationでは空のeventsが正しい。
    """
    ev = get_event(sop_def, event_id)
    if ev is None:
        return False
    sop_def["events"] = [e for e in sop_def["events"] if e["id"] != event_id]
    return True
