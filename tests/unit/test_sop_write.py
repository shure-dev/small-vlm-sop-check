"""save_sop と SOP編集ヘルパー(annotatorがブラウザから呼ぶ)の回帰テスト。

VLMもブラウザも不要。dict変換とYAML書き戻しの決定性だけを検証する。
現行SOP契約は events = [{id, ask, values, min_frames?}, ...]。
"""
import copy

import pytest

from small_vlm_sop_check.core.sop import (
    delete_event, get_event, load_sop, rename_event, save_sop, set_domain_hint,
    upsert_event, validate_sop,
)


def _base_sop():
    return {
        "sop": {"id": "demo", "name": "Demo", "domain_hint": "工場の一人称視点"},
        "events": [
            {"id": "assemble", "ask": "組み立てているか？", "values": ["yes", "no"], "min_frames": 4},
            {"id": "wrap", "ask": "袋を丸めているか？", "values": ["yes", "no"], "min_frames": 2},
        ],
        "benchmark": {"status": "provisional"},
    }


def test_save_sop_roundtrip_and_yes_no_quoting(tmp_path):
    """save→loadで同値に戻り、yes/no はブール化せず文字列のまま残る。"""
    sop = _base_sop()
    path = tmp_path / "sop" / "sop.yaml"
    save_sop(path, sop)

    text = path.read_text(encoding="utf-8")
    assert "'yes'" in text and "'no'" in text          # クォートされている
    reloaded = load_sop(path)
    assert reloaded == sop
    assert reloaded["events"][0]["values"] == ["yes", "no"]
    assert all(isinstance(v, str) for v in reloaded["events"][0]["values"])


def test_save_sop_preserves_unknown_blocks(tmp_path):
    """benchmarkブロック等の未知キーは書き戻しても保持される。"""
    sop = _base_sop()
    path = tmp_path / "sop.yaml"
    save_sop(path, sop)
    assert load_sop(path)["benchmark"] == {"status": "provisional"}


def test_save_sop_is_atomic_no_tmp_left(tmp_path):
    """書き込み後に .tmp が残らない(原子的置き換え)。"""
    path = tmp_path / "sop.yaml"
    save_sop(path, _base_sop())
    assert not list(tmp_path.glob("*.tmp"))


def test_save_sop_accepts_empty_manual_draft_and_rejects_duplicate_id(tmp_path):
    """未着手annotationはevents空を許し、id重複は拒否する。"""
    sop = _base_sop()
    sop["events"] = []
    save_sop(tmp_path / "empty.yaml", sop)
    assert load_sop(tmp_path / "empty.yaml")["events"] == []
    dup = _base_sop()
    dup["events"].append({"id": "wrap", "ask": "重複"})
    with pytest.raises(ValueError):
        save_sop(tmp_path / "dup.yaml", dup)


def test_validate_rejects_unsupported_questions_key():
    unsupported = {"sop": {"id": "x", "name": "X"},
              "questions": [{"id": "q", "ask": "?", "values": ["yes", "no"]}],
              "events": {"e": {"evidence": "q==yes"}}}
    with pytest.raises(ValueError, match="questionsは未対応"):
        validate_sop(unsupported)


def test_load_sop_fills_defaults(tmp_path):
    """ask/values を省略したイベントは読み込み時に既定値で埋まる。"""
    (tmp_path / "s.yaml").write_text(
        "sop: {id: x, name: X}\nevents:\n- id: only_id\n", encoding="utf-8")
    sop = load_sop(tmp_path / "s.yaml")
    assert sop["events"][0] == {"id": "only_id", "ask": "", "values": ["yes", "no"]}


def test_set_domain_hint_set_and_clear():
    sop = _base_sop()
    set_domain_hint(sop, "  新しいヒント  ")
    assert sop["sop"]["domain_hint"] == "新しいヒント"    # 前後空白は除去
    set_domain_hint(sop, "")
    assert "domain_hint" not in sop["sop"]                # 空文字でキーごと削除


def test_upsert_event_creates_event():
    """新規イベントは id/ask/values(yes/no) を持つ形で末尾に追加される。"""
    sop = _base_sop()
    upsert_event(sop, "inspect", ask="検査しているか？", min_frames=3)
    assert sop["events"][-1] == {
        "id": "inspect", "ask": "検査しているか？", "values": ["yes", "no"], "min_frames": 3,
    }


def test_upsert_event_updates_existing():
    """既存イベントの編集は ask/min_frames だけ更新し、他フィールドと宣言順を保つ。"""
    sop = _base_sop()
    upsert_event(sop, "assemble", ask="組み立て中？", min_frames=6)
    assert [ev["id"] for ev in sop["events"]] == ["assemble", "wrap"]
    ev = get_event(sop, "assemble")
    assert ev["ask"] == "組み立て中？"
    assert ev["min_frames"] == 6
    assert ev["values"] == ["yes", "no"]


def test_upsert_event_rejects_bad_id():
    with pytest.raises(ValueError):
        upsert_event(_base_sop(), "not an id", ask="x")


def test_rename_event_preserves_order():
    """id変更は宣言順を保つ。GTキーの追随は呼び出し側(annotator)が行う。"""
    sop = _base_sop()
    assert rename_event(sop, "wrap", "roll_bag") is True
    assert [ev["id"] for ev in sop["events"]] == ["assemble", "roll_bag"]
    assert get_event(sop, "roll_bag")["ask"] == "袋を丸めているか？"


def test_rename_event_collision_and_bad_id_rejected():
    sop = _base_sop()
    with pytest.raises(ValueError):
        rename_event(sop, "wrap", "assemble")      # イベントid衝突
    with pytest.raises(ValueError):
        rename_event(sop, "wrap", "not an id")     # 不正id
    assert rename_event(sop, "nope", "x") is False  # 無いidはFalse
    assert rename_event(sop, "wrap", "wrap") is False  # 同名は何もしない


def test_delete_event():
    sop = _base_sop()
    assert delete_event(sop, "wrap") is True
    assert [ev["id"] for ev in sop["events"]] == ["assemble"]
    assert delete_event(sop, "nope") is False       # 無いidはFalse
    assert delete_event(sop, "assemble") is True
    assert sop["events"] == []


def test_validate_sop_accepts_base():
    assert validate_sop(copy.deepcopy(_base_sop())) is not None
