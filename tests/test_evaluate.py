"""evaluate(正解アノテーションとの突き合わせ)の回帰テスト。VLM不要。

観察ログは test_judge.py と同じ実データ(Qwen3-VL-4B)。正解アノテーションは
「人間が付けたらこうなる」体の合成フィクスチャで、基準検出(ignite 1-5 / flame 3-3 /
point1 4-5 / grill 7-8 / point2 10-14 / battery 12-13 / gloves なし)と境界を
わざと1〜2フレームずらしてある。境界のズレがtIoUには出るが関係の保存・合否には
影響しない、という設計上の主張をここで固定する。
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sop import load_sop, load_answer_log
from judge import Run
from evaluate import evaluate, tiou, load_ground_truth

EXAMPLE_DIR = Path(__file__).parent.parent / "examples" / "konro_inspection"
ANSWER_LOG = EXAMPLE_DIR / "sample_output" / "answer_log.json"


def make_gt(events: dict) -> dict:
    return {"schema_version": "0.1", "sop_id": "konro_inspection",
            "fps": 1.0, "n_frames": 16, "events": events}


GT_EVENTS = {
    "ignite":        {"start_idx": 1, "end_idx": 4},    # 検出1-5 -> tIoU 0.8
    "flame_seen":    {"start_idx": 3, "end_idx": 3},    # 一致 -> 1.0
    "point1":        {"start_idx": 4, "end_idx": 5},    # 一致 -> 1.0
    "grill_open":    {"start_idx": 6, "end_idx": 8},    # 検出7-8 -> 2/3
    "point2":        {"start_idx": 10, "end_idx": 13},  # 検出10-14 -> 0.8
    "battery_check": {"start_idx": 12, "end_idx": 13},  # 一致 -> 1.0
    "gloves_worn":   None,                              # 起きていない(注釈済み)
}


def test_tiou_math():
    r = lambda s, e: Run(start_idx=s, end_idx=e, t=0, hits=e - s + 1)
    assert tiou(r(1, 5), r(1, 5)) == 1.0
    assert tiou(r(0, 2), r(5, 8)) == 0.0
    assert tiou(r(1, 4), r(1, 5)) == 0.8       # 4/5
    assert tiou(r(6, 8), r(7, 8)) == 0.667     # 2/3


def test_boundary_shift_reduces_tiou_but_preserves_relations():
    """境界が1〜2フレームずれた注釈: tIoUは1.0を切るが、関係の保存・verdictは満点。
    「成功条件は関係とverdictで定義し、tIoUは診断」という切り分けの根拠。"""
    sop = load_sop(EXAMPLE_DIR / "sop.yaml")
    ev = evaluate(sop, make_gt(GT_EVENTS), load_answer_log(ANSWER_LOG))
    s = ev["summary"]

    by_name = {r["event"]: r for r in ev["events"]}
    assert by_name["ignite"]["tiou"] == 0.8
    assert by_name["grill_open"]["tiou"] == 0.667
    assert by_name["gloves_worn"]["status"] == "true_absent"
    assert all(r["status"] in ("match", "true_absent") for r in ev["events"])

    assert s["mean_tiou"] < 1.0                      # 境界ズレはtIoUに出る
    assert s["tiou@0.3"] == s["n_gt_present"] == 6   # だが全部ゆるいしきい値は超える
    assert s["relation_agree"] == s["relation_total"] == 6   # 関係は完全に保存
    assert s["detected_verdict"] == s["gt_verdict"] == s["expect_verdict"] == "PASS"


def test_miss_and_false_detection_statuses():
    """注釈と検出が食い違うケース: 見逃し(miss)・誤検出(false_detection)が
    区別され、関係の不一致・gt_verdictのFAILとして表面化する。"""
    sop = load_sop(EXAMPLE_DIR / "sop.yaml")
    events = dict(GT_EVENTS)
    events["gloves_worn"] = {"start_idx": 0, "end_idx": 1}   # 起きたと注釈(検出は無し)
    events["battery_check"] = None                           # 起きてないと注釈(検出は有り)
    ev = evaluate(sop, make_gt(events), load_answer_log(ANSWER_LOG))

    by_name = {r["event"]: r for r in ev["events"]}
    assert by_name["gloves_worn"]["status"] == "miss"
    assert by_name["battery_check"]["status"] == "false_detection"

    by_rel = {r["relation"]: r for r in ev["relations"]}
    assert by_rel["not gloves_worn"]["agree"] is False           # GTでは違反・検出は成立
    assert by_rel["point2 overlaps battery_check"]["agree"] is False
    assert ev["summary"]["gt_verdict"] == "FAIL"
    assert ev["summary"]["detected_verdict"] == "PASS"


def test_unannotated_event_is_excluded():
    """キーごと無い(未注釈)イベントは no_gt となり、それを含む関係は評価不能(None)。"""
    sop = load_sop(EXAMPLE_DIR / "sop.yaml")
    events = {k: v for k, v in GT_EVENTS.items() if k != "flame_seen"}
    ev = evaluate(sop, make_gt(events), load_answer_log(ANSWER_LOG))

    by_name = {r["event"]: r for r in ev["events"]}
    assert by_name["flame_seen"]["status"] == "no_gt"
    by_rel = {r["relation"]: r for r in ev["relations"]}
    assert by_rel["flame_seen overlaps ignite"]["agree"] is None
    assert ev["summary"]["relation_total"] == 5


def test_frame_level_diagnostics_derived_from_gt():
    """(question, value)ごとのフレーム診断が正解区間から導出される。
    point1/point2は同じ質問(pointing==yes)なので正例は両区間の和集合になる。"""
    sop = load_sop(EXAMPLE_DIR / "sop.yaml")
    ev = evaluate(sop, make_gt(GT_EVENTS), load_answer_log(ANSWER_LOG))
    rows = {(r["question"], r["value"]): r for r in ev["frames"]}

    assert rows[("pointing", "yes")]["gt_frames"] == 2 + 4     # point1(4-5) + point2(10-13)
    assert rows[("gloves", "yes")]["gt_frames"] == 0           # null注釈 -> 正例なし
    for r in ev["frames"]:
        for key in ("precision", "recall"):
            assert r[key] is None or 0.0 <= r[key] <= 1.0


def test_load_ground_truth_rejects_bad_span(tmp_path):
    bad = make_gt({"ignite": {"start_idx": 5, "end_idx": 2}})
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_ground_truth(p)
