"""detect_events(回答ログ→イベント区間)の回帰テスト。

VLM(mlx_vlm)を必要としない — datasets/konro_inspection の固定回答ログは
実際にQwen3-VL-4Bが回答した本物のデータ(2026-07実行、runコマンドで生成)。
これに対して区間検出ロジックだけを検証するので、GPUもモデルダウンロードも不要でCIで回せる。
"""
from pathlib import Path

from small_vlm_sop_check.core.events import detect_events
from small_vlm_sop_check.core.sop import load_answer_log, load_sop

DATASET_DIR = Path(__file__).resolve().parents[2] / "datasets" / "konro_inspection"
SOP_PATH = DATASET_DIR / "sops" / "konro_inspection" / "konro_inspection.yaml"
ANSWER_LOG = DATASET_DIR / "fixtures" / "reference_outputs" / "answer_log.json"


def _detect(sop, frames):
    return detect_events(sop["events"], frames, sop.get("defaults"))


def test_reference_log_detects_expected_spans():
    """基準の回答ログ(Qwen3-VL-4B)から、5工程すべての区間が検出され、
    動画に存在しない動作(gloves_worn)は検出されない。"""
    sop = load_sop(SOP_PATH)
    events = _detect(sop, load_answer_log(ANSWER_LOG))

    spans = {name: (r.start_idx, r.end_idx) if r else None for name, r in events.items()}
    assert spans == {
        "ignite": (1, 5),
        "flame_seen": (3, 3),
        "point1": (4, 5),
        "grill_open": (7, 8),
        "point2": (10, 14),
        "battery_check": (12, 13),
        "gloves_worn": None,
    }


def test_all_no_answers_detect_nothing():
    """検出コードが「やった」とごまかさないこと: 全フレームnoならすべて未検出。"""
    sop = load_sop(SOP_PATH)
    frames = load_answer_log(ANSWER_LOG)
    for f in frames:
        f["answers"] = {q: "no" for q in f["answers"]}
    events = _detect(sop, frames)
    assert all(r is None for r in events.values())


def test_min_frames_filters_single_frame_runs():
    """min_frames:2 のイベントは1フレームだけの一致では検出されない。"""
    sop = load_sop(SOP_PATH)
    frames = load_answer_log(ANSWER_LOG)
    for f in frames:                       # knob==yes を idx=3 の1フレームだけに潰す
        f["answers"]["knob"] = "yes" if f["idx"] == 3 else "no"
    events = _detect(sop, frames)
    assert events["ignite"] is None


def test_occurrence_is_order_independent():
    """events の宣言順を入れ替えても、occurrence指定があれば結果は変わらない
    (occurrence未指定だと宣言順が結果を左右してしまう脆さへの回帰テスト)。
    """
    sop = load_sop(SOP_PATH)
    frames = load_answer_log(ANSWER_LOG)

    events = dict(sop["events"])
    # point2 を point1 より先に持ってくる(わざと逆順)
    reordered = {
        "point2": events["point2"], "point1": events["point1"],
        **{k: v for k, v in events.items() if k not in ("point1", "point2")},
    }

    r1 = detect_events(sop["events"], frames, sop.get("defaults"))
    r2 = detect_events(reordered, frames, sop.get("defaults"))
    assert r1["point1"].t == r2["point1"].t
    assert r1["point2"].t == r2["point2"].t
