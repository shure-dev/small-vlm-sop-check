"""judge の回帰テスト。

VLM(mlx_vlm)を必要としない — datasets/konro_inspection の固定回答ログは
実際にQwen3-VL-4Bが回答した本物のデータ(2026-07実行、runコマンドで生成)。
これに対してjudgeロジックだけを検証するので、GPUもモデルダウンロードも不要でCIで回せる。
"""
from pathlib import Path

from small_vlm_sop_check.core.judge import check_expectation, judge
from small_vlm_sop_check.core.sop import load_answer_log, load_sop

DATASET_DIR = Path(__file__).resolve().parents[2] / "datasets" / "konro_inspection"
SOP_DIR = DATASET_DIR / "sops" / "konro_inspection"
ANSWER_LOG = DATASET_DIR / "fixtures" / "reference_outputs" / "answer_log.json"


def test_correct_sop_passes():
    sop = load_sop(SOP_DIR / "correct.yaml")
    frames = load_answer_log(ANSWER_LOG)
    result = judge(sop, frames)
    assert result.verdict == "PASS", result.violations
    assert result.coverage == 1.0
    assert result.violations == []


def test_wrong_order_sop_fails_with_correct_reason():
    sop = load_sop(SOP_DIR / "wrong_order.yaml")
    frames = load_answer_log(ANSWER_LOG)
    result = judge(sop, frames)
    assert result.verdict == "FAIL"
    assert any("battery_check" in v and "ignite" in v for v in result.violations)


def test_missing_step_sop_fails_as_not_detected():
    sop = load_sop(SOP_DIR / "missing_step.yaml")
    frames = load_answer_log(ANSWER_LOG)
    result = judge(sop, frames)
    assert result.verdict == "FAIL"
    assert result.events["gloves_check"] is None
    assert result.coverage < 1.0


def test_reference_localizes_violation_reason():
    """基準の回答ログ(Qwen3-VL-4B)は、3条件すべてで verdict だけでなく『なぜ違反か(理由)』も当てる。
    - 正解手順   : PASS
    - 順序違反   : battery_check before ignite が order_reversed で違反
    - ステップ欠落: gloves_check が missing(未検出)で違反
    """
    frames = load_answer_log(ANSWER_LOG)
    for name in ("correct.yaml", "wrong_order.yaml", "missing_step.yaml"):
        sop = load_sop(SOP_DIR / name)
        ev = check_expectation(sop, judge(sop, frames))
        assert ev is not None, name
        assert ev["localized"], (name, ev)


def test_missing_detection_is_not_counted_as_localized():
    """順序違反SOPで、電池を一度も検出しない回答ログは FAIL にはなるが『理由は当てていない』
    (順序を取り違えた order_reversed ではなく、未検出 missing 起因のFAILだから)。"""
    sop = load_sop(SOP_DIR / "wrong_order.yaml")
    frames = load_answer_log(ANSWER_LOG)
    for f in frames:                      # battery を全フレーム no に潰す
        f["answers"]["battery"] = "no"
    result = judge(sop, frames)
    ev = check_expectation(sop, result)
    assert result.verdict == "FAIL"
    assert ev["verdict_ok"] is True       # FAILではある
    assert ev["localized"] is False       # だが理由(順序逆転)は当てていない


def test_occurrence_is_order_independent():
    """events の宣言順を入れ替えても、occurrence指定があれば結果は変わらない
    (このセッションで見つかった脆さ: occurrence未指定だと宣言順が結果を左右してしまう)。
    """
    sop = load_sop(SOP_DIR / "correct.yaml")
    frames = load_answer_log(ANSWER_LOG)

    reordered = dict(sop)
    events = dict(sop["events"])
    # point2 を point1 より先に持ってくる(わざと逆順)
    reordered["events"] = {
        "point2": events["point2"], "point1": events["point1"],
        **{k: v for k, v in events.items() if k not in ("point1", "point2")},
    }

    r1 = judge(sop, frames)
    r2 = judge(reordered, frames)
    assert r1.events["point1"].t == r2.events["point1"].t
    assert r1.events["point2"].t == r2.events["point2"].t
    assert r1.verdict == r2.verdict == "PASS"
