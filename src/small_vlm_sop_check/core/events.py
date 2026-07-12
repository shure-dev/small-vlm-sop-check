"""宣言的SOP定義 × 回答ログ → イベント区間の検出（決定論的・ルールベース）。

設計思想（experiments/sop_step_detect/confidence_judge/ での実験で確立）:
  質問への回答はVLMにやらせるが、回答列から「いつ何が起きたか(区間)」を導出する
  部分は、VLM(自然文推論)に投げるとよくある単純な数値・順序比較すら
  間違えることが実験で確認された。したがって導出は常にこの決定的なコードで行う。

events: 名前 -> 検出条件(evidence式)。同じ質問を複数eventが参照してもよい
        (例: 指差しを2回する手順は同じ質問を2つのeventで参照し、occurrenceで区別)
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# evidence式のパース・評価
# ---------------------------------------------------------------------------

def parse_clauses(expr: str) -> list[tuple[str, str]]:
    """'question==value [and question2==value2 ...]' を [(question,value), ...] にパースする。"""
    clauses = []
    for part in re.split(r"\s+and\s+", expr.strip()):
        m = re.match(r"\s*(\w+)\s*==\s*(\w+)\s*$", part)
        if not m:
            raise ValueError(f"未対応のevidence式: {part!r} (元: {expr!r})")
        clauses.append((m.group(1), m.group(2)))
    return clauses


def frame_matches(answers: dict[str, str], clauses: list[tuple[str, str]]) -> bool:
    return all(answers.get(question, "unclear") == val for question, val in clauses)


@dataclass
class Run:
    start_idx: int
    end_idx: int
    t: float       # 区間内フレームの時刻の平均(代表時刻)
    hits: int
    idxs: tuple[int, ...] | None = None  # 実際に一致したフレームidx(max_gapで橋渡しした隙間は含まない)。
                                         # None = 不明(正解区間のように連続とみなす場合)


def find_runs(frames: list[dict], clauses: list[tuple[str, str]],
              min_frames: int = 1, max_gap: int = 0) -> list[Run]:
    """value一致の極大区間を探す。不一致(unclear/反対値)を max_gap フレームまで橋渡しする
    (VLM推論の孤立した誤検出・ブレを吸収するノイズ耐性)。
    frames: [{"idx": int, "t": float, "answers": {question_id: value, ...}}, ...]
    """
    runs, cur_hits, gap = [], [], 0
    for f in frames:
        if frame_matches(f["answers"], clauses):
            cur_hits.append((f["idx"], f["t"]))
            gap = 0
        elif cur_hits and gap < max_gap:
            gap += 1
        else:
            if cur_hits:
                runs.append(cur_hits)
            cur_hits, gap = [], 0
    if cur_hits:
        runs.append(cur_hits)

    out = []
    for r in runs:
        if len(r) < min_frames:
            continue
        idxs = [i for i, _ in r]
        ts = [t for _, t in r]
        out.append(Run(start_idx=min(idxs), end_idx=max(idxs),
                        t=round(sum(ts) / len(ts), 2), hits=len(r), idxs=tuple(idxs)))
    return out


# ---------------------------------------------------------------------------
# イベント検出
# ---------------------------------------------------------------------------

def detect_events(event_defs: dict[str, Any], frames: list[dict],
                   defaults: dict[str, Any] | None = None) -> dict[str, Run | None]:
    """events定義(YAMLの `events:` セクション)を評価し、各イベント名 -> Run|None を返す。

    同じevidenceを複数eventが参照する場合の割り当て方法は2通り:
      - occurrence: N を明示 -> 時系列N番目(1始まり)を宣言順に関係なく採用(堅牢・推奨)
      - 未指定     -> YAMLの宣言順に処理し、未使用の最初の区間を早い者勝ちで取る
                      (宣言順を入れ替えると結果も変わる点に注意。挙動を固定したいなら
                      occurrenceを明示すること)
    """
    defaults = defaults or {}
    claimed: dict[str, list[tuple[int, int]]] = {}
    runs_cache: dict[tuple, list[Run]] = {}
    result: dict[str, Run | None] = {}

    def runs_for(evidence, clauses, min_frames, max_gap):
        key = (evidence, min_frames, max_gap)
        if key not in runs_cache:
            runs_cache[key] = find_runs(frames, clauses, min_frames=min_frames, max_gap=max_gap)
        return runs_cache[key]

    for name, spec in event_defs.items():
        if isinstance(spec, str):
            spec = {"evidence": spec}
        evidence = spec["evidence"]
        clauses = parse_clauses(evidence)
        min_frames = spec.get("min_frames", defaults.get("min_frames", 2))
        max_gap = spec.get("max_gap_frames", defaults.get("max_gap_frames", 2))
        occurrence = spec.get("occurrence")
        runs = runs_for(evidence, clauses, min_frames, max_gap)

        if occurrence is not None:
            chosen = runs[occurrence - 1] if 0 < occurrence <= len(runs) else None
        else:
            used = claimed.setdefault(evidence, [])
            chosen = None
            for r in runs:
                span = (r.start_idx, r.end_idx)
                if not any(a <= span[1] and span[0] <= b for a, b in used):
                    chosen = r
                    used.append(span)
                    break
        result[name] = chosen
    return result
