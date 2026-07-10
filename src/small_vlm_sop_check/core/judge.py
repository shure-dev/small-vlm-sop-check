"""宣言的SOP定義 × 回答ログ → 遵守判定（決定論的・ルールベース）。

設計思想（experiments/sop_step_detect/confidence_judge/ での実験で確立）:
  質問への回答はVLMにやらせるが、それを手順書と突き合わせて「守られているか」を
  判定する部分は、VLM(自然文推論)に投げるとよくある単純な数値・順序比較すら
  間違えることが実験で確認された。したがって判定は常にこの決定的なコードで行う。

正解条件は「events(何を検出するか)」と「relations(それらの時間的関係)」に分離する:
  events   : 名前 -> 検出条件(evidence式)。同じ質問を複数eventが参照してもよい
             (例: 指差しを2回する手順は同じ質問を2つのeventで参照し、occurrenceで区別)
  relations: "A before B" / "A overlaps B" / "not A" の3種類だけで表現する
             - before      : Aの代表時刻がBの代表時刻より前(order_tolerance_s の許容あり)
             - overlaps    : Aの検出区間とBの検出区間が重なっている(「同時でよい」「この区間内の
                              どこかで一度起きればよい」の両方をこれ1つで表現できる)
             - not_overlaps: 同時に起きてはいけない
             - not A       : Aは一度も検出されてはいけない(安全条件・禁止工程の検出)
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


# ---------------------------------------------------------------------------
# 関係(relations)の評価
# ---------------------------------------------------------------------------

REL_RE = re.compile(r"^\s*(\w+)\s+(before|overlaps|not_overlaps)\s+(\w+)\s*$")
NOT_RE = re.compile(r"^\s*not\s+(\w+)\s*$")


def not_only_events(relation_strs: list[str]) -> set[str]:
    """'not X' でのみ参照されるイベント名の集合(=検出されない方が正しいイベント)。"""
    return {NOT_RE.match(r).group(1) for r in relation_strs if NOT_RE.match(r)}


def check_relations(relation_strs: list[str], events: dict[str, Run | None],
                     tolerance_s: float = 0.0, gap_tolerance_s: float = 0.0) -> list[dict]:
    """relations(文字列のリスト)を評価し、違反の詳細dictのリストを返す。空なら違反なし。

    各違反 = {"kind", "relation", "events":[...], "message"}。message は人間向け文字列(不変)。
    kind: order_reversed / overlap_missing / overlap_forbidden / forbidden / missing
      - missing は「関係の一方が未検出で評価できない」= coverage起因のFAIL。
        「順序を実際に取り違えた(order_reversed)」とは別物として区別できるようにする。
    """
    violations = []
    for rel in relation_strs:
        m_not = NOT_RE.match(rel)
        if m_not:
            name = m_not.group(1)
            if events.get(name) is not None:
                violations.append({"kind": "forbidden", "relation": rel, "events": [name],
                    "message": f"「{name}」は検出されてはいけないが、t={events[name].t}sで検出された"})
            continue

        m = REL_RE.match(rel)
        if not m:
            raise ValueError(f"未対応のrelation式: {rel!r}")
        a_name, op, b_name = m.groups()
        a, b = events.get(a_name), events.get(b_name)
        if a is None or b is None:
            missing = a_name if a is None else b_name
            violations.append({"kind": "missing", "relation": rel, "events": [missing],
                "message": f"「{missing}」が検出されていないため関係「{rel}」を評価できない"})
            continue

        if op == "before":
            if b.t < a.t - tolerance_s:
                violations.append({"kind": "order_reversed", "relation": rel, "events": [a_name, b_name],
                    "message": f"「{a_name}」(t={a.t}s)の後に「{b_name}」(t={b.t}s)"
                               f"が来るはずが、検出順序は逆だった"})
        elif op == "overlaps":
            overlap = not (a.end_idx < b.start_idx or b.end_idx < a.start_idx)
            if not overlap and gap_tolerance_s > 0:
                gap = max(a.start_idx, b.start_idx) - min(a.end_idx, b.end_idx)
                overlap = gap <= gap_tolerance_s
            if not overlap:
                violations.append({"kind": "overlap_missing", "relation": rel, "events": [a_name, b_name],
                    "message": f"「{a_name}」(t={a.t}s)と「{b_name}」(t={b.t}s)は"
                               f"重なっているはずだが、検出区間が離れていた"})
        elif op == "not_overlaps":
            overlap = not (a.end_idx < b.start_idx or b.end_idx < a.start_idx)
            if overlap:
                violations.append({"kind": "overlap_forbidden", "relation": rel, "events": [a_name, b_name],
                    "message": f"「{a_name}」と「{b_name}」は同時に起きてはいけないが、重なっていた"})
    return violations


# ---------------------------------------------------------------------------
# トップレベルAPI
# ---------------------------------------------------------------------------

@dataclass
class JudgeResult:
    events: dict[str, Run | None]
    coverage: float
    violations: list[str]           # 人間向けメッセージ(後方互換)
    verdict: str                    # "PASS" | "FAIL"
    violation_details: list[dict]   # {"kind","relation","events","message"} — 理由照合用


def judge(sop_def: dict[str, Any], frames: list[dict]) -> JudgeResult:
    """SOP定義(events/relations/defaults を含む dict)とフレームごとの回答列から判定する。

    frames: [{"idx": int, "t": float, "answers": {question_id: value}}, ...]
            answers の値は "yes"/"no"/"unclear" 等の文字列(信頼度から argmax を取ったもの、
            またはVLMの生JSON出力をそのまま使ってもよい)。
    """
    defaults = sop_def.get("defaults", {})
    tolerance_s = defaults.get("order_tolerance_s", 0.0)
    relations = sop_def.get("relations", [])

    events = detect_events(sop_def["events"], frames, defaults)
    details = check_relations(relations, events, tolerance_s=tolerance_s)
    violations = [d["message"] for d in details]

    excluded = not_only_events(relations)
    required = {k: v for k, v in events.items() if k not in excluded}
    n_done = sum(1 for v in required.values() if v is not None)
    coverage = n_done / len(required) if required else 1.0

    verdict = "PASS" if (coverage == 1.0 and not violations) else "FAIL"
    return JudgeResult(events=events, coverage=coverage, violations=violations,
                       verdict=verdict, violation_details=details)


def check_expectation(sop_def: dict[str, Any], result: JudgeResult) -> dict | None:
    """SOPに `expect` があれば「verdict」と「なぜ違反か(理由)」の一致を採点する。

    expect:
      verdict: PASS | FAIL
      because:                       # FAIL時、当てるべき違反理由(なくてもよい)
        - relation: "A before B"     # relation式で指定 (順序・重なり系)
          kind: order_reversed
        - event: gloves_check        # or event名で指定 (missing系)
          kind: missing

    戻り値: None(expect未定義) or
      {"verdict_ok": bool, "reasons": [{... , "caught": bool}], "localized": bool}
      localized = verdict一致 かつ 期待した理由をすべて(正しいkindで)当てた。
    """
    exp = sop_def.get("expect")
    if not exp:
        return None
    verdict_ok = (result.verdict == exp.get("verdict"))
    reasons = []
    for want in exp.get("because", []):
        kind = want.get("kind")
        if "relation" in want:
            caught = any(d["kind"] == kind and d["relation"] == want["relation"]
                         for d in result.violation_details)
        else:
            ev = want.get("event")
            caught = any(d["kind"] == kind and ev in d["events"]
                         for d in result.violation_details)
        reasons.append({**want, "caught": caught})
    localized = verdict_ok and all(r["caught"] for r in reasons)
    return {"verdict_ok": verdict_ok, "reasons": reasons, "localized": localized}
