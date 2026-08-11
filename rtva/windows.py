"""窓の切り出し・分類・履歴テキスト・窓結果のマージ。

タイムライン(1秒解像度のラベル列)を W秒の窓に切り、
相対タイムスタンプのイベント区間リストと相互変換する。
"""


def clip_events(tl: list, w_start: int, w: int) -> list[dict]:
    """タイムラインを窓 [w_start, w_start+w) で切って相対区間リストに。"""
    events = []
    t = w_start
    end = min(w_start + w, len(tl))
    while t < end:
        e = t
        while e < end and tl[e] == tl[t]:
            e += 1
        if tl[t] is not None:
            events.append({"start_s": t - w_start, "end_s": e - w_start, "step": tl[t]})
        t = e
    return events


def categorize(events: list[dict], w: int) -> str:
    """窓の種類: multi(遷移2+) / boundary(遷移1) / single(1step継続) / gap(無ラベル混在) / empty"""
    if not events:
        return "empty"
    n_boundaries = sum(1 for ev in events if ev["start_s"] > 0) + sum(
        1 for ev in events if ev["end_s"] < w
    )
    covered = sum(ev["end_s"] - ev["start_s"] for ev in events)
    if len(events) == 1 and events[0]["start_s"] == 0 and events[0]["end_s"] == w:
        return "single"
    if covered < w and len(events) >= 1:
        return "gap" if n_boundaries <= 1 else "multi"
    return "multi" if len(events) >= 3 or n_boundaries >= 3 else "boundary"


def has_boundary(events: list[dict], w: int) -> bool:
    return categorize(events, w) in ("boundary", "multi", "gap")


def history_text(tl: list, w_start: int, horizon: int = 60) -> str:
    """条件B用: 窓より前 horizon 秒のイベント要約(絶対秒表記)。"""
    lo = max(0, w_start - horizon)
    events = clip_events(tl, lo, w_start - lo)
    if not events:
        return "history (before this clip): none"
    lines = [f"{ev['start_s'] + lo}-{ev['end_s'] + lo}s: {ev['step']}" for ev in events]
    return "history (before this clip):\n" + "\n".join(lines)


def events_to_secondly(events: list[dict], w: int) -> list:
    """相対イベント区間 → 秒ごとのラベル列(長さw、無ラベルはNone)。重複は先勝ち。"""
    tl = [None] * w
    for ev in sorted(events, key=lambda e: e["start_s"]):
        for t in range(max(0, int(ev["start_s"])), min(w, int(ev["end_s"]))):
            if tl[t] is None:
                tl[t] = ev["step"]
    return tl


def merge_windows(window_results: list[tuple[int, list[dict]]], w: int) -> list[dict]:
    """非重複窓の予測結果 [(w_start, events), ...] を絶対時刻のタイムラインにマージ。

    窓境界で接する同一ラベル区間は結合する(窓をまたいで続くstepの復元)。
    """
    segs = []
    for w_start, events in sorted(window_results):
        for ev in sorted(events, key=lambda e: e["start_s"]):
            a, b = w_start + ev["start_s"], w_start + ev["end_s"]
            if segs and segs[-1]["step"] == ev["step"] and abs(segs[-1]["end_s"] - a) <= 1:
                segs[-1]["end_s"] = b
            else:
                segs.append({"start_s": a, "end_s": b, "step": ev["step"]})
    return segs
