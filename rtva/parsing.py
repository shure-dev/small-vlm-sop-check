"""モデル出力のパースと検証。壊れた出力で下流を汚染しない。"""

import json
import re

from .goalstep import normalize_label


def parse_events(text: str, vocab: set[str], w: int) -> dict:
    """出力テキスト → {"events": [...], "format_ok": bool, "out_of_vocab": int}。

    - コードフェンス等のノイズを除去して最初のJSONオブジェクトを抽出
    - スキーマ検証・範囲クランプ(0..w)・ラベル正規化・語彙照合
    - パース不能は events=[] (format_ok=False)
    """
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"events": [], "format_ok": False, "out_of_vocab": 0}
    try:
        obj = json.loads(m.group(0))
        raw = obj["events"]
        assert isinstance(raw, list)
    except Exception:
        return {"events": [], "format_ok": False, "out_of_vocab": 0}

    events, oov = [], 0
    for ev in raw:
        try:
            a = max(0, min(w, int(float(ev["start_s"]))))
            b = max(0, min(w, int(float(ev["end_s"]))))
            step = normalize_label(str(ev["step"]))
        except Exception:
            return {"events": [], "format_ok": False, "out_of_vocab": oov}
        if b <= a:
            continue
        if step not in vocab:
            oov += 1
            continue
        events.append({"start_s": a, "end_s": b, "step": step})
    return {"events": events, "format_ok": True, "out_of_vocab": oov}
