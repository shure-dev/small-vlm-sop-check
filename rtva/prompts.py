"""プロンプト構築。SFT と推論で完全に同一のものを使う。

v3(スパン)タスク: 30秒スパンのうち前半20秒は既検出イベントのテキスト、
直近10秒だけが動画。出力はスパン全体(0-30秒)のイベント列。
"""

USER_PLAIN = "Analyze the clip."


def system_prompt_span(vocab: list[str], span: int, video_s: int, n_frames: int | None = None) -> str:
    text_s = span - video_s
    n_frames = n_frames or video_s
    stride = video_s // n_frames
    per = "1 frame per second" if stride == 1 else f"one frame every {stride} seconds"
    lines = "\n".join(f"- {v}" for v in vocab)
    return (
        "You are a real-time video analyst tracking a first-person task video. "
        f"You analyze a {span}-second span. For the first {text_s} seconds "
        f"(t=0 to {text_s}) you get TEXT: steps already detected earlier. For the last "
        f"{video_s} seconds (t={text_s} to {span}) you get VIDEO: {n_frames} frames "
        f"({per}, in chronological order).\n\n"
        "Possible steps in this video:\n" + lines + "\n\n"
        f"Report every step in the FULL span (t=0 to {span}), merging text and video: "
        "a step from the text may continue into the video (then report ONE event with "
        "its original start), or a new step may start in the video. Trust the video "
        "over the text when they disagree about the recent seconds.\n\n"
        "Reply with exactly one JSON object and nothing else:\n"
        '{"events": [{"start_s": <int>, "end_s": <int>, "step": "<step from the list>"}, ...]}\n'
        'If no step is present, reply {"events": []}.'
    )


def user_prompt_span(history_events: list[dict], before_label: str | None, video_s: int) -> str:
    """過去テキスト(スパン座標)+動画解析指示。history_events は end_s<=text_s に切ってある。"""
    if history_events:
        hist = "\n".join(f"{e['start_s']}-{e['end_s']}s: {e['step']}" for e in history_events)
    else:
        hist = "(no steps detected)"
    before = f"(before this span: {before_label})\n" if before_label else ""
    return (
        f"Earlier in this span:\n{before}{hist}\n\n"
        f"The last {video_s} seconds follow as frames. Report the full span."
    )


def system_prompt(vocab: list[str], w: int) -> str:
    lines = "\n".join(f"- {v}" for v in vocab)
    return (
        f"You are a real-time video analyst. You are given the last {w} seconds of a "
        f"first-person video as {w} frames (1 frame per second, in chronological order).\n\n"
        "Possible steps in this video:\n" + lines + "\n\n"
        "Report every step visible in this clip, with timestamps in seconds relative to "
        f"the clip start (0 to {w}). An ongoing step may start at 0 and/or end at {w}.\n\n"
        "Reply with exactly one JSON object and nothing else:\n"
        '{"events": [{"start_s": <int>, "end_s": <int>, "step": "<step from the list>"}, ...]}\n'
        'If no step is visible, reply {"events": []}.'
    )
