"""デモ動画レンダリング: 左=動画再生 / 右=トークンストリーミング(実測タイミング再現)。

複数の記録(demo_record.py 出力)を連続で1本のmp4にする。
窓iの解析は動画時刻 (i+1)*W 秒(=窓の映像が揃った瞬間)に開始し、
トークンは実測 t_ms で出現。タイムスタンプは元動画の絶対秒で表示。
冒頭にコールドスタート(モデルロード実測時間)を正直に表示する。

使い方:
    python scripts/demo_render.py rec1.jsonl [rec2.jsonl ...] out.mp4 \
        [--label "SFT v2 (86 videos)"]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtva.goalstep import VIDEO_DIR

FPS = 30
W_CANVAS, H_CANVAS = 1280, 720
BG = (16, 20, 28)
PANEL = (24, 30, 42)
FG = (226, 232, 240)
DIM = (130, 140, 155)
BLUE = (90, 156, 224)
ORANGE = (235, 104, 52)
GREEN = (80, 200, 120)
RED = (240, 90, 90)


def font(size: int, mono: bool = False, bold: bool = False):
    cands = (["/System/Library/Fonts/Menlo.ttc"] if mono else []) + [
        "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc" if bold
        else "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for p in cands:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_HEAD = font(22)
F_LABEL = font(16)
F_RIGHT = font(20)          # 右パネル見出し・判定
F_MONO = font(19, mono=True)  # 右パネルのJSONストリーム
F_STAT = font(24, bold=True)  # TTFT等の速度指標(強調)
F_BIG = font(34, bold=True)


def wrap_mono(draw, text, width_px):
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=F_MONO) > width_px or ch == "\n":
            lines.append(cur)
            cur = "" if ch == "\n" else ch
        else:
            cur += ch
    lines.append(cur)
    return lines


def base_canvas(label, load_s, w=5):
    img = Image.new("RGB", (W_CANVAS, H_CANVAS), BG)
    d = ImageDraw.Draw(img)
    d.text((24, 14), f"rtva — W秒の動画をW秒以内に解析する (W={w})", font=F_HEAD, fill=FG)
    d.text((24, 44), f"{label}   Qwen3.5-0.8B 4bit on M4 Pro   cold start(model load): {load_s:.1f}s",
           font=F_LABEL, fill=DIM)
    return img, d


def render_recording(vw, rec_path, vid_ix, n_vids, label, load_s):
    rows = [json.loads(l) for l in open(rec_path)]
    meta = rows[0]["meta"]
    wins = rows[1:]
    w = meta["window"]
    uid, start = meta["uid"], meta["start"]
    n_win = len(wins)
    win_by_ix = {r["w_ix"]: r for r in wins}
    verdict_mark = {"ok": ("○", GREEN), "partial": ("△", ORANGE), "ng": ("×", RED)}

    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{uid}.mp4"))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30

    # 動画間スレート(1秒)
    for _ in range(FPS):
        img, d = base_canvas(label, load_s, w)
        d.text((W_CANVAS // 2 - 260, H_CANVAS // 2 - 24),
               f"VIDEO {vid_ix+1}/{n_vids}   {uid[:8]}   t={start}s〜", font=F_BIG, fill=BLUE)
        vw.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

    # 最後の窓の解析完了まで描く(+1秒の余韻)
    last = win_by_ix.get(n_win - 1)
    tail = (last["total_ms"] / 1000 if last else 0) + 1.0
    seg_dur = n_win * w + tail
    for fidx in range(int(seg_dur * FPS)):
        t = fidx / FPS
        vt = min(t, n_win * w - 1 / FPS)
        abs_t = start + vt
        img, d = base_canvas(label, load_s, w)

        # 左: 動画フレーム
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(abs_t * vfps))
        ok, frame = cap.read()
        if ok:
            fh, fw2 = frame.shape[:2]
            scale = min(560 / fw2, 340 / fh)
            frame = cv2.resize(frame, (int(fw2 * scale), int(fh * scale)))
            img.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (24, 84))
        d.text((24, 84 + 346), f"video {uid[:8]} ({vid_ix+1}/{n_vids})   t = {abs_t:6.1f}s",
               font=F_LABEL, fill=DIM)

        # 左下: 窓タイムライン(状態色: 青=取得中 / 橙=解析中 / マーク=結果)
        cell = 560 // n_win
        for i in range(n_win):
            x0, x1 = 24 + i * cell, 24 + (i + 1) * cell - 6
            r = win_by_ix.get(i)
            cap_s, cap_e = i * w, (i + 1) * w
            d.rectangle([x0, 486, x1, 508], fill=PANEL)
            if cap_s <= t < cap_e:  # 取得中: 進捗で塗る
                d.rectangle([x0, 486, x0 + int((x1 - x0) * (t - cap_s) / w), 508], fill=BLUE)
            elif t >= cap_e and r is not None:
                if (t - cap_e) * 1000 < r["total_ms"]:  # 解析中
                    d.rectangle([x0, 486, x1, 508], fill=ORANGE)
                else:  # 完了
                    d.rectangle([x0, 486, x1, 508], fill=(40, 52, 70))
                    mark, mcol = verdict_mark[r["verdict"]]
                    d.text(((x0 + x1) // 2 - 9, 512), mark, font=F_RIGHT, fill=mcol)
        d.text((24, 545), f"セル={w}秒窓   青=映像取得中 / 橙=解析中 / ○△×=結果", font=F_LABEL, fill=DIM)

        # 右: ストリーミングパネル(全窓のブロックを状態つきで表示)
        px, py, pw_ = 640, 84, 616
        d.rectangle([px - 12, py - 8, px + pw_ + 8, 700], fill=PANEL)
        y = py
        for i in range(n_win):
            r = win_by_ix.get(i)
            ws_abs = start + i * w
            cap_s, cap_e = i * w, (i + 1) * w  # この窓の映像取得区間(セグメント内時刻)
            hdr_col = DIM if t < cap_s else BLUE
            d.text((px, y), f"WINDOW [{ws_abs}-{ws_abs+w}s]", font=F_RIGHT, fill=hdr_col)

            if t < cap_s:  # 1) 待機中
                d.text((px + 250, y), "待機中", font=F_RIGHT, fill=DIM)
                y += 34
                continue
            if t < cap_e:  # 2) 映像取得中(バッファリング)
                frac = (t - cap_s) / w
                d.text((px + 250, y), f"映像取得中 {t-cap_s:.1f}/{w:.0f}s", font=F_RIGHT, fill=BLUE)
                y += 28
                d.rectangle([px, y, px + pw_ - 20, y + 10], outline=(60, 72, 95))
                d.rectangle([px, y, px + int((pw_ - 20) * frac), y + 10], fill=BLUE)
                y += 24
                continue
            if r is None:
                y += 34
                continue

            el_ms = (t - cap_e) * 1000
            done = el_ms >= r["total_ms"]
            if not done:  # 3) 解析中
                d.text((px + 250, y), f"解析中… {el_ms/1000:.2f}s / 締切{w:.0f}s",
                       font=F_RIGHT, fill=ORANGE)
            else:  # 4) 解析終了
                d.text((px + 250, y), f"解析完了 {r['total_ms']/1000:.2f}s (締切の{r['total_ms']/(w*1000)*100:.0f}%)",
                       font=F_RIGHT, fill=GREEN)
            y += 28
            text = "".join(tk["text"] for tk in r["tokens"] if tk["t_ms"] <= el_ms)
            for line in wrap_mono(d, text, pw_ - 8)[:3]:
                d.text((px, y), line, font=F_MONO, fill=FG)
                y += 25
            if r["ttft_ms"] is not None and el_ms >= r["ttft_ms"]:
                d.text((px, y), f"TTFT {r['ttft_ms']/1000:.2f}s", font=F_STAT, fill=ORANGE)
                if done:
                    n_tok = len(r["tokens"])
                    tps = n_tok / (r["total_ms"] / 1000) if r["total_ms"] else 0
                    d.text((px + 200, y), f"{tps:.0f} tok/s", font=F_STAT, fill=GREEN)
                y += 34
            if done:
                mark, mcol = verdict_mark[r["verdict"]]
                pred_s = "; ".join(f"{ws_abs+e['start_s']}-{ws_abs+e['end_s']}s {e['step']}"
                                   for e in r["pred"]) or "(イベントなし)"
                gold_s = "; ".join(f"{ws_abs+e['start_s']}-{ws_abs+e['end_s']}s {e['step']}"
                                   for e in r["gold"]) or "(イベントなし)"
                d.text((px, y), f"検出: {pred_s[:52]}", font=F_RIGHT, fill=FG)
                y += 26
                d.text((px, y), f"{mark} GT: {gold_s[:52]}", font=F_RIGHT, fill=mcol)
                y += 34
        vw.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    cap.release()
    return sum(1 for r in wins if r["verdict"] == "ok"), n_win


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="記録jsonl... 出力mp4")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    recs, out = args.files[:-1], args.files[-1]

    meta0 = json.loads(open(recs[0]).readline())["meta"]
    label = args.label or meta0["adapter"]
    load_s = meta0["model_load_ms"] / 1000
    w0 = meta0["window"]

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W_CANVAS, H_CANVAS))

    # コールドスタート表示(実測値、表示は最大3秒に短縮)
    for _ in range(int(min(load_s, 3.0) * FPS)):
        img, d = base_canvas(label, load_s, w0)
        d.text((W_CANVAS // 2 - 300, H_CANVAS // 2 - 24),
               f"コールドスタート: モデルロード中… (実測 {load_s:.1f}s)", font=F_BIG, fill=ORANGE)
        vw.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

    tot_ok = tot_n = 0
    for k, rp in enumerate(recs):
        n_ok, n = render_recording(vw, rp, k, len(recs), label, load_s)
        tot_ok += n_ok
        tot_n += n
    vw.release()
    print(f"saved {out} ({len(recs)}動画 {tot_n}窓中 ○{tot_ok})")


if __name__ == "__main__":
    main()
