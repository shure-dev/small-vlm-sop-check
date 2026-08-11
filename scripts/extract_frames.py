"""manifest.json の全動画から 1fps でフレームを抽出する(再実行可・既存スキップ)。

出力: data/goalstep/frames/<uid>/f<秒:05d>.jpg
使い方: python scripts/extract_frames.py
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtva.goalstep import FRAME_DIR, VIDEO_DIR, duration_of, load_annotations, load_manifest


def extract(uid: str, T: int) -> int:
    out_dir = FRAME_DIR / uid
    out_dir.mkdir(parents=True, exist_ok=True)
    if sum(1 for _ in out_dir.glob("f*.jpg")) >= T:
        return 0
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{uid}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    saved, t, frame_ix = 0, 0, 0
    while t < T:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_ix == round(t * fps):
            cv2.imwrite(str(out_dir / f"f{t:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            saved += 1
            t += 1
        frame_ix += 1
    cap.release()
    return saved


def main() -> None:
    ann = load_annotations()
    total = 0
    for m in load_manifest():
        n = extract(m["uid"], duration_of(ann[m["uid"]]))
        total += n
        print(f"{m['uid'][:8]}: {n}枚抽出")
    print(f"合計 {total}枚")


if __name__ == "__main__":
    main()
