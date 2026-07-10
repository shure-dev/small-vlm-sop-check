"""動画から一定fpsでフレームをJPEGとして書き出す。

Mac(Apple Silicon)注意: mujoco系ではなくcv2だけなのでPATH調整等は不要。
"""
from __future__ import annotations
import glob
import os
import cv2


def extract_frames(video_path: str, out_dir: str, fps: float = 1.0, width: int = 640) -> list[dict]:
    """video_path から fps 間隔でフレームを抽出し、out_dir に f000.jpg... として保存する。
    戻り値: [{"idx": int, "t": float, "path": str}, ...]
    """
    os.makedirs(out_dir, exist_ok=True)
    for p in glob.glob(os.path.join(out_dir, "*.jpg")):
        os.remove(p)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けませんでした: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / video_fps if video_fps else 0.0

    meta = []
    idx, t = 0, 0.0
    while t < duration:
        frame_no = min(int(round(t * video_fps)), n_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > width:
            frame = cv2.resize(frame, (width, int(h * width / w)))
        path = os.path.join(out_dir, f"f{idx:03d}.jpg")
        cv2.imwrite(path, frame)
        meta.append({"idx": idx, "t": round(t, 2), "path": path})
        idx += 1
        t += 1.0 / fps
    cap.release()
    return meta
