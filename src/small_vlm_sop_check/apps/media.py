"""注釈UIで動画を表示するためのローカル媒体処理。"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .catalog import Unit


def preview_video(unit: Unit) -> Path | None:
    """元動画を優先し、無ければ抽出フレームからGit管理外previewを生成する。"""
    direct = unit.video_path
    if direct is not None and direct.is_file():
        return direct
    frames = unit.frame_paths()
    if not frames:
        return None
    if unit.meta_path is not None:
        dataset_root = unit.meta_path.parents[2]
        repo_root = dataset_root.parents[1]
        output = (
            repo_root / "data" / unit.dataset / "units" / unit.unit_id
            / ".annotation-preview.mp4"
        )
    else:
        output = unit.frames_dir.parent / ".annotation-preview.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    signature_path = output.with_suffix(".json")
    signature = {
        "fps": unit.fps,
        "frames": [f"{path.name}:{path.stat().st_size}:{path.stat().st_mtime_ns}" for path in frames],
    }
    if output.is_file() and signature_path.is_file():
        try:
            if json.loads(signature_path.read_text(encoding="utf-8")) == signature:
                return output
        except (OSError, json.JSONDecodeError):
            pass
    pattern = _frame_pattern(frames)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(unit.fps),
        "-i", str(pattern), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("動画previewの生成にはffmpegが必要です") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"動画previewを生成できません: {exc.stderr.strip()}") from exc
    signature_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _frame_pattern(frames: list[Path]) -> Path:
    first = frames[0]
    digits = len(first.stem.removeprefix("f"))
    expected = [f"f{index:0{digits}d}.jpg" for index in range(len(frames))]
    if [path.name for path in frames] != expected:
        raise RuntimeError("preview生成にはf0000.jpg形式の連番フレームが必要です")
    return first.parent / f"f%0{digits}d.jpg"
