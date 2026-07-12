"""イベント区間アノテーションツール(標準ライブラリのみ・自動保存)。

なにをするか:
  抽出済みフレームをブラウザに並べ、SOPの各イベントについて「実際に起きた区間」を
  開始・終了フレームのクリック2回で注釈する。付けた注釈は操作のたびに即
  ground_truth.json へ保存される(保存ボタンは無い)。途中で閉じても再開できる。

注釈するのは事実(いつ何が起きたか=区間)だけ。評価(tIoU・フレーム一致)は
`sop-check eval` が行う。

使い方:
  sop-annotate            # 同梱サンプルを注釈(既定値で全部埋まる)
  sop-annotate \
    --sop datasets/konro_inspection/sops/konro_inspection/konro_inspection.yaml \
    --frames-dir datasets/konro_inspection/units/konro_inspection/frames \
    --out datasets/konro_inspection/annotations/human-v001/konro_inspection.json

前提: フレームは extract.py が吐く連番jpg(f000.jpg, ...)で、t = idx / fps とみなす。
occurrenceを持つイベント(1回目/2回目など)は全出現をそれぞれ注釈すること。
"""
from __future__ import annotations
import argparse
import json
import os
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..core.sop import load_sop
from .resources import repository_root, template_text


ROOT = repository_root()
DEMO_ROOT = ROOT / "datasets" / "konro_inspection"
DEFAULT_SOP = DEMO_ROOT / "sops" / "konro_inspection" / "konro_inspection.yaml"
DEFAULT_FRAMES = DEMO_ROOT / "units" / "konro_inspection" / "frames"
DEFAULT_GT = DEMO_ROOT / "annotations" / "human-v001" / "konro_inspection.json"


def display_path(p: Path) -> str:
    """ヘッダー表示用の短いパス(カレントディレクトリ相対。外なら絶対のまま)。"""
    try:
        return str(p.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(p)


def build_page_data(sop_def: dict, frame_files: list[str], fps: float,
                    out_path: Path, gt_events: dict) -> dict:
    events = []
    for name, spec in sop_def["events"].items():
        evidence = spec if isinstance(spec, str) else spec["evidence"]
        occurrence = None if isinstance(spec, str) else spec.get("occurrence")
        events.append({"name": name, "evidence": evidence, "occurrence": occurrence})
    return {
        "sop": {"id": sop_def["sop"]["id"], "name": sop_def["sop"]["name"]},
        "events": events,
        "frames": frame_files,
        "times": [round(i / fps, 2) for i in range(len(frame_files))],
        "out_path": display_path(out_path),
        "gt_events": gt_events,   # 再開用: 保存済みの注釈
    }


class Annotator(ThreadingHTTPServer):
    """ハンドラから参照する状態(SOP・フレーム・保存先)を持つHTTPサーバ。"""

    def __init__(self, addr, sop_def, frames_dir: Path, frame_files: list[str],
                 fps: float, out_path: Path):
        super().__init__(addr, Handler)
        self.sop_def = sop_def
        self.frames_dir = frames_dir
        self.frame_files = frame_files
        self.fps = fps
        self.out_path = out_path

    def load_gt_events(self) -> dict:
        if self.out_path.exists():
            return json.loads(self.out_path.read_text(encoding="utf-8"))["events"]
        return {}

    def save(self, events: dict) -> None:
        """検証して原子的に書き込む(tmpに書いてから置き換え)。"""
        known = set(self.sop_def["events"])
        n = len(self.frame_files)
        for name, span in events.items():
            if name not in known:
                raise ValueError(f"SOPに無いイベント: {name}")
            if span is None:
                continue
            s, e = span.get("start_idx"), span.get("end_idx")
            if not (isinstance(s, int) and isinstance(e, int) and 0 <= s <= e < n):
                raise ValueError(f"{name} の区間が不正: {span}")
        doc = {
            "schema_version": "0.1",
            "sop_id": self.sop_def["sop"]["id"],
            "fps": self.fps,
            "n_frames": n,
            "annotated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "events": events,
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        os.replace(tmp, self.out_path)


class Handler(BaseHTTPRequestHandler):
    server: Annotator

    def log_message(self, fmt, *args):  # フレーム毎のGETでターミナルを埋めない
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def do_GET(self):
        if self.path == "/":
            data = build_page_data(self.server.sop_def, self.server.frame_files,
                                   self.server.fps, self.server.out_path,
                                   self.server.load_gt_events())
            template = template_text("annotator.html")
            data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
            html = template.replace('"__ANNOTATOR_DATA__"', data_json)
            self._send(200, html.encode(), "text/html; charset=utf-8")
        elif self.path.startswith("/frames/"):
            name = self.path.rsplit("/", 1)[1]
            if name not in self.server.frame_files:   # パス走査対策: 既知の連番のみ
                self._send(404, b"not found", "text/plain")
                return
            self._send(200, (self.server.frames_dir / name).read_bytes(), "image/jpeg")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/save":
            self._send(404, b"not found", "text/plain")
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            events = json.loads(body)["events"]
            self.server.save(events)
        except (ValueError, KeyError) as e:
            self._send_json({"ok": False, "error": str(e)}, code=400)
            return
        self._send_json({"ok": True})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sop", default=str(DEFAULT_SOP))
    ap.add_argument("--frames-dir",
                    default=str(DEFAULT_FRAMES))
    ap.add_argument("--fps", type=float, default=1.0,
                    help="フレーム抽出時のfps(t=idx/fpsの換算に使う。既定: 1.0)")
    ap.add_argument("--out", default=None,
                    help="保存先(既定: SOPと同じディレクトリの ground_truth.json)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    sop_def = load_sop(args.sop)
    frames_dir = Path(args.frames_dir)
    frame_files = sorted(p.name for p in frames_dir.glob("*.jpg"))
    if not frame_files:
        raise SystemExit(f"[annotator] {frames_dir} に.jpgが見つかりません")
    if args.out:
        out_path = Path(args.out)
    elif Path(args.sop).resolve() == DEFAULT_SOP.resolve():
        out_path = DEFAULT_GT
    else:
        out_path = Path(args.sop).parent / "ground_truth.json"

    server = Annotator(("127.0.0.1", args.port), sop_def, frames_dir,
                       frame_files, args.fps, out_path)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"[annotator] {url} で待機中 ({len(frame_files)}フレーム, "
          f"{len(sop_def['events'])}イベント)")
    print(f"[annotator] 保存先: {out_path} (操作のたびに自動保存。Ctrl-Cで終了)")
    if not args.no_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[annotator] 終了")


if __name__ == "__main__":
    main()
