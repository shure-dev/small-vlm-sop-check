"""動画アノテーションWebアプリのCLI launcher。"""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn

from .server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="sop-app")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--unit", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--read-only", action="store_true")
    args = parser.parse_args()

    read_only = args.read_only or Path(sys.argv[0]).name == "sop-view"
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not read_only:
        parser.error("書き込み可能なアノテーションアプリはlocalhostにだけ公開できます")
    app = create_app(
        read_only=read_only,
        initial_dataset=args.dataset,
        initial_unit=args.unit,
    )
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, args=(f"http://{args.host}:{args.port}",)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
