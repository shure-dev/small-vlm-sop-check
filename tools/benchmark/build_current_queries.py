#!/usr/bin/env python3
"""既存の英語queryから現行SOPと完全一致するbenchmark snapshotを作る。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    output: dict[str, dict[str, str]] = {}
    removed: dict[str, list[str]] = {}
    for unit_id, queries in source.items():
        sop_path = ROOT / "datasets" / "factory_ego" / "sops" / unit_id / "sop.yaml"
        sop = yaml.safe_load(sop_path.read_text(encoding="utf-8"))
        event_ids = [event["id"] for event in sop["events"]]
        missing = [event_id for event_id in event_ids if event_id not in queries]
        if missing:
            raise SystemExit(f"英語queryが不足しています: {unit_id}: {missing}")
        output[unit_id] = {event_id: queries[event_id] for event_id in event_ids}
        extras = sorted(set(queries) - set(event_ids))
        if extras:
            removed[unit_id] = extras

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[queries] units={len(output)} events={sum(map(len, output.values()))}")
    if removed:
        print(f"[queries] removed stale events: {json.dumps(removed, ensure_ascii=False)}")
    print(f"[queries] {args.out}")


if __name__ == "__main__":
    main()
