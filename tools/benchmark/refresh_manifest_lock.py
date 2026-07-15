"""SOP編集後に datasets/factory_ego/manifest.lock.json のSOPハッシュを更新する。

SOPはprovisional（人手定義の暫定仕様）で、sop-appから編集される。
一方 manifest.lock.json はデータセットの整合性台帳としてSOPのsha256を持つため、
SOPを編集したらこのツールでハッシュを追随させる(tools/benchmark/validate.py が照合する)。

runs/ 配下の inputs.lock.json は「そのrunが当時使った入力」の歴史記録なので触らない。

使い方:
  python3 tools/benchmark/refresh_manifest_lock.py            # 差分表示のみ(dry-run)
  python3 tools/benchmark/refresh_manifest_lock.py --apply    # 書き込み
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dataset", default="factory_ego")
    parser.add_argument("--apply", action="store_true", help="lockを書き換える(省略時はdry-run)")
    args = parser.parse_args()

    dataset_root = args.repo / "datasets" / args.dataset
    lock_path = dataset_root / "manifest.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    changed = []
    for unit_id, entry in lock.get("units", {}).items():
        unit_dir = dataset_root / "units" / unit_id
        meta = json.loads((unit_dir / "meta.json").read_text(encoding="utf-8"))
        sop_path = (unit_dir / meta["sop_ref"]["path"]).resolve()
        for key, path in (("sop_sha256", sop_path), ("meta_sha256", unit_dir / "meta.json")):
            new = sha256(path)
            if entry.get(key) != new:
                changed.append(f"{unit_id}: {key} {entry.get(key, '')[:12]}... -> {new[:12]}...")
                entry[key] = new

    if not changed:
        print("[refresh_manifest_lock] 変更なし(lockは現状と一致)")
        return 0
    for line in changed:
        print(f"[refresh_manifest_lock] {line}")
    if not args.apply:
        print(f"[refresh_manifest_lock] dry-run: {len(changed)}件の差分。--apply で書き込み")
        return 1
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    print(f"[refresh_manifest_lock] {lock_path} を更新しました({len(changed)}件)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
