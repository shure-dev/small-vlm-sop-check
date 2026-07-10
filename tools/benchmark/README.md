# Factory Ego benchmark tools

設計・評価・データ追加の正本は[ベンチマーク運用ガイド](../../docs/benchmark/operations.md)です。このページはコマンド固有のメモだけを扱います。

## Fetch media

```bash
python3 tools/benchmark/fetch_factory_ego.py          # dry-run
python3 tools/benchmark/fetch_factory_ego.py --apply  # 照合済みフレームを配置
```

gated accessに同意済みのHFアカウント（`hf auth login`）が前提。unit metaが参照する5clipだけをtarヘッダ走査で取り出し、1fps抽出して `frames.sha256.json` と照合する。不一致は書き込まない。抽出仕様を変えてmanifest側を作り直す時だけ `--apply --update-manifest` を使う（runsの `inputs.lock.json` は当時の記録として変更しない）。

## Validate

```bash
python3 tools/benchmark/validate.py
```

フレームSHA-256、unit/SOP lock、factory/worker split、prediction coverage、run不変条件、`runs/index.jsonl`を検査する。VLMやネットワークは不要。

## Re-run the legacy migration

移行コマンドは既定でdry-runになり、既存ファイルと1byteでも異なる場合は上書きせず停止する。

```bash
python3 tools/benchmark/migrate_factory_ego.py \
  --legacy-examples /path/to/legacy/examples \
  --scratchpad /path/to/ego10k

# dry-run確認後のみ
python3 tools/benchmark/migrate_factory_ego.py \
  --legacy-examples /path/to/legacy/examples \
  --scratchpad /path/to/ego10k \
  --apply
```

必要なscratchpad構造は `frames_00000` 等の1fpsフレームと、`vlm_unit_a`〜`vlm_unit_h` の `sop.yaml` / `answer_log.json`。移行後のベンチはscratchpadへ依存しない。
