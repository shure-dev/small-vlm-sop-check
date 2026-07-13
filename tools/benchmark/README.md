# Factory Ego benchmark tools

設計・評価・データ追加の正本は[ベンチマーク運用ガイド](../../docs/benchmark/operations.md)です。このページはコマンド固有のメモだけを扱います。

## Fetch media

```bash
python3 tools/benchmark/fetch_factory_ego.py          # dry-run
python3 tools/benchmark/fetch_factory_ego.py --apply  # 照合済みフレームを配置
```

gated accessに同意済みのHFアカウント（`hf auth login`）が前提。unit metaからfactory/workerと必要clipを導出し、対象clipだけをtarヘッダ走査で取り出して1fps抽出、`frames.sha256.json` と照合する。不一致は書き込まない。新規unitのmanifest生成や抽出仕様変更時だけ `--apply --update-manifest` を使う（runsの `inputs.lock.json` は当時の記録として変更しない）。

## Sample units (層化抽出)

```bash
python3 tools/benchmark/sample_units.py \
  --annotations-root /path/to/annotated-egocentric-10k-dataset --n 20          # dry-run
python3 tools/benchmark/sample_units.py \
  --annotations-root /path/to/annotated-egocentric-10k-dataset --n 20 --apply  # meta.json書き出し
```

annotated-egocentric-10kのイベントログから「10秒に3〜6イベント開始が入る」窓を持つクリップを候補化し、作業種類→工場→workerの優先度で決定論的に選ぶ（乱数なし）。既存unitのclipは除外されるので、データを増やすときは `--n` を増やして再実行すれば既存選定は変わらない。

## New prediction run (local mlx-vlm)

```bash
../../../.venv-vlm/bin/python tools/benchmark/run_local_prediction.py \
  --model qwen2.5-3b \
  --model-name "Qwen2.5-VL-3B-Instruct 4-bit" \
  --run-id <日付>-factory_ego-qwen2.5-3b-baseline-r1
```

ローカルモデルでFactory Ego全unitの回答を収集し、既存runと同形式の不変prediction run（raw・正規化predictions・run.yaml・inputs.lock・index追記）を作る。rawはフレームごとに逐次保存するので、GPU Hang等で落ちても再実行すれば途中から再開する。`run.yaml` が既にあるrun IDは不変として拒否する。要mlx-vlm（Apple Silicon）。

## Reference tIoU (予備比較)

```bash
python3 tools/benchmark/reference_tiou.py \
  --reference <reference run_id> --json out/tiou.json
```

各runの回答から決定論的ルールでイベント区間を導き、reference runとの区間tIoUを測る。比較は共通unit・共通フレームidxに制限し、mean tIoUは両run検出ペアのみの平均（`core.evaluate` と同じ流儀）。referenceは人手GTではないため精度ではなく、[評価ポリシー](../../docs/benchmark/evaluation.md)の予備比較（モデル間一致・境界差）に当たる。VLM不要。

## Marlin-2B temporal grounding

Marlinはフレームごとの質問回答ではなく、動画とイベント文から開始・終了秒を直接返す。
専用runnerはその区間を既存の `frame_question_answers` schemaへ正規化するため、生成後のrunは
既存のvalidate・評価・replay viewerで扱える。コア検出・評価器は変更しない。

```bash
uv pip install --python ../../../.venv-vlm/bin/python -e '.[marlin]'

../../../.venv-vlm/bin/python tools/benchmark/run_marlin_prediction.py \
  --queries tools/benchmark/marlin-pilot4-v001.json \
  --run-id <日付>-factory_ego-marlin-2b-pilot4-r1

PYTHONPATH=src ../../../.venv/bin/python -m small_vlm_sop_check.apps.replay \
  --runs-dir runs --dataset-root datasets/factory_ego \
  --out out/replay_marlin.html
```

前提はgated framesの取得、`ffmpeg`、Apple SiliconではMPS対応PyTorch。query JSONは
unitごとにSOPの全event IDをちょうど1回含める。rawはイベントごとに逐次保存されるので、
中断時は同じコマンドで再開できる。完了済み `run.yaml` は他runnerと同様に上書きしない。
モデル独自の `find()` を呼ぶため `trust_remote_code=True` を使用するが、既定では実験時の
Hugging Face commitを `--revision` で固定する。別revisionを試す場合はコード差分を確認する。

4 unitで実行した結果とqueryは[Marlin-2B pilot4レポート](../../reports/marlin-2b-pilot4.md)に記録する。

## Validate

```bash
python3 tools/benchmark/validate.py
```

フレームSHA-256、unit/SOP lock、factory/worker split、prediction coverage、run不変条件、`runs/index.jsonl`を検査する。期待フレーム数は各unitの `meta.json`（`sampling.n_frames`）から導出する。VLMやネットワークは不要。

旧factory051データ（8 unit・5 run）と、そのlegacy移行ツール `migrate_factory_ego.py` は2026-07-11のデータセット刷新で廃止した。必要ならgit履歴を参照。
