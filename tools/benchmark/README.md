# Factory Ego tools

## 媒体取得

```bash
python tools/benchmark/fetch_factory_ego.py          # dry-run
python tools/benchmark/fetch_factory_ego.py --apply
```

unit metadataから必要なsource clipと20秒窓を導出し、2fpsフレームを取得してSHA manifestと照合します。

## Marlin-2B

人手annotationと英訳が完了した後、SOP event IDを保ったquery JSONで実行します。

```bash
.venv-vlm/bin/python tools/benchmark/run_marlin_prediction.py \
  --queries /path/to/english-queries.json \
  --run-id <date>-factory_ego-marlin-2b-human
```

query event IDは各unitのSOPと完全一致する必要があります。出力は共通の秒区間predictionとして保存されます。
runnerはcanonical frameから640px幅のMP4を再生成し、実際の動画hash・解像度・fps・query snapshotをrunへ固定します。Marlin `find()`は1 queryにつき1区間を返すため、同じイベントの複数occurrenceは全区間評価と単一区間診断を分けて解釈してください。

## 4B以下model matrix

現行SOPと英語queryを同期し、候補モデルを1件ずつスモークしてから全20動画へ進めます。

```bash
.venv/bin/python tools/benchmark/build_current_queries.py \
  --source out/factory-ego-final-queries-r2.json \
  --out out/factory-ego-benchmark-v1-queries.json

.venv/bin/python tools/benchmark/run_model_matrix.py --mode smoke

.venv/bin/python tools/benchmark/run_model_matrix.py --mode full \
  --model qwen3.5-0.8b \
  --model qwen3.5-2b \
  --model qwen3.5-4b \
  --model qwen3-vl-2b \
  --model gemma4-e2b
```

matrix条件は[`configs/benchmark/factory_ego_model_matrix_v1.yaml`](../../configs/benchmark/factory_ego_model_matrix_v1.yaml)に固定します。スモークは`out/`へ保存し、正式runや`runs/index.jsonl`を汚しません。各モデルは別プロセスで実行されるため、Apple Siliconのunified memoryをモデル間で持ち越しません。

評価後はfull評価とスモーク失敗を集約します。

```bash
.venv/bin/python tools/benchmark/summarize_model_matrix.py \
  --config configs/benchmark/factory_ego_model_matrix_v1.yaml \
  --smoke-status out/benchmarks/factory_ego_model_matrix_v1/smoke-status.json \
  --out evaluations/factory_ego_model_matrix_v1.json
```

## 検証

```bash
python tools/benchmark/validate.py
python tools/benchmark/validate.py --require-media
```

後者はローカル媒体の全フレームSHAも確認します。
