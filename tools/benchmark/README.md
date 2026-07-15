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

## 検証

```bash
python tools/benchmark/validate.py
python tools/benchmark/validate.py --require-media
```

後者はローカル媒体の全フレームSHAも確認します。
