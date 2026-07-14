# Marlin-2B temporal grounding — Factory Ego pilot4

## 条件

- モデル: `lunahr/Marlin-2B-ungated`
- revision: `de783b96b80f477c5e665d2202571a84cb0761da`
- backend: Transformers / PyTorch / MPS、float16
- 入力: Factory Egoのcanonical frames（2fps・40フレーム・20秒）から生成したMP4
- 推論: `Marlin.find(video, event)`。モデル内部promptは
  `Identify the timestamps during which "{event}" takes place. Output the time range as "From <start> to <end>." (numbers in seconds).`
- query: [`tools/benchmark/marlin-pilot4-v001.json`](../tools/benchmark/marlin-pilot4-v001.json)
- run: [`runs/20260713-factory_ego-marlin-2b-pilot4-r1`](../runs/20260713-factory_ego-marlin-2b-pilot4-r1)

モデルの秒区間は `floor(start*fps)` から `ceil(end*fps)` までのフレームyesへ量子化し、
既存の `frame_question_answers` schemaへ正規化した。推論時に人手GTは使用していない。
評価は既存 `core/evaluate.py` のtime-order pairingをそのまま使用し、採点方法は変更していない。

## 結果

| unit | mean tIoU | 対応区間 / GT区間 |
|---|---:|---:|
| f001_w004_material_replenishment | 0.628 | 3 / 3 |
| f001_w011_metal_stamping | 0.325 | 4 / 4 |
| f002_w002_garment_bagging | 0.711 | 4 / 4 |
| f002_w003_fabric_folding | 0.111 | 3 / 6 |
| **micro（対応した14区間）** | **0.454** | **14 / 17** |

この値は開発用4 unitでの予備結果であり、formal accuracyではない。Marlinは1 queryにつき
単一区間を返すため、同一イベントの複数出現には未対応。またqueryは英語で人手作成しており、
query wordingの変更は別条件として新しいrun IDで記録する必要がある。

## 再現

```bash
../../../.venv-vlm/bin/python tools/benchmark/run_marlin_prediction.py \
  --queries tools/benchmark/marlin-pilot4-v001.json \
  --run-id <新しいrun-id>
```

`raw/<unit>.json` にMarlinの原文・秒区間・query、`predictions/<unit>.json` に既存schemaへ
正規化したフレーム回答を保存する。完了runは不変とし、条件変更時は別run IDを使う。
