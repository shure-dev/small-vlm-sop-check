# 学習バックエンド: ms-swift

## 境界とバージョン

[ms-swift](https://github.com/modelscope/ms-swift)を最初の学習backendとします。動画を含む
custom JSONL、SFT、LoRA、学習後の推論を外部backendへ任せ、このrepositoryはデータ契約、
4B上限、入力lock、run状態、評価への接続を担当します。

統合契約は **ms-swift 4.4.1** と公式revision
[`9938a46`](https://github.com/modelscope/ms-swift/commit/9938a463946beb66d6b502d6f3a7dc64845c4df1)
に固定しています。根拠は公式の
[custom dataset形式](https://github.com/modelscope/ms-swift/blob/v4.4.1/docs/source_en/Customization/Custom-dataset.md)
と[動画LoRA例](https://github.com/modelscope/ms-swift/blob/v4.4.1/examples/train/multimodal/video.sh)です。

## 1. complete annotationをexportする

```bash
sop-export-ms-swift \
  --dataset my_factory \
  --split datasets/my_factory/splits/benchmark.json \
  --subset train \
  --out out/training/my-factory-train

sop-export-ms-swift \
  --dataset my_factory \
  --split datasets/my_factory/splits/benchmark.json \
  --subset validation \
  --out out/training/my-factory-validation
```

抽出フレームからunit動画を再生成し、`train.jsonl` とsplit hash・unit一覧・JSONL hashを
持つ `export.json` を出力します。1 unitが1サンプルで、公式形式の `messages`、`videos`、
ユーザー発話内の `<video>` を使います。応答は推論と同じ `events` の秒区間JSONです。
未発生eventも空リストとして負例に含めます。

一部unitだけのannotationは既定で拒否します。`--allow-partial` は形式確認専用であり、そのexportから
正式なtraining runは作れません。

## 2. 学習runを準備する

ms-swiftは学習環境側へ導入します。

```bash
python -m pip install 'ms-swift==4.4.1'

sop-train prepare \
  --run-id 20260714-my-factory-qwen25vl3b-lora-r1 \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --model-parameters-b 3 \
  --train-export out/training/my-factory-train \
  --validation-export out/training/my-factory-validation
```

`training_runs/<run-id>/` に次を作ります。

- `run.yaml`: backend revision、モデル規模、dataset/split、手法、状態
- `command.json`: shell評価しない `swift sft` 引数配列
- `inputs.lock.json`: exportとJSONLのSHA-256、unit一覧

4Bを超えるモデル、未完了annotation、train/validationのunit重複、hash不一致は準備時に
拒否します。validationを省略してもunit単位の暗黙ランダム分割はせず、
`--split_dataset_ratio 0` を指定します。

## 3. 確認して実行する

```bash
sop-train run \
  --run-dir training_runs/20260714-my-factory-qwen25vl3b-lora-r1 \
  --dry-run

sop-train run \
  --run-dir training_runs/20260714-my-factory-qwen25vl3b-lora-r1
```

実行終了後は成功・失敗を問わずrunをimmutableにし、exit codeとlog hashを記録します。
checkpointとlogは大容量・環境依存なのでGit管理外です。GPUメモリ、CUDA、動画sampling上限は
モデルと環境に依存するため、準備後の `command.json` をレビューしてから実行してください。

## 4. 学習前後を比較する

base modelと学習checkpointを、同一generation設定・同一test unitでprediction runへ保存してから
比較します。

```bash
sop-compare \
  --baseline-run runs/<base-run> \
  --tuned-run runs/<tuned-run> \
  --out evaluations/<comparison>/comparison.json
```

dataset ID、split、target unitのいずれかが異なる比較は拒否します。集約mean Temporal IoU、
tIoU@0.1/0.3/0.5のprecision/recall/F1、unit別deltaを保存します。training lossだけで改善を
主張しません。

## 参考OSSとの役割分担

| OSS | 参考にする点 | 本repositoryの責務 |
|---|---|---|
| [ms-swift](https://github.com/modelscope/ms-swift) | 動画VLMのSFT/LoRA、推論 | export、設定固定、評価への接続 |
| [FiftyOne](https://github.com/voxel51/fiftyone) | 大規模な動画dataset探索・QA | `sop-app` とinterval annotation |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | 代替fine-tuning backend | 将来のbackend adapter候補 |
