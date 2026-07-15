# 自分の動画データを持ち込む

`sop-dataset` がdataset、unit metadata、SOP、split、hash manifestを生成します。動画と抽出フレームはGit管理外の `data/` に置き、ローカルの絶対パスはmetadataへ書き込みません。

## 最短手順

ffmpegを用意し、空のdatasetを作成します。

```bash
sop-dataset init \
  --dataset my_factory \
  --name "My Factory" \
  --description "組立工程の一人称視点動画"

sop-dataset add-video \
  --dataset my_factory \
  --unit worker01_clip01 \
  --video /path/to/private.mp4 \
  --start 10 --end 30 --fps 2 \
  --source-group worker01

sop-dataset split --dataset my_factory --split-id benchmark \
  --group-by source_group --validation-ratio 0.1 --test-ratio 0.1 --seed 42
sop-dataset validate --dataset my_factory
sop-app --dataset my_factory
sop-dataset validate --dataset my_factory --require-media
```

イベントは動画を見ながら `sop-app` で追加します。既知のイベントを先に定義したい場合だけ `--event 'inspect_part=作業者が部品を検査している'` を複数回指定できます。元動画は変更せず、指定区間のMP4とフレームを `data/my_factory/units/worker01_clip01/` に生成します。既存dataset/unit IDは上書きしません。

## 生成される構成

```text
datasets/my_factory/
├── dataset.yaml
├── README.md
├── units/<unit_id>/
│   ├── meta.json
│   └── frames.sha256.json
├── sops/<unit_id>/sop.yaml
├── annotations/human/<unit_id>.json
└── splits/development.json

data/my_factory/
└── units/<unit_id>/
    ├── video.mp4
    └── frames/f0000.jpg ...
```

`unit_id` はdataset内で不変の識別子にします。元動画内の切り出し区間は `source.start_second` / `source.end_second` に記録し、annotationはunit先頭を0秒とする相対時刻で扱います。

## 確認・修正・評価

```bash
sop-app --dataset my_factory
sop-check eval --ground-truth datasets/my_factory/annotations/human/<unit>.json \
  --prediction runs/<run>/predictions/<unit>.json
```

アプリで動画を確認し、日本語イベントと区間を作成します。推論結果はhuman GTと別のrunへ保存し、人手データを上書きしません。

学習・評価分割はフレーム単位ではなく、worker・現場・撮影セッション等のgroup単位で作ります。`--source-group` には分割を跨がせてはいけない単位を指定してください。testを目視する前に `sop-dataset split` を一度だけ実行し、以後は同じsplitを上書きしません。複数キーが必要なら `--group-by participant_id --group-by video_id` のように指定できます。`sop-dataset validate` はunit重複、SOPにないevent、動画長を越えるannotation、同じgroupのsubset跨ぎを検出します。媒体を共有できないCIでは引数なし、ローカルの完全検査では `--require-media` を付けます。
