# Konro Inspection

ガスコンロの始業前点検を題材に、動画抽出、VLMの回答収集、決定論的判定、人手GT評価、replay viewerまでを再現する完結デモです。

```text
konro_inspection/
├── dataset.yaml
├── units/konro_inspection/
│   ├── meta.json
│   ├── media/konro_inspection.mp4
│   ├── procedure.md
│   └── frames/
├── sops/konro_inspection/
│   ├── correct.yaml
│   ├── wrong_order.yaml
│   └── missing_step.yaml
├── annotations/human-v001/konro_inspection.json
└── fixtures/reference_outputs/
```

`fixtures/reference_outputs/` はREADME、viewer、回帰テストをVLMなしで再現するための固定出力です。人手GTでも、新しい実験runの保存先でもありません。

主要コマンドは[運用ガイド](../../docs/benchmark/operations.md)を参照してください。
