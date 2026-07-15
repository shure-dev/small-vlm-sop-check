# Konro Inspection

ガスコンロの始業前点検を題材に、動画抽出、VLMの回答収集、決定論的な区間検出、人手GT評価、共通viewerまでを再現する完結デモです。

```text
konro_inspection/
├── dataset.yaml
├── units/konro_inspection/
│   ├── meta.json
│   ├── procedure.md
│   └── frames/
├── sops/konro_inspection/
│   └── konro_inspection.yaml
├── annotations/human/konro_inspection.json
└── fixtures/reference_outputs/
```

`fixtures/reference_outputs/` はREADME、viewer、回帰テストをVLMなしで再現するための固定出力です。人手GTでも、新しい実験runの保存先でもありません。

MP4はGitへ含めません。アノテーションアプリは同梱した小さな回帰用フレームから、`data/konro_inspection/` 配下へローカルpreviewを生成します。

主要コマンドは[運用ガイド](../../docs/benchmark/operations.md)を参照してください。
