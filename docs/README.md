# Documentation

READMEは製品概要とクイックスタート、`docs/` は設計・運用・意思決定の正本です。

## Start here

- [実験プラットフォーム設計](design/experiment-platform.md)
- [自分の動画データを持ち込む](guides/bring-your-own-data.md)
- [学習バックエンド: ms-swift](training/ms-swift.md)

## Benchmark

- [全体像](benchmark/README.md)
- [Konroベンチマーク結果](benchmark/konro-results.md)
- [データ契約とフォルダ境界](benchmark/data-contract.md)
- [動画ごとのイベント定義](benchmark/events.md)
- [評価ポリシー](benchmark/evaluation.md)
- [運用・検証・追加手順](benchmark/operations.md)

## Reference

- [動画アノテーションアプリ](reference/annotator.md)
- [SOPフォーマット](reference/sop-format.md)
- [モデルと生成オプション](reference/models.md)

## Decisions

- [0001: 事実・予測・評価を分離する](decisions/0001-separate-facts-predictions-evaluations.md)
- [0002: src-layoutとCLI packageを採用する](decisions/0002-adopt-src-layout.md)
- [0003: 動画区間を主契約にし、学習backendを分離する](decisions/0003-video-first-experiment-platform.md)

## Development

- [リポジトリ構造とpackage境界](development/repository-layout.md)
- [公開前チェックリスト](development/public-release.md)

## Assets

画像やGIFなど、本文から参照する静的資産は `assets/` に置きます。
