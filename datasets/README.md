# Datasets

このディレクトリは、入力媒体・SOP・人手アノテーションを管理する事実／仕様層です。モデル予測は原則としてリポジトリ直下の `runs/` に置きます。

| dataset | 役割 | unit | human GT | split |
|---|---|---:|---|---|
| [konro_inspection](konro_inspection/README.md) | CLI・注釈・評価・viewerの完結デモ | 1 | あり | `demo` |
| [factory_ego](factory_ego/README.md) | Egocentric-10KによるVLM精度比較 | 8 | なし | `dev_seen` |

共通の設計と追加手順は[ベンチマーク文書](../docs/benchmark/README.md)を参照してください。
