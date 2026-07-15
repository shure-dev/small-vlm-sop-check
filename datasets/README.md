# Datasets

このディレクトリは、出典、unit metadata、SOP、人手annotation、split、hash manifestを管理する事実／仕様層です。実動画・抽出フレームはGit管理外の `data/`、モデル予測はリポジトリ直下の `runs/` に置きます。

| dataset | 役割 | unit | human GT | split |
|---|---|---:|---|---|
| [konro_inspection](konro_inspection/README.md) | CLI・注釈・評価・viewerの完結デモ | 1 | あり | `demo` |
| [factory_ego](factory_ego/README.md) | 工場一人称動画の手動annotation pilot | 20 | 作業中 | `dev_seen` |

共通の設計と追加手順は[ベンチマーク文書](../docs/benchmark/README.md)を参照してください。
