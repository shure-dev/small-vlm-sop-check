# Benchmark architecture

このリポジトリでは、デモと精度比較を同じデータ契約で扱います。`examples/` は設けず、すべての入力データを `datasets/` に統合します。

```mermaid
flowchart LR
  D["datasets/<br/>媒体・SOP・人手GT"] --> R["runs/<br/>モデル予測"]
  D --> E["evaluations/<br/>人手GT revisionで評価"]
  R --> E
  E --> P["reports/<br/>比較・考察"]
```

## 現在のデータセット

- [Konro Inspection](../../datasets/konro_inspection/README.md): 人手GTまで揃った完結デモ
- [Factory Ego](../../datasets/factory_ego/README.md): 6工場・20作業種類の20 unit（各10秒）の精度比較データ。人手GT作成前

モデル間一致は予備比較であり、人手GTに対する評価とは区別します。詳細は[評価ポリシー](evaluation.md)を参照してください。
