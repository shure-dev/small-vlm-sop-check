# Factory Ego model comparison

## 現在地

データセットを刷新しました。旧構成（factory051/worker001の8 unit × 20フレーム）と、それに対するprediction run 5本・reference tIoU予備比較は廃止済みです（git履歴には残っています）。

現行データセットは **6工場・18 worker・20作業種類の20 unit（各10秒・1fps・10フレーム）** で、SOPは手順判定向けのイベント定義を持ちます（[データセットREADME](../datasets/factory_ego/README.md)参照）。

現時点のprediction runは0本です。今後の予定:

1. SOPイベント定義の人手レビュー（進行中）
2. reference prediction run（大型モデルによるフレーム閲覧・回答）の作成
3. ローカル小型VLMのbaseline run作成（questionsが「直近数フレーム＋最新フレームの状態」を問う動画解析設計になったため、回答収集は複数画像入力への拡張が必要）
4. モデル間一致・回答分布・境界差の予備比較（人手GTができるまで「精度」とは表記しない）

## 評価の原則（変わらない）

- 正式なprecision、recall、F1、tIoUは人手GT revisionを固定したevaluation runでのみ計算する
- reference予測との一致は「大型モデルとどれだけ同じ区間を見たか」であり精度ではない
- 再現: `python3 tools/benchmark/reference_tiou.py --reference <run_id>`（run作成後）
