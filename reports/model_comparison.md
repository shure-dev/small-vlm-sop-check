# Factory Ego model comparison

## 現在地

データセットを刷新しました。旧構成（factory051/worker001の8 unit × 20フレーム）と、それに対するprediction run 5本・reference tIoU予備比較は廃止済みです（git履歴には残っています）。

現行データセットは **6工場・18 worker・20作業種類の20 unit（各20秒・2fps・40フレーム）** で、SOPは手順判定向けのイベント定義を持ちます（[データセットREADME](../datasets/factory_ego/README.md)参照）。

イベント定義を全面改訂しました（2026-07-11）。当初のtranscription由来の英語定義は実フレームと照合するとズレが多く、Fable×Opusの予備比較でも64イベント中16件が「両者とも非検出」＝窓内で起きていない疑いが出たため、**抽出フレーム（2fps・20秒窓）の目視による日本語イベント定義**へ置き換えました（方法論は[イベント定義ガイド](../docs/benchmark/events.md)）。

これに伴い、旧英語SOPに対するreference run 2本（Fable 5・Opus 4.8、10フレーム時代）は廃止済みです（git履歴には残る）。現時点のprediction runは0本で、今後の予定:

1. 新イベント定義に対するアノテーション（区間ラベル付け）
2. ローカル小型VLMのbaseline run作成（回答収集は複数画像入力への拡張が必要）
3. モデル間一致・回答分布・境界差の予備比較（人手GTができるまで「精度」とは表記しない）

## 評価の原則（変わらない）

- 正式なprecision、recall、F1、tIoUは人手GT revisionを固定したevaluation runでのみ計算する
- reference予測との一致は「大型モデルとどれだけ同じ区間を見たか」であり精度ではない
- 再現: `python3 tools/benchmark/reference_tiou.py --reference <run_id>`（run作成後）
