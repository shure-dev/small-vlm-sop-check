# Data contract

## 正本の境界

| path | 正本 | 入れないもの |
|---|---|---|
| `datasets/` | 出典、媒体参照、日本語イベント、人手GT、split、hash | 実動画、モデル出力を正解扱いしたファイル |
| `data/` | 各ユーザーのローカル媒体 | 公開metadataの唯一の正本 |
| `runs/` | モデル、prompt、入力lock、raw、prediction | 後から書き換えたGTやmetrics |
| `evaluations/` | runと評価時のannotation/prediction hashを固定した指標 | 入力hash不明の数値 |
| `training_runs/` | 学習設定、入力lock、状態 | Git管理すべきでない重みとlog |

## Human annotation

人間が動画を確認した結果だけを `annotations/human/` に置きます。日本語イベント文は推論用SOPとannotation JSONの `event_labels` に同期します。JSON単体でもevent IDの意味を解釈でき、SOPは将来の推論・学習入力として利用できます。翻訳は派生データであり、日本語の正本を上書きしません。

区間はunit先頭を0秒とするhalf-open interval `[start_s, end_s)` です。

- 区間リスト: イベントが起きた
- `null`: 候補を確認したが起きていない
- event keyがない: 未注釈

annotation JSONは `unit_id`、`annotation_revision`、`interval_convention`、`event_labels`、`events` だけを持ちます。SOPの全event IDが `events` にあれば完了、一部なら作業途中と導出します。承認状態や確認チェックは正解データへ混ぜません。

## Prediction

predictionは `run_id`、`unit_id`、`method` と同じ秒区間eventsを持ちます。推論runnerはhuman annotationを読みません。人手GTを修正しても完了済みrunは変更せず、新しいevaluationを作ります。
