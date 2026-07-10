# Factory Ego model comparison

## 現在地

8 unit × 20 framesを対象に、Fable 5、Opus 4.8、Qwen3-VL-4Bの予測を同一形式へ移行しました。

| run | role | unit coverage | formal accuracy |
|---|---|---:|---|
| `20260710-fable5-reference-r1` | large-model reference prediction | 8/8 | 未評価（人手GTなし） |
| `20260710-opus48-reference-r1` | large-model reference prediction | 1/8、10 framesのみ | 未評価（人手GTなし） |
| `20260710-qwen3-4b-baseline-r1` | local small-VLM baseline | 8/8 | 未評価（人手GTなし） |

人手GT作成前に計算できるのは、一致率・回答分布・境界差などの予備比較です。precision、recall、F1、balanced accuracy、tIoUはhuman annotation revisionを入力にしたevaluation runで計算します。
