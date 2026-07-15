# Evaluations

評価条件の正本は[評価ポリシー](../docs/benchmark/evaluation.md)です。

評価はprediction runとは別の、不変なevaluation artifactとして作成します。入力にはprediction run IDだけでなく、使用した各annotationとpredictionのSHA-256を必ず固定します。`human` のような可変ディレクトリ名だけでは評価入力を再現できません。

Factory Egoには一部unitのdevelopment GTがありますが、未見testではないため正式精度には使いません。development値には `formal_accuracy: false` と制約を明記します。`sop-compare` は既定で全unitのannotationが完了したデータセットだけを受け付けます。モデル間一致だけを精度として報告しないでください。
