# Human annotations

このディレクトリには、人間が動画全体と区間境界を確認したannotationだけを置きます。

`human/<unit_id>.json` は `sop-app` によって作成されます。日本語イベント文はJSONの `event_labels` と対応する `sops/<unit_id>/sop.yaml` に同期されます。モデル出力、翻訳、外部annotationはここへ置きません。
