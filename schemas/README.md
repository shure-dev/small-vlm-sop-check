# Schemas

`benchmark/` が現在サポートする唯一の機械可読データ契約です。

- `annotation.schema.json`: 人手による秒単位イベント区間
- `prediction.schema.json`: 推論方式に依存しない秒単位イベント区間
- `unit.schema.json`: 媒体・sampling・SOP参照
- `split.schema.json`: group-safeな分割
- `run.schema.json`: 再現可能な実験run

過去の形式は現行treeに併置せず、必要な場合はGit履歴を参照します。
