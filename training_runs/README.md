# Training runs

`sop-train prepare` が、学習設定 `run.yaml`、安全な引数配列 `command.json`、入力hash
`inputs.lock.json` をここへ作成します。checkpointと学習ログはGit管理しません。

`prepared` runは実行前だけ可変です。`sop-train run` が終了すると成功・失敗を問わず
`immutable: true` になり、同じrun IDを再利用できません。
