# Fixtures

`reference_outputs/` は、VLM・GPU・ネットワークなしで区間検出の回帰テストを実行するための固定出力です。

- `answer_log.json`: 既定のQwen3-VL-4B回答ログ

これはground truthではありません。新しい実験結果はここへ追加せず、`runs/<run_id>/` に保存します。
