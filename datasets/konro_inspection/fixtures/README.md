# Fixtures

`reference_outputs/` は、VLM・GPU・ネットワークなしでREADME、replay viewer、回帰テストを再現するための固定出力です。

- `answer_log.json`: 既定のQwen3-VL-4B回答ログ
- `models/*.json`: viewerと比較表に使うモデル別回答ログ

これらはground truthではありません。新しい実験結果はここへ追加せず、`runs/<run_id>/` に保存します。
