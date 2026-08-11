# AGENTS.md — 作業規範

このリポジトリで作業するエージェント向けの共通ルール。

## プロジェクトの絶対要件

**W秒の動画窓を W秒以内に解析する（リアルタイム保証）。** 設計変更はこの制約を壊さないこと。
レイテンシに影響する変更をしたら `scripts/eval_windows.py` / `scripts/eval_stream.py` の締切遵守率を実測して報告する。

## 原則

- 報告・ドキュメントは日本語。コード・コミットメッセージの識別子は英語のまま
- 小さく刻んで必ず検証: 実装 → 実行して出力確認 → 意味のある単位で commit
- **push・PR作成・外部公開はユーザー承認が必要**
- データ実体（動画・フレーム・重み・認証情報）は Git に入れない（`data/` `runs/` `training_runs/` は gitignore）
- 実験結果の数値は実測のみ。推定値には「推定」と明記する
- GTをモデル出力に合わせて改変しない。評価セット（`validation_eval.jsonl`）は学習に使わない
- Ego4D の認証情報は一時ファイルで扱い、使用後に削除する

## 環境

- Mac (Apple Silicon, 24GB) / MLX。学習・推論とも `.venv-vlm`（mlx-vlm 0.6.3）
- 補助スクリプト（データ処理・レポート生成）はリポジトリ直下の `.venv`
- 長時間ジョブは `caffeinate -is` を付け、ログを `training_runs/<run>/train.log` に残す
- mlx-vlm 0.6.3 / transformers には学習・前処理を壊す問題が5件あり、回避は
  `scripts/train.py` と `rtva/imaging.py` に同梱（削除しない）。画像トークン数の制御は
  必ず `rtva.imaging.set_max_image_pixels` を使う
