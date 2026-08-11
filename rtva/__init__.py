"""rtva — Real-Time Video Analysis with small VLMs.

W秒の動画窓を W秒以内に解析し、タイムスタンプ付きイベント区間を出力する
リアルタイム動画解析の最小フレームワーク。

構成:
- goalstep: データセットアダプタ (Ego4D GoalStep のアノテーション・タイムライン)
- windows:  窓の切り出し・分類・履歴テキスト・窓結果のマージ
- prompts:  プロンプト構築 (閉語彙)
- parsing:  モデル出力のパースと検証
- metrics:  秒単位正解率・イベントP/R・境界誤差
"""

__version__ = "0.1.0"
