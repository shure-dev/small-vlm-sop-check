# 動画VLM実験プラットフォーム設計

## 中心タスク

メインは、動画クリップとイベント文をVideo VLMへ入力し、開始・終了タイムスタンプを直接得るTemporal Groundingである。静止画を1枚ずつ入力してyes/no回答列から区間を作るFrame Classificationもbaselineとして実験できるが、プラットフォーム全体の前提ではない。

両方式は、動画相対秒のhalf-open intervalという同じprediction contractへ正規化する。Temporal Groundingの直接区間は情報を落とさず保持し、Frame Classificationだけが回答列を決定論的ルールで区間へ変換する。

## 現在の成功条件

1. 非公開動画をGitへ追加せずdatasetとして登録できる
2. 人間が動画を見て、日本語イベントと秒区間を快適に記録できる
3. 未入力と、映像内に存在しないイベントを区別できる
4. predictionをhuman GTと同じ時間軸で確認し、Temporal IoUを評価できる
5. 全unitのイベント入力が揃ったデータを将来の学習exportへ使える

## 依存方向

```text
local media
  ↓
human annotation facts
  ↓                 ↘
inference runs       training export → checkpoints → inferenceへ戻る
  ↓
evaluations and reports
```

dataset層はモデルを知りません。推論runnerはhuman GTを読みません。評価はraw応答ではなく、共通の秒区間predictionを読みます。

## UIのメンタルモデル

アノテータの仕事は「動画を選ぶ → 見る → 動作を言葉にする → 境界を合わせる → 全体を確認する」です。モデル名、prompt形式、学習backendをannotation画面へ持ち込みません。結果レビューは別ワークスペースに分離しますが、同じunitと時間軸を共有します。

## 拡張点

- dataset: 共通unit metadataを満たす任意のローカル／gated動画
- inference: Marlin、Transformers、MLXなど4B以下の動画VLM
- training: ms-swiftを最初のbackendとし、必要なら他backendをadapter追加
- tracking: repository内のimmutable runを正本とし、外部trackerは任意

## ロードマップ

1. 20動画の日本語human annotationを完成する
2. Codexで日本語を英訳し、人間が意味を確認する
3. Marlin-2Bなどへ同じ英語queryを入力する
4. human GTに対してtIoUを評価する
5. annotation guidelineを固定し、未見clipのvalidation/testを作る
6. completeなtrain splitをms-swiftへexportしてfine-tuningする
