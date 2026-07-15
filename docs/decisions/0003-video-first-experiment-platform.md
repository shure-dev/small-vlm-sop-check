# ADR 0003: 動画区間を主契約にし、学習backendを分離する

## Status

Accepted

## Context

既存実装は各フレームへyes/no質問を行い、その回答列から区間を導出する方式を中心にしていました。しかし、動画Temporal Groundingモデルは開始・終了秒を直接返します。これを再びフレーム回答へ量子化すると、境界精度と「区間なし」の意味が失われ、方式比較も不自然になります。

また、データ確認、学習、推論、評価を1つの巨大な独自frameworkへまとめると、モデル更新へ追随できません。

## Decision

1. 主たるprediction contractをイベントごとの秒単位intervalとする
2. Frame Classificationはintervalを生成する一つのbackendとして残す
3. raw出力と正規化predictionを分離する
4. 内蔵viewer/annotatorはdataset contractを直接読む
5. private/gated mediaは `data/` に置き、Gitへ含めない
6. 学習はms-swift等の外部backendへ委譲し、本repositoryはexport、run固定、共通評価を担当する
7. 現行treeは秒区間契約だけを持ち、過去の形式はGit履歴にのみ保存する

## Consequences

- Marlin等の区間出力を情報を失わず保存できる
- フレーム方式と動画方式を同じTemporal IoUで比較できる
- 学習backendを交換してもdatasetと評価が変わらない
- 入力方式が増えても、保存するannotationとpredictionの形は増やさない
