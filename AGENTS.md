# AGENTS.md

## 目的

産業向け一人称動画について、データ準備、人手アノテーション、小型VLM推論、tIoU評価、学習を再現可能に行う公開実験基盤です。現在の主対象はFactory Egoの20秒動画であり、外部の機械生成アノテーションを正解として使いません。

## 作業原則

- 日本語で報告し、判断理由と検証結果を簡潔に残す。
- 変更前に関連するコード、テスト、データ契約、ドキュメントを確認する。
- 古いV1/V2や用途の重なるアプリを増やさず、現行の実装を更新する。
- データ本体、認証情報、ローカル絶対パス、モデル重みをGitへ追加しない。
- 正解アノテーションは動画を人間が確認した結果だけにする。モデル出力を正解へコピーしない。
- イベント文は推測でなく画面上の動作を記述する。手段が識別に重要なら省略しない。
- 正解区間をモデル出力に合わせて動かさない。プロンプト改善とGT修正を分離する。
- 推論条件、query、入力hash、raw出力をrunに固定し、完了済みrunを書き換えない。
- 4B以下を主対象とし、7B以上を既定の実験対象へ追加しない。
- コミットはユーザーが明示的に依頼した場合だけ行う。

## 品質ゲート

```bash
python -m pytest
python tools/benchmark/validate.py --require-media
python tools/quality/check_docs.py
python tools/quality/check_public.py
git diff --check
```

アプリ変更では `npm run build`、API integration test、実ブラウザ確認を行い、保存層のテストで範囲外区間、重複区間、不正fieldを検証します。推論結果の数値は、ハッシュを固定した人手GTとprediction runから再計算できる場合だけ開発用の参考値として記載します。

## 主要導線

- `web/`: React/TypeScriptによる単一の動画UI
- `src/small_vlm_sop_check/apps/`: FastAPI、保存層、媒体preview、結果比較
- `datasets/<dataset>/`: 公開metadata、SOP、人手annotation、split
- `data/<dataset>/`: Git管理外のローカル動画・フレーム
- `tools/benchmark/`: データ再構成、検証、推論runner
- `runs/`: 不変の推論成果物
- `evaluations/`: runとhuman revisionを固定した評価
- `docs/benchmark/data-contract.md`: データ境界の正本
