# Repository layout

## Python package

import可能なコードは `src/small_vlm_sop_check/` のみに置きます。rootのdataset、docs、toolsを偶然importできないsrc-layoutにすることで、checkout上だけ動いてinstall後に壊れる問題を防ぎます。

```text
src/small_vlm_sop_check/
├── core/       # PyYAML以外に依存しない判定・評価
├── inference/  # cv2 / mlx-vlm / transformers等のoptional backend
├── training/   # 学習exportと外部backend run
├── evaluation/ # run間の集約・比較
├── apps/       # FastAPI、annotation保存、媒体preview、結果比較、frontend build
└── cli.py      # sop-check
```

- `sop-check`: VLMの回答収集、判定、評価
- `sop-app`: 人手annotationとprediction reviewの共通Web UI
- `sop-view`: 同じUIの読み取り専用入口

entry pointとPython依存は `pyproject.toml` を唯一の正本とします。UIの正本は `web/`、HTTP APIは `apps/server.py`、CLI起動は `apps/launcher.py` です。`apps/frontend_dist/` は配布用の生成物で、手編集しません。

UIを変更した場合だけNode.jsを使います。通常利用者はビルド済み資産を含むPython packageを使うため、Node.jsは不要です。

```bash
npm ci --prefix web
npm --prefix web run build
python -m pytest tests/integration/test_app_api.py
```

参考: [PyPA src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)、[PyPA command-line tools](https://packaging.python.org/en/latest/guides/creating-command-line-tools/)、[setuptools package data](https://setuptools.pypa.io/en/stable/userguide/datafiles.html)

## Non-package code

`tools/` はrepositoryの保守にしか使わないscriptに限定します。

- `tools/benchmark/`: dataset移行・lock/hash検証
- `tools/quality/`: 文書リンクなどの静的検査

ユーザーが直接使う機能は `tools/` へ置かず、packageのentry pointとして公開します。

## Data boundary

- `datasets/`: Git管理する出典、unit metadata、SOP、人手GT、split、hash manifest
- `data/`: Git管理しない各ユーザーの動画、音声、抽出フレーム
- `runs/`: 入力revisionを固定した不変の予測
- `training_runs/`: 学習設定・入力lock・状態。重みとlogはGit管理外
- `evaluations/`: predictionとhuman GT revisionを固定した派生評価

dataset metadataの `media.path` はunit内の論理パスです。実媒体は
`data/<dataset_id>/units/<unit_id>/<media.path>` から解決します。動画本体とアプリが生成する
previewは例外なく `data/` に置きます。小さな回帰テスト用静止フレームだけは
`media.availability: bundled` を明示して `datasets/` に同梱できます。

## Tests and schemas

- `tests/unit/`: coreロジックの振る舞い
- `tests/integration/`: dataset、run、docs、CLI境界
- `schemas/benchmark/`: 現在サポートする唯一の公開データ契約

pytestのimport pathは `pyproject.toml` で設定し、test file内で `sys.path` を変更しません。
