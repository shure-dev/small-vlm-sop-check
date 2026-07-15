# Operations

## 回帰検査

```bash
python -m pytest -q
python tools/benchmark/validate.py
python tools/quality/check_docs.py
python tools/quality/check_public.py
```

媒体取得済み環境では `python tools/benchmark/validate.py --require-media` も実行します。

## Factory Ego媒体

Hugging Faceで `builddotai/Egocentric-10K` のgated accessへ同意し、認証後に実行します。

```bash
python -m pip install -e ".[fetch]"
python tools/benchmark/fetch_factory_ego.py          # dry-run
python tools/benchmark/fetch_factory_ego.py --apply
python tools/benchmark/validate.py --require-media
```

必要なclipだけを取得し、各unitの20秒区間を2fpsで抽出して公開SHA manifestと照合します。不一致の媒体は書き込みません。

## 手動アノテーション

```bash
sop-app --dataset factory_ego
```

20件すべてを日本語でアノテーションします。各unitは、SOPにある全event IDについて区間または非該当の `null` が保存されると完了です。完了後に別工程で英訳を作成し、SOPのevent IDを保ったquery JSONを推論runnerへ渡します。

## 推論

```bash
.venv-vlm/bin/python tools/benchmark/run_marlin_prediction.py \
  --queries /path/to/english-queries.json \
  --run-id <date>-factory_ego-marlin-2b-human-r1
```

queryはunitごとにSOPのevent IDをちょうど一度含む必要があります。runnerはhuman GTを読みません。完了runは上書きしません。

## 評価

```bash
sop-check eval \
  --ground-truth datasets/factory_ego/annotations/human/<unit>.json \
  --prediction runs/<run-id>/predictions/<unit>.json
sop-app --dataset factory_ego
```

アプリの「結果レビュー」で人手とモデルのタイムライン、mean tIoU、tIoU@0.5を確認します。

## データ追加

一般のローカル動画は `sop-dataset add-video` で追加します。Factory Egoのpilotを増やす場合も、機械生成イベントによる選定はせず、source clipと切り出し範囲を明示して同じunit契約へ登録します。
