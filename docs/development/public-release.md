# Public release checklist

このrepositoryは公開を前提とします。commit前に次を確認します。

```bash
python3 tools/quality/check_public.py
python3 tools/quality/check_docs.py
python3 tools/benchmark/validate.py
python3 -m pytest -q
```

## Data

- Factory Ego: upstreamがgatedのため、抽出フレームは同梱しない
- Factory Egoで公開するのはsource metadata、SHA manifest、SOP、人手annotation、モデル予測
- README冒頭の縮小GIFだけをデモ用派生物として同梱し、`docs/assets/README.md` に出典、変更内容、適用条件を記載する
- upstream媒体の再配布は、最新license、アクセス時の同意条件、プライバシー要件を確認して別途判断する

## Repository hygiene

- `/Users/...` やprivate tempなどローカル絶対パスを含めない
- token、private key、API secretを含めない
- 95MBを超える単一ファイルを含めない
- 生成物は `out/`、cache、build directoryへ隔離する
- 完了runとdataset lockを手編集で上書きしない

`check_public.py` は、Gitが追跡中または新規追跡候補と判断するファイルだけを検査します。`.gitignore`されたローカルgated媒体は候補から除外されます。
