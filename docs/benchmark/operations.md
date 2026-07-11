# Operations

## Quick regression

```bash
python3 -m pip install -e ".[test]"
python3 tools/benchmark/validate.py
python3 tools/quality/check_docs.py
python3 tools/quality/check_public.py
python3 -m pytest -q
```

Factory Egoのgated媒体を取得済みのローカル環境では、`python3 tools/benchmark/validate.py --require-media` で全フレームのSHAまで検証します。公開cloneでは媒体なしが正常です。

## Konro demo

```bash
sop-check judge \
  --sop datasets/konro_inspection/sops/konro_inspection/correct.yaml \
  --answer-log datasets/konro_inspection/fixtures/reference_outputs/answer_log.json

sop-check eval \
  --sop datasets/konro_inspection/sops/konro_inspection/correct.yaml \
  --ground-truth datasets/konro_inspection/annotations/human-v001/konro_inspection.json \
  --answer-log datasets/konro_inspection/fixtures/reference_outputs/answer_log.json
```

注釈toolとreplay viewerは引数なしでKonro datasetを開きます。

## Factory Ego media fetch

gated媒体は再配布しないため、各自がupstreamから取得します。

1. Hugging Faceで [builddotai/Egocentric-10K](https://huggingface.co/datasets/builddotai/Egocentric-10K) のgated accessに同意する（連絡先共有が必要）
2. `hf auth login` 済みの環境で取得スクリプトを実行する

```bash
python3 -m pip install -e ".[fetch]"                    # huggingface_hub + opencv
python3 tools/benchmark/fetch_factory_ego.py            # dry-run: 取得・抽出・SHA照合のみ
python3 tools/benchmark/fetch_factory_ego.py --apply    # 照合済みフレームを配置
python3 tools/benchmark/validate.py --require-media     # 全フレームのSHAまで検証
```

スクリプトはunit metaのsampling条件（clip・秒区間・1fps）から対象factory/workerとclipを導出し、必要なclipだけをtarのヘッダ走査で取り出すため、worker全体のtarをダウンロードしません（データを増やしても引数は不要）。抽出結果が `frames.sha256.json` と一致しない場合は書き込まず終了します（新規unitやmanifest再生成時のみ `--apply --update-manifest`）。

## Adding a local model prediction run

mlx-vlmが動く環境で、Konroベンチ上位モデルなどをFactory Egoへ追加するときは専用ツールを使います（run形式・lock・indexを自動で揃える）。

```bash
../../../.venv-vlm/bin/python tools/benchmark/run_local_prediction.py \
  --model <alias|HF ID> --model-name "<表示名>" \
  --run-id <日付>-factory_ego-<モデル>-baseline-r1
python3 tools/benchmark/validate.py
```

reference runとの区間一致は `tools/benchmark/reference_tiou.py` で測ります。人手GTがないため、この数値は精度ではなく予備比較として扱います。

## Adding data

unitの追加は `tools/benchmark/sample_units.py` の層化サンプリングを起点にします（決定論的・追記型。既存選定は変わらない）。

1. `python3 tools/benchmark/sample_units.py --annotations-root <annotated-egocentric-10kのclone> --n <合計数> --apply` でunit metaを追加する
2. `fetch_factory_ego.py --apply --update-manifest` で媒体・sampling条件・hashを固定する
3. SOPをversion付きで追加する（手順ステップ粒度のevents。[イベント定義](events.md)参照）
4. factory/worker単位でsplitを事前確定する
5. モデル出力はdatasetではなく新しいprediction runへ保存する（run_idは `{日付}-{dataset}-{モデル}-{役割}-r{通し番号}`）
6. validatorとpytestを通す
