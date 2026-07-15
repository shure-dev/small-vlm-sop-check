# Factory Ego manual annotation pilot

Egocentric-10Kから固定した20本の工場一人称clipを、人間が日本語で時間区間アノテーションするための開発セットです。各unitは20秒、2fps、40フレームです。

## 現在の状態

- 20 unitすべて媒体参照とSHA manifestを固定済み
- イベント候補は空から開始
- `annotations/human/` は手動作業によって作成される
- 全unitは `dev_seen`
- 人手確認完了までは正式精度を計算しない

## 配布境界

実動画と抽出フレームはgated source由来のためGitへ含めません。公開するのはsource pointer、切り出し開始・終了、sampling条件、SHA manifest、SOP、人手annotationです。

```bash
python -m pip install -e ".[fetch]"
python tools/benchmark/fetch_factory_ego.py --apply
python tools/benchmark/validate.py --require-media
sop-app --dataset factory_ego
```

Upstreamは `builddotai/Egocentric-10K`（Apache-2.0、gated access）です。利用・再配布時は最新のライセンス、アクセス条件、プライバシー要件を確認してください。

## フォルダ

```text
dataset.yaml                 dataset全体の状態
units/<unit>/meta.json       source clip、20秒窓、sampling、媒体参照
units/<unit>/frames.sha256.json
sops/<unit>/sop.yaml         日本語イベント定義
annotations/human/<unit>.json
splits/development.json
manifest.lock.json
```

アノテーション方法は[操作ガイド](../../docs/reference/annotator.md)、データ契約は[Data contract](../../docs/benchmark/data-contract.md)を参照してください。
