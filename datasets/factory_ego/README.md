# Factory Ego accuracy comparison dataset

全体のフォルダ境界と評価条件は[ベンチマーク文書](../../docs/benchmark/README.md)を参照してください。

Egocentric-10K の `factory051 / worker001` から切り出した、VLMがフレームごとの質問にどれだけ正しく答えられるかを比較するための開発用データです。

## 境界

- `units/`: 出典、区間、ローカル1fpsフレームのSHA manifest
- `sops/`: 比較時に使う質問・イベント仕様。現時点では `provisional`（暫定）
- `annotations/`: 人手で検証した正解だけを置く。現在のFactory Ego 8 unitには人手GTがない
- `splits/`: factory/workerを跨がせない分割。既存8 unitは全て `dev_seen`

Fable、Opus、Qwenなどモデルの出力は事実層には置かず、リポジトリ直下の `runs/` に保存します。モデル間一致は予備比較であり、人手GTができるまで「精度」とは扱いません。

## 出典と配布

- Upstream: `builddotai/Egocentric-10K`
- Upstream license: Apache-2.0
- Access: Hugging Face上で連絡先共有への同意が必要なgated dataset
- Sampling: 1920x1080映像から1fpsで抽出（JPEG quality 85。`frames.sha256.json` とバイト一致する条件）

抽出フレームは公開repositoryへ含めません。upstreamのgated accessを通して取得したローカル媒体を使い、`frames.sha256.json` で同一性を検証します。再配布する場合はupstreamの最新ライセンスだけでなく、アクセス時に同意した条件とプライバシー要件を確認してください。

ローカル媒体の再構成は `tools/benchmark/fetch_factory_ego.py` で自動化されています。gated accessに同意した後の手順は[ベンチマーク運用ガイド](../../docs/benchmark/operations.md)を参照してください。
