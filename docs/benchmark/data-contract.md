# Data contract

## Top-level responsibilities

| path | 正本 | 入れてはいけないもの |
|---|---|---|
| `datasets/` | 媒体、出典、SOP、人手GT、split | モデルを人手GTとして扱ったファイル |
| `runs/` | モデル・prompt・入力lock・raw・正規化予測 | 後から上書きした評価値 |
| `evaluations/` | prediction runとhuman GT revisionを固定した指標 | 入力revision不明の数値 |
| `reports/` | 比較表と考察 | 唯一の機械可読な実験記録 |
| `fixtures/` | GPUなしでデモを再現する固定出力 | 新規実験の正本 |

## Dataset minimum

各datasetは `dataset.yaml` と `README.md` を持ちます。各unitはstableな `unit_id`、媒体、sampling条件、SOP参照、annotation参照を持ちます。

gated datasetの媒体は公開repositoryへ再配布せず、取得条件・source pointer・SHA manifestを公開します。公開cloneで媒体が無い状態を正常とし、媒体必須の検査は明示的なlocal validationとして実行します。

SOPは仕様であり、GTではありません。Factory EgoのSOPは現在 `provisional`、KonroのSOPは人手GTで回帰検証済みです。

## Run invariants

完了runは不変です。後からGTが追加・修正された場合もprediction runへmetricsを追記せず、新しいevaluation runを作ります。`runs/index.jsonl` は検索用の再生成可能な索引で、正本は各 `run.yaml` です。
