# SOPフォーマット

SOPは、動画から検出したいイベントをYAMLで定義します。**イベント = 「作業者が〜している」という記述文**で、VLMは各フレームでその文が成り立つかを yes/no で答えます（疑問形「〜か？」では書かない）。回答が `"yes"` の連続区間が、そのイベントの出現になります。動画で実際に起きたことを記録する人手アノテーションとは分離します。

## 最小例

```yaml
sop:
  id: konro_inspection
  name: コンロ始業前点検
  domain_hint: "ガスコンロの点検作業を上から撮った動画"

events:
  - id: knob
    ask: "手がコンロ手前のつまみを操作している"
    values: ["yes", "no"]
    min_frames: 2
  - id: pointing
    ask: "人が対象を指差している"
    values: ["yes", "no"]
```

## フィールド

| キー | 必須 | 意味 |
|---|---|---|
| `sop.id` / `sop.name` | ✓ | unitと同名のid / 表示名 |
| `sop.domain_hint` | | 撮影状況の説明。全フレームのプロンプト冒頭に入る |
| `events[].id` | ✓ | イベントid（英数字/`_`）。回答ログ・GT・検出結果すべてのキー |
| `events[].ask` | ✓ | イベントの記述文（「作業者が〜している」）。VLMが各フレームで成否をyes/noで判定する |
| `events[].values` | | 回答語彙（既定 `["yes", "no"]`）。**必ずクォートする**（裸のyes/noはYAML 1.1でブール値になる） |
| `events[].min_frames` | | 検出に必要な最小連続フレーム数（既定は `defaults.min_frames`、無ければ2） |
| `defaults.min_frames` / `defaults.max_gap_frames` | | 全イベント共通の既定値 |

## 同じ動作が複数回起こる場合

何も特別なことは要りません。検出は各イベントの**出現区間のリスト**を返し、人手アノテーション（`ground_truth.json`）も同じイベントidに**区間のリスト**を記録します。

```json
"events": {
  "pointing": [{"start_idx": 5, "end_idx": 5}, {"start_idx": 12, "end_idx": 13}],
  "gloves": null
}
```

評価はGT・検出の両方を時系列順に並べ、k番目どうしを突き合わせます（余ったGT区間=見逃し、余った検出区間=誤検出）。

## 旧v1形式からの変更

旧形式（`questions:` と `events:` の2層 + `evidence` 式 + `occurrence`）は廃止しました。イベントが質問を直接持ちます。複数回の出現は `occurrence` 付きの別イベントではなく、上記のとおり同じイベントの区間リストで扱います。旧形式のSOPは `load_sop` が明確なエラーで拒否します。
