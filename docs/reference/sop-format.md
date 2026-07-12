# SOPフォーマット

SOPは、VLMに聞く質問と、回答から検出するイベントをYAMLで定義します。動画で実際に起きたことを記録する人手アノテーションとは分離します。

## 最小例

```yaml
sop:
  id: konro_inspection
  name: コンロ始業前点検
  domain_hint: "ガスコンロの点検作業を上から撮った動画"

questions:
  - id: knob
    ask: "手がコンロ手前のつまみを操作しているか"
    values: ["yes", "no"]
  - id: pointing
    ask: "人が対象を指差しているか"
    values: ["yes", "no"]

events:
  ignite:
    evidence: "knob==yes"
    min_frames: 2
  point1:
    evidence: "pointing==yes"
    occurrence: 1
```

YAML 1.1の処理系では裸の `yes` / `no` が真偽値として解釈されることがあるため、値はクォートします。

## occurrence

同じ質問が動画内で複数回 `yes` になる場合に、「何回目の動作か」を指定します。

```yaml
events:
  point1:
    evidence: "pointing==yes"
    occurrence: 1
  point2:
    evidence: "pointing==yes"
    occurrence: 2
```

複数回の動作には明示的な `occurrence` を推奨します。省略すると、イベントの宣言順が割り当てに影響します。

## イベント検出の調整

| フィールド | 用途 |
|---|---|
| `min_frames` | Nフレーム以上続いた回答だけをイベントとして扱う |
| `max_gap_frames` | 短い回答の揺れをまたいで区間を接続する |
| `defaults` | 複数イベントの既定値をまとめて指定する |

動画ごとにイベント語彙がどこで定義されるかは、[動画ごとのイベント定義](../benchmark/events.md)を参照してください。実例は [`datasets/konro_inspection/sops/konro_inspection/`](../../datasets/konro_inspection/sops/konro_inspection/) にあります。
