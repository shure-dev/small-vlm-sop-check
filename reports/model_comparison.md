# Factory Ego model comparison

## 現在地

データセットを刷新しました。旧構成（factory051/worker001の8 unit × 20フレーム）と、それに対するprediction run 5本・reference tIoU予備比較は廃止済みです（git履歴には残っています）。

現行データセットは **6工場・18 worker・20作業種類の20 unit（各20秒・2fps・40フレーム）** で、SOPは手順判定向けのイベント定義を持ちます（[データセットREADME](../datasets/factory_ego/README.md)参照）。

参照predictionとしてClaude Opus 4.8のオンライン推論run（20/20 unit）が1本あり、それに対して**ローカル小型VLM（≤6B）12本のpilot run（6工場×各1 unit = 6 unit）**を追加しました。人手GTがないため、以下はすべて「精度」ではなく **Opus参照との一致（予備比較）** です。

## prediction run

| 種別 | run | model | 方式 | unit |
|---|---|---|---|---:|
| 参照 | `20260711-factory_ego-opus48-online5-r1` | Claude Opus 4.8 | オンライン（直近5フレームの因果窓） | 20 |
| ローカル | `20260711-factory_ego-<alias>-pilot6-r1` × 12 | 下表の12モデル | **単一フレーム入力**（最新1フレームのみ・時間文脈なし） | 6 |

ローカルrunは `tools/benchmark/run_local_prediction.py --subset pilot6-v001` で作成。対象6 unitは各工場から「Opus参照が定義イベントを全検出した最もクリーンなunit」を選定（[pilot6-v001.json](../datasets/factory_ego/subsets/pilot6-v001.json)）。mlxモデルは4bit・mlx-vlm 0.6.3、SmolVLM2は公式transformers実装（fp32）。

## pilot6 予備比較（Opus 4.8参照との一致・6 unit・全21イベント）

対象はpilot6の6 unit。参照のOpus 4.8はこの6 unitで計21イベントを検出している。行はおおむね健全なものを上、定数出力へ退化したもの（⚠）を下に並べた。

| モデル | サイズ | バックエンド | mean tIoU | ピークメモリ (MB) | 所見 |
|---|---|---|---:|---:|---|
| Qwen3-VL-4B-Instruct | 4B | mlx | 0.34 | 4271 | Opusに最も近く、崩れなし |
| MiniCPM-V-4.6 | 1.3B | mlx | 0.32 | 3647 | f006で値のクォート欠落→unclear化（5/6 unitは健全） |
| Qwen3-VL-2B-Instruct | 2B | mlx | 0.29 | 2854 | 0.6.3のJSON崩壊懸念はprefillで回避 |
| Gemma4-E2B-it | E2B | mlx | 0.28 | 4541 | 崩れなし |
| SmolVLM2-2.2B-Instruct | 2.2B | transformers/fp32 | 0.27 | — | 崩れなし。fp32必須（下記） |
| InternVL3-2B | 2B | mlx | 0.27 | 2458 | yes過剰傾向 |
| Qwen3.5-4B-MLX | 4B | mlx | 0.25 | 5369 | yes過剰で検出区間が膨張 |
| Qwen3.5-2B-MLX | 2B | mlx | 0.23 | 3331 | yes過剰 |
| Qwen2.5-VL-3B-Instruct | 3B | mlx | 0.20 | 4131 | unclear 10% |
| Qwen3.5-0.8B-MLX | 0.8B | mlx | 0.38 | 2237 | ⚠ ほぼ全noへ退化。tIoU 0.38は識別力ではなく"no"基準の見かけ |
| SmolVLM2-500M-Video | 0.5B | transformers/fp32 | 0.17 | — | ⚠ 退化（多くのunitで全yes/unclear・全40フレーム同一） |
| SmolVLM2-256M-Video | 0.26B | transformers/fp32 | 0.14 | — | ⚠ 退化（f002は全unclear・f003は76%unclear） |
| LFM2.5-VL-1.6B | 1.6B | mlx | — | — | 未計測：mlx-vlm 0.6.3の `lfm2_vl` layer_normバグでロード不可（修正版0.6.4はwheel入手不可） |

列の意味：

- **mean tIoU** — Opus参照と「両方が検出した」イベント区間の時間的IoU（重なり率・0〜1、1で完全一致）の平均。片方しか検出しなかったイベントは平均に入らない。
- **ピークメモリ (MB)** — 推論中に使われたGPU（Metal）メモリのピーク量。単位はMB（メガバイト。1000 MB ≒ 1 GB）。mlxの実測値で、値が大きいほど推論時に多くメモリを消費する。SmolVLM2（transformers経路）は計測手段が無いため「—」。本機は24 GB（≒24000 MB）で、全モデル2〜5.4 GB程度に収まり余裕がある。

### この表の読み方（重要・数値の落とし穴）

- **これは精度ではなくOpus参照との一致**。Opusも人手GTではないので、上位＝正しいとは限らない。
- **mean tIoUは両者が検出したイベント区間のみの平均**で、区間の開始/終了フレームのズレに厳しい。単一フレーム入力では0.14〜0.38に留まる。
- **⚠のモデルは定数出力へ退化している**。特にQwen3.5-0.8Bは実質全"no"で、質問の実際の答えの約8割が"no"のため、tIoU 0.38は識別力ではなく偶然の見かけ。数値の高さで上位と誤解しない。

### 所見（単一フレーム入力・pilot6での予備的傾向）

- **単一フレーム入力の限界**: ローカル勢はどれもmean tIoUが0.14–0.38と低い。Opus参照は「直近5フレームの因果窓」で手順の状態遷移を追えたが、ローカルは最新1フレームのみで時間文脈が無く、動作の途中/完了の区別や短い動作の検出でOpusと区間がずれる。2枚以上入力の効果検証は次フェーズ。
- **中量級（2–4B）が総じてOpusに近い**。特にQwen3-VL-4Bが最もOpus寄り。Qwen3.5系（2B/4B）はyesを出しすぎてOpusの全イベントを拾う一方、検出区間が膨らんでtIoUが下がる。
- **1B未満は識別力が崩れる**。Qwen3.5-0.8Bは実質全noへ、SmolVLM2-500M/256Mは全yes/unclearへ退化。手順判定の解像度に達していない。
- **SmolVLM2はbfloat16×MPSで視覚が壊れる**（自由記述が「A--」など全フレーム文字化け）。**fp32で視覚が正常化**することを1フレーム自由記述で確認し、fp32で計測した（`observe.py` TransformersObserver）。2.2Bは健全だが500M/256Mはfp32でもタスクに追従できない。

可視化は `sop-replay --runs-dir runs`（イベント検出を日本語ラベルで表示。1枚HTMLでdataset/sample/model切替）。

再現: `python3 tools/benchmark/reference_tiou.py --reference 20260711-factory_ego-opus48-online5-r1 --json reports/data/reference_tiou_pilot6_vs_opus48.json`

今後の予定:

1. 複数フレーム入力（2枚〜）への拡張と、単一フレームからの精度変化の検証（`observe.py` は現在単一画像入力）
2. pilot6を残り14 unitへ拡大
3. 人手GTの作成と、それを入力にしたevaluation runでの正式評価（precision/recall/F1/tIoU）

## 評価の原則（変わらない）

- 正式なprecision、recall、F1、tIoUは人手GT revisionを固定したevaluation runでのみ計算する
- reference予測との一致は「大型モデルとどれだけ同じ区間を見たか」であり精度ではない
- Factory Egoのモデル出力（Opus含む）をground truthへ昇格しない
