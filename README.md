# small-vlm-sop-check

<p align="right"><a href="#english"><b>English</b></a></p>

作業動画が手順書どおりに行われたかを、ローカルの小型VLM（Qwen3-VL / Apple Silicon）だけで判定するデモ。

作業を撮った動画を渡すと、決められた手順（例：「点火は指差し確認より前」「手袋は着けない」）が守られているかを **PASS / FAIL** で返す。クラウドにも大型モデルにも投げない。

肝は **「観察」と「判定」を分けている** こと。VLMはフレームごとの見た目を yes / no で答えるだけで、順序や遵守のロジックは決定論的なルールエンジンが受け持つ。VLMに時刻の前後関係まで推論させると単純な比較すら間違えるため、そこは機械に任せる。

<p align="center">
  <img src="docs/replay_demo.gif" alt="再生ビューアのデモ" width="560"><br>
  <sub><a href="#結果を再生する">再生ビューア</a>：各フレームで VLM が何と答え、どのイベントが検出され（右のタイムライン）、総合判定が PASS / FAIL かを再生できる（同梱の qwen3-4b は PASS）。</sub>
</p>

```mermaid
flowchart LR
    V([動画]) --> F["フレーム抽出<br/>1 fps"]

    subgraph P1["Phase 1 · observe（フレーム毎に VLM）"]
        direction TB
        f0["f0"] --> a0["質問に yes / no"]
        f1["f1"] --> a1["質問に yes / no"]
        fN["fN"] --> aN["質問に yes / no"]
    end

    F --> f0
    F --> f1
    F --> fN

    a0 --> L[("answer_log<br/>全フレーム × 質問")]
    a1 --> L
    aN --> L

    subgraph P2["Phase 2 · judge（全体で1回）"]
        J["events / relations を機械判定"]
    end

    L --> J --> R([PASS / FAIL])

    style a0 fill:#e0ecff,stroke:#3b82f6
    style a1 fill:#e0ecff,stroke:#3b82f6
    style aN fill:#e0ecff,stroke:#3b82f6
    style J fill:#e6f6ec,stroke:#22c55e
```

## しくみ

パイプラインは observe（Phase 1）と judge（Phase 2）の2段。その手前に、人間が手順書を用意する準備が要る。VLMを使うのは Phase 1 だけ。

- **準備（人間）** 動画を見て、守るべき手順を SOP（YAML）に書き下す。何を質問し、何をイベントとみなし、イベント間にどんな前後関係が要るか。
- **Phase 1 — observe（VLM）** 各フレームを VLM に見せ、SOPで決めた質問（例：「手がつまみを触っているか」）に `yes` / `no` / `unclear` を信頼度つきで答えさせる。
- **Phase 2 — judge（ルールエンジン）** 回答を手順ルール（例：「点火は指差し確認より前」）と機械的に突き合わせ、PASS / FAIL を出す。**ここに VLM は使わない。**

## クイックスタート

前提：macOS（Apple Silicon）/ Python ≥ 3.10。`observe`・`run` には [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) が要る（`judge` だけなら不要）。

```bash
pip install -r requirements.txt   # judge だけ使うなら: pip install pyyaml
```

同梱の実データだけで、抽出 → 観察 → 判定を1コマンドで試せる（初回はモデルDLが走る）：

```bash
python src/cli.py run \
  --sop examples/konro_inspection/sop.yaml \
  --video examples/konro_inspection/data/konro_inspection.mp4 \
  --model 4b \
  --out-dir out/
```

一番手軽なのは、観察済みログだけで判定を回すこと（GPU不要・数秒で終わる）：

```bash
python src/cli.py judge \
  --sop examples/konro_inspection/sop.yaml \
  --answer-log examples/konro_inspection/sample_output/answer_log.json
```

## CLI

| コマンド | 内容 |
|---|---|
| `python src/cli.py run --sop --video --model --out-dir` | 抽出 → 観察 → 判定を一気通貫で実行 |
| `python src/cli.py observe --sop --frames-dir --out` | Phase 1 のみ |
| `python src/cli.py judge --sop --answer-log` | Phase 2 のみ |
| `python src/cli.py eval --sop --answer-log` | 観察ログを正解アノテーションと突き合わせて評価（[後述](#正解アノテーションと観察精度の評価)） |
| `python src/cli.py models` | `--model` に使える動作確認済みエイリアス一覧 |

## SOPフォーマット

YAML1ファイルに3セクション書く。役割はそれぞれ違う：

1. **questions** — フレームごとに VLM に聞く質問
2. **events** — 質問への回答が N フレーム以上続いたら「起きた」とみなす条件
3. **relations** — event どうしの前後・同時性・禁止を宣言

`questions` → `events` → `relations` の順に、observe が答えたものを judge が検出条件に変換し、その検出結果どうしの関係をチェックする。

```yaml
sop:
  id: konro_inspection
  name: コンロ始業前点検
  domain_hint: "これはガスコンロの点検作業を上から撮った動画の1フレームです"

questions:                           # Phase 1 — VLMへのプロンプトをここから自動生成
  - id: knob
    ask: "手がコンロ手前のつまみを操作しているか"
    values: ["yes", "no"]            # クォート必須。裸の yes/no は YAML の真偽値になる

events:                              # Phase 2 — 何を検出するか
  ignite:
    evidence: "knob==yes"
    min_frames: 2                    # 持続する動作はここを上げてノイズ耐性を持たせる
  point1:
    evidence: "pointing==yes"
    occurrence: 1                    # 時系列N番目を明示（宣言順に依存しない。後述）

relations:                           # Phase 2 — イベント間の時間的関係
  - ignite before point1
  - point2  overlaps battery         # 同時に起きてよい
  - not gloves_worn                  # 一度も検出されてはいけない
```

上の例を読み下すと：`knob`（つまみを触っているか）を毎フレーム VLM に聞く（question）→ `knob==yes` が2フレーム以上続いたら `ignite`（点火）が起きたとみなす（event）→ `ignite` は `point1` より前に起きなければならない（relation）。

**relations は3つだけ**

手順書の文は必ずこの3つのどれかに翻訳できる：

| 手順書の文 | relation | 意味論（実装） |
|---|---|---|
| 「〜してから〜する」 | `A before B` | 代表時刻（区間の平均時刻）の比較（± `order_tolerance_s`）。Allen区間代数の before / meets の粗視化 |
| 「〜しながら／〜の間に」 | `A overlaps B` | 検出区間の交差。Allenの「交わる」関係群（overlaps / during / starts / …）の粗視化 |
| 「〜してはいけない」 | `not A` | 一度も検出されないこと（DECLARE の absence） |

区間同士の時間関係は [Allen の区間代数](https://en.wikipedia.org/wiki/Allen%27s_interval_algebra)で13種類に尽きるが、1fps＋VLMの境界ノイズの下では meets と overlaps のような細かい区別は観測不能。**境界ノイズで壊れない同値類まで潰したのがこの3語彙**で、だから安易に増やさない。`before` だけ点（代表時刻）ベースなのも、順序判定を境界のブレから守るため。

**occurrence（何回目か）**

同じ質問（例：「指差ししてる？」）を動画中で何度も聞くので、「1回目」「2回目」を区別する番号。指定しないと「YAMLに書いた順番」でなんとなく割り振られ、書く順番を変えると結果が変わってしまう（`tests/test_judge.py::test_occurrence_is_order_independent` で検証）。

**expect（正解＝Phase 0・任意）**

その動画に対する**期待判定と「なぜ違反か（理由）」**を宣言する。judge の結果と突き合わせ、verdict だけでなく違反の理由まで当てられたかを採点できる（[ベンチマーク](#ベンチマーク)はこれで評価している）。省略可。

```yaml
expect:
  verdict: FAIL            # PASS | FAIL（この動画に対する正しい判定）
  because:                 # FAILの「理由」= 当てるべき違反（PASS時は不要）
    - relation: "battery_check before ignite"   # この関係が…
      kind: order_reversed                       # …順序逆転で破られること
    # 未検出（工程の欠落）を当てさせたい場合は event で指定:
    # - event: gloves_check
    #   kind: missing
```

`kind` は違反の種類：`order_reversed`（before の順序が逆）/ `missing`（関係の一方が未検出）/ `overlap_missing`（overlaps なのに離れている）/ `overlap_forbidden`（not_overlaps なのに重なる）/ `forbidden`（`not X` なのに検出）。同梱3条件の正解は各SOPの `expect` に入っている（`sop.yaml` = PASS / `sop_wrong_order.yaml` = 順序逆転 / `sop_missing_step.yaml` = 欠落）。

## 使えるモデル

`--model` にはエイリアス（`qwen3-4b`・`internvl3-2b`・`minicpm-4.6` など）か HF / mlx-community のフルIDを渡せる。一覧は `python src/cli.py models`。既定は `qwen3-4b`（同梱動画で総合 PASS する）。

実際に動くことを確認済みのモデル：

| エイリアス / ID | モデル |
|---|---|
| `qwen3-2b` / `qwen3-4b` | Qwen3-VL 2B / 4B（`qwen3-4b` が既定。2B は JSON が崩れやすい・[ベンチマーク](#ベンチマーク)参照） |
| `qwen3.5-0.8b` / `qwen3.5-2b` / `qwen3.5-4b` | Qwen3.5 0.8B / 2B / 4B（早期fusionのネイティブVLM） |
| `lfm2.5-1.6b` | LFM2.5-VL 1.6B（**要 mlx-vlm ≥ 0.6.4**。0.6.3 は lfm2_vl 実装が layer_norm を無条件生成するバグでロード不可） |
| `qwen2.5-3b` | Qwen2.5-VL-3B |
| `internvl3-2b` | InternVL3-2B |
| `gemma4-e2b` | Gemma4-E2B |
| `minicpm-4.6` | MiniCPM-V 4.6（思考モデル・1.3B） |
| `molmo-7b` | Molmo-7B |
| `cosmos-7b` | Cosmos-Reason1-7B（NVIDIA物理推論・思考モデル） |

試して**動かなかった**もの：FastVLM は2ルートとも不可（mlx-community の bf16 版は画像プロセッサが torch 必須、InsightKeeper の MLX 4bit 版は重み名が mlx-vlm の fastvlm 実装と不一致）。InternVL3.5-30B-A3B は 4bit でも重みだけで約17GBあり、24GB RAM の Mac では非現実的。

観察の生成まわりは3つのオプションで調整する：

- `--prefill STR`（既定 `{"`）— アシスタント応答の先頭に差し込む文字列。JSONを最初のキーの途中まで固定することで、**Molmo のように最初のトークンで EOS を出して空応答になるモデルや、MiniCPM-V のように思考（`<think>`）でトークンを使い切るモデルでも、既定のまま全フレームでクリーンな yes/no JSON を返させられる**。思考の連鎖をあえて使いたい場合は `--prefill ''` で無効化する。
- `--max-tokens N`（既定200）— 1フレームあたりの最大生成トークン。`--prefill ''` で思考モデルを回す場合は1024程度に上げる。
- `--thinking {auto,on,off}`（既定auto）— 思考モードの明示指定。チャットテンプレートが対応する場合のみ有効。

## ベンチマーク

同梱の `konro_inspection`（同一の16フレーム / 1fps の作業動画）を **3つのSOP条件** で判定させ、各ローカルVLMを評価した。動画は正しい手順どおりなので、正解は「正解手順 = PASS」「順序違反・ステップ欠落 = FAIL、かつ **なぜ違反かを正しく指せること**」。観察は全モデル既定の `--prefill '{"'` で96セル全てに回答する。

各条件の正解（PASS か／違反の「理由」は何か）は、その条件の SOP YAML の `expect:` に宣言してある（[SOPフォーマット](#sopフォーマット)の `expect` を参照）。judge はこの `expect` と実際の判定を突き合わせ、`python src/cli.py judge` が `[正解照合] … 箇所特定 ✓/✗` を出す。

### 判定精度（正しい手順は PASS、違反は"理由"まで当てられるか）

違反2条件の ✅ は「FAIL を出したか」ではなく **「なぜ違反かを正しく指したか」**（順序違反なら `battery_check before ignite` の順序逆転、欠落なら `gloves_check` の未検出）。単に FAIL を出すだけなら全モデル当たるが、それは "常に FAIL" でも当たる 2/3 のベースラインにすぎない。

| モデル | サイズ | 正解手順<br>→ PASS | 順序違反<br>→ 順序逆転を指摘 | ステップ欠落<br>→ 欠落を指摘 | 正答 |
|---|---:|:---:|:---:|:---:|:---:|
| Qwen3-VL-4B | 4B | ✅ | ✅ | ✅ | 3/3 |
| Qwen3.5-4B | 4B | ❌ | ✅ | ✅ | 2/3 |
| Qwen2.5-VL-3B | 3B | ❌ | ✅ | ✅ | 2/3 |
| MiniCPM-V 4.6 | 1.3B | ❌ | ✅ | ✅ | 2/3 |
| InternVL3-2B | 2B | ❌ | ✅ | ✅ | 2/3 |
| Molmo-7B | 7B | ❌ | ✅ | ✅ | 2/3 |
| Gemma4-E2B | 2B | ❌ | ❌ | ✅ | 1/3 |
| Cosmos-Reason1-7B | 7B | ❌ | ❌ | ✅ | 1/3 |
| Qwen3.5-2B | 2B | ❌ | ❌ | ✅ | 1/3 |
| Qwen3.5-0.8B | 0.8B | ❌ | ❌ | ✅ | 1/3 |
| LFM2.5-VL-1.6B | 1.6B | ❌ | ❌ | ✅ | 1/3 |
| Qwen3-VL-2B | 2B | ❌ | ❌ | ✅ | 1/3 |

*（✅ = その条件の正解を当てた。違反列は理由の一致まで要求する）*

**正しい手順を PASS と見抜けたのは Qwen3-VL-4B だけ**（過検出による偽陽性の FAIL を出さないのが難所）。さらに順序違反では、**Gemma4-E2B と Cosmos-Reason1-7B は順序逆転を捕まえたのではなく、電池を一度も検出できず（`battery_check` 未検出）に FAIL している**——どんな誤順序SOPでも「電池が見えない」だけで FAIL するので理由は当てていない。verdict の一致だけでは横並びに見える差が、理由の照合で表に出る。

2026年の新顔（Qwen3.5・LFM2.5-VL）で 1/3 のものも同型で、順序違反の FAIL はいずれも「2回目の指差し（`point2`）未検出」によるもので順序逆転は指摘できていない。注目は同サイズ対決の **Qwen3.5-4B vs Qwen3-VL-4B**：最新世代の早期fusionネイティブVLMでも 2/3（`battery` は完璧だが `point2` を取りこぼして正解手順を偽陽性 FAIL）で、この用途では視覚特化系の Qwen3-VL-4B に及ばなかった。Qwen3-VL-2B は回答の中身以前に **JSON 形式が半数のフレームで崩れる**（クォート欠落・同一キーの繰り返し。mlx-vlm 0.6.3 実測）ため表中最下位。

### どの観察が弱いか（人手の正解アノテーション基準）

各モデルの観察を、人間が動画に付けたイベント区間の正解（`examples/konro_inspection/ground_truth.json`。付け方は[後述](#正解アノテーションと観察精度の評価)）と突き合わせる。`python src/cli.py eval` で再現できる。列は3層：

- **relations 正答** — SOP に書いた6つのルール（`ignite before point1` のような relations）それぞれについて、正解区間で評価したときと同じ結論（成立／違反）を出せた数。**判定の合否を直接決めるのはこのスコア**
- **mean tIoU** — 検出したイベント区間と正解区間の重なり（境界の完全一致は要求しない）
- 質問別 — 正解区間から導出したフレームラベルとの一致率（evidence の値 `q==yes` での二値採点・16フレーム）

| モデル | relations<br>正答 | mean<br>tIoU | 総合 | 点火<br>`knob` | 炎<br>`flame` | 指差し<br>`pointing` | グリル<br>`grill` | 電池<br>`battery` | 手袋<br>`gloves` |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-4B | 6/6 | 0.78 | 96% | 94% | 100% | 81% | 100% | 100% | 100% |
| Qwen2.5-VL-3B | 5/6 | 0.56 | 83% | 50% | 100% | 75% | 94% | 81% | 100% |
| Cosmos-Reason1-7B | 4/6 | 0.58 | 81% | 44% | 100% | 100% | 62% | 81% | 100% |
| Qwen3.5-4B | 4/6 | 0.51 | 77% | 38% | 100% | 75% | 50% | 100% | 100% |
| LFM2.5-VL-1.6B | 4/6 | 0.32 | 50% | 31% | 100% | 44% | 12% | 12% | 100% |
| Qwen3.5-2B | 4/6 | 0.31 | 47% | 25% | 100% | 31% | 12% | 12% | 100% |
| InternVL3-2B | 4/6 | 0.31 | 50% | 31% | 100% | 25% | 12% | 31% | 100% |
| Gemma4-E2B | 3/6 | 0.62 | 82% | 88% | 100% | 31% | 88% | 88% | 100% |
| Molmo-7B | 3/6 | 0.38 | 68% | 50% | 100% | 31% | 31% | 94% | 100% |
| Qwen3.5-0.8B | 3/6 | 0.14 | 67% | 62% | 94% | 31% | 38% | 75% | 100% |
| Qwen3-VL-2B | 3/6 | 0.10 | 77% | 75% | 94% | 88% | 44% | 62% | 100% |
| MiniCPM-V 4.6 | 2/6 | 0.51 | 83% | 44% | 100% | 69% | 88% | 100% | 100% |

**relations を 6/6 正答できたのは Qwen3-VL-4B だけで、これがそのまま唯一の PASS に対応する**（上の判定精度表と一致）。その Qwen3-VL-4B も100%ではない——指差しを正解区間外で3フレーム過検出している（`pointing` 81%）——が、イベントの前後関係は正解と同じ結論になるので判定には響かない。境界のズレやノイズは注釈ではなく指標側で吸収する、という設計の実例。

3層を並べると乖離が見える。MiniCPM-V 4.6 はセル一致率83%と上位なのに relations 正答は2/6で最下位——見た目はだいたい合っているが、**判定を分ける境目でちょうど間違える**。逆に Qwen3-VL-2B は JSON 崩壊で mean tIoU 0.10 なのにセル一致率は77%と高く出る（二値採点では「答えられなかった」が「no」と同じ扱いになり、大半のフレームが陰性なので稼げてしまう）。**セル一致率だけでモデルを選ぶのは危険で、合否に効くのは relations の正答**。なおサイズは効かない（2Bの Gemma4 が総合82%で 7B の Molmo を上回る）という結論は、採点を人手の正解基準に変えても変わらなかった。

<details><summary>再現方法</summary>

```bash
# 観察(1回)→ 3条件で判定
# lfm2.5-1.6b は mlx-vlm>=0.6.4 が必要（0.6.3はロード不可）
for m in qwen3-4b qwen3-2b qwen3.5-4b qwen3.5-2b qwen3.5-0.8b lfm2.5-1.6b \
         gemma4-e2b cosmos-7b qwen2.5-3b minicpm-4.6 internvl3-2b molmo-7b; do
  python src/cli.py observe \
    --sop examples/konro_inspection/sop.yaml \
    --frames-dir examples/konro_inspection/sample_output/frames \
    --model "$m" --out "out/al_$m.json"
  for cond in sop sop_wrong_order sop_missing_step; do
    python src/cli.py judge \
      --sop "examples/konro_inspection/$cond.yaml" --answer-log "out/al_$m.json"
  done
  # 観察精度(relations正答・tIoU・質問別)は正解アノテーションとの突き合わせで出す
  python src/cli.py eval \
    --sop examples/konro_inspection/sop.yaml --answer-log "out/al_$m.json"
done
```

questions は3つのSOPで共通なので観察は1回でよい。観察精度の正解は人手アノテーション `examples/konro_inspection/ground_truth.json`（tools/annotator で作成。16フレーム全部を目視して付けた区間）。
</details>

## 正解アノテーションと観察精度の評価

ベンチマークの一次基準は expect（verdict＋違反理由）の一致だが、「なぜそのモデルが外すのか」を診断するにはイベント区間そのものの正解が要る。人手の正解付け → 評価は2コマンドで回る：

```bash
# 1) ブラウザで各イベントの「実際に起きた区間」を注釈する
#    クリック2回（開始・終了）×イベント数で終わる。操作のたびに自動保存・途中再開可。
#    既定では同梱サンプルを開き examples/konro_inspection/ground_truth.json に保存する
python tools/annotator/serve.py

# 2) 観察ログを正解と突き合わせる
python src/cli.py eval \
  --sop examples/konro_inspection/sop.yaml \
  --answer-log examples/konro_inspection/sample_output/answer_log.json
```

アノテーションが記録するのは**事実（いつ何が起きたか＝区間）だけ**。順序や遵守の「べき」は SOP の relations が持ち、評価は両者から機械的に導出する——注釈者が関係を定義することはない。`eval` は3層を出す：

- **イベント区間** — 検出区間 vs 正解区間の tIoU。境界の完全一致は要求しない（境界±数フレームは人間同士でも割れる）。Ego4D 等の時間的アクション検出と同じく、許容誤差はアノテーション側ではなく指標のしきい値（0.1 / 0.3 / 0.5）で吸収する
- **relations の正答** — SOP の各ルール（relation）を正解区間で評価した結論と、検出区間で評価した結論が同じか。合否を実際に分けるのはここで、tIoU が低くても前後関係の結論が正解と同じなら判定は正しい
- **フレーム回答** — 正解区間から導出したフレームラベルと VLM 回答の precision / recall（参考値）

正解区間は **SOP 定義の自己検証**にも使える：正解区間を SOP の規則で評価した `gt_verdict` が `expect.verdict` と食い違えば、SOP の翻訳かアノテーションのどちらかが間違っている（`eval` が ⚠ で知らせる）。SOP は「書く → 正解動画で検証 → 直す」というテスト駆動で作れる。

## 結果を再生する

観察・判定の結果を、フレーム画像と一緒にブラウザで再生できる：

```bash
python tools/replay_viewer/build.py   # tools/replay_viewer/replay.html を生成
```

出力は依存ファイルのない1枚のHTML（フレーム画像も埋め込み済み）で、ダブルクリックで開くだけで動く。「今どのフレームで」「VLMが各質問に何と答え」「どのイベントが検出されて」「最終判定が PASS / FAIL か」を1画面で確認できる。`replay.html` はフレーム画像を base64 で埋め込む生成物のため git には含めない（`frames/` は同梱済みなので上記コマンドですぐ作れる）。

**ヘッダのプルダウンでモデルを切り替えられる**（既定で `examples/konro_inspection/sample_output/models/` の12モデルを束ねる）。同じ動画・同じSOPで、Qwen3-VL-4B が PASS する一方、他モデルがどの質問を過検出して FAIL に至るかを見比べられる——ベンチマークの数字を実際のフレームで確かめられる。

- `--sop examples/konro_inspection/sop_wrong_order.yaml` を渡すと、全モデルを順序違反SOPで判定した様子を見られる
- `--answer-log <path>` を渡すと、単一の観察ログだけを表示する（モデル切替なし）
- `--models-dir <dir>` で別のモデルログ群（`<表示名>.json`）に差し替えられる
- SOPと同じディレクトリに `ground_truth.json`（[正解アノテーション](#正解アノテーションと観察精度の評価)）があれば、正解区間の帯（□）と tIoU を検出区間に自動で重ねる（`--ground-truth <path>` で明示も可）

## リポジトリ構成

```
small-vlm-sop-check/
├── src/
│   ├── observe.py   # Phase 1: questionsからプロンプト生成 + VLM呼び出し + 信頼度抽出
│   ├── judge.py     # Phase 2: events/relations ルールエンジン
│   ├── extract.py   # 動画 -> フレーム(cv2)
│   ├── sop.py       # SOP YAMLの読み込み・検証
│   ├── evaluate.py  # 正解アノテーションとの突き合わせ(tIoU・relations正答)
│   └── cli.py       # `run`/`observe`/`judge`/`eval` サブコマンド
├── examples/konro_inspection/   # 実動画・フレーム・観察ログ・SOP3種(注釈すると ground_truth.json もここに入る)
├── tools/replay_viewer/         # 結果をブラウザで再生する1枚HTMLの生成（replay.htmlはbuild.pyで生成・git管理外）
├── tools/annotator/             # 正解区間をブラウザで注釈するツール(標準ライブラリのみ・自動保存)
└── tests/                       # 実データに対する回帰テスト(VLM不要)
```

## English

**small-vlm-sop-check** checks whether a work video was performed according to a written procedure (SOP), using only a small local VLM (Qwen3-VL on Apple Silicon). No cloud, no large models.

Feed it a video of a task and it returns **PASS / FAIL** for whether the defined steps were followed (e.g. "ignition must come before the point-and-call check", "no gloves worn").

The core idea is **separating observation from judgement**. The VLM only answers yes/no about what each frame looks like; a deterministic rule engine handles ordering and compliance. Asking a small VLM to also reason about temporal order makes it fail even trivial comparisons, so that part is left to code.

### How it works

Two automated stages, preceded by human prep. The VLM runs only in Phase 1.

- **Prep (human)** — Watch the video and write the procedure as an SOP (YAML): what to ask per frame, what counts as an event, and which temporal relations must hold.
- **Phase 1 · observe (VLM)** — Show each frame to the VLM and have it answer the SOP questions with `yes` / `no` / `unclear`, with confidence.
- **Phase 2 · judge (rule engine)** — Match the answers against the procedure's rules and emit PASS / FAIL. **No VLM here.**

### Quickstart

macOS (Apple Silicon), Python ≥ 3.10. `observe` / `run` need [mlx-vlm](https://github.com/Blaizzy/mlx-vlm); `judge` alone does not.

```bash
pip install -r requirements.txt          # judge only: pip install pyyaml

# full pipeline on the bundled sample (downloads the model on first run)
python src/cli.py run \
  --sop examples/konro_inspection/sop.yaml \
  --video examples/konro_inspection/data/konro_inspection.mp4 \
  --model 4b --out-dir out/

# fastest check: judge a pre-recorded observation log (no GPU, seconds)
python src/cli.py judge \
  --sop examples/konro_inspection/sop.yaml \
  --answer-log examples/konro_inspection/sample_output/answer_log.json
```

### Key finding

Across three SOP conditions (correct / wrong-order / missing-step) on the bundled `konro_inspection` clip, every model can FAIL the two violation cases — but that is the 2/3 baseline you get from "always say FAIL". Only **Qwen3-VL-4B** also recognises the correct run as PASS (3/3). Not over-detecting — avoiding false-positive FAILs — is the real difficulty, and it hinges on observation quality (Phase 1), not parameter count (the 2B Gemma4 beats the 7B Molmo/Cosmos on observation agreement).

> For the SOP format, full benchmark tables, model list and the replay viewer, see the Japanese sections above.

## ライセンス

MIT — [LICENSE](LICENSE) を参照。
