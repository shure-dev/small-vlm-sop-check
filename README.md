# small_vlm_video_analysis

作業動画が宣言的なSOP（手順書）通りかを、ローカルの小型VLM（Qwen3-VL, Apple Silicon）だけで判定する。クラウド不使用・大型モデル不使用。

- **Phase 0（正解を決める）**: 動画を実際に見て、フォーマットに従って正解の手順を人間が確認する。
- **Phase 1（observe）**: VLMに動画の各フレームを見せて、あらかじめ決めた質問（例:「手がつまみを触っているか」）に yes / no / unclear で答えさせる。答えがどれくらい確かかも一緒に記録する。
- **Phase 2（judge）**: VLMの答えを、あらかじめ決めた手順のルール（例:「点火は指差し確認より前」）と機械的に突き合わせて、PASS / FAIL を出す。ここにVLMは使わない。


## 動作環境

- macOS（Apple Silicon）
- Python >= 3.10
- `observe`/`run`コマンドには [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) が必要

## インストール

```bash
pip install -r requirements.txt   # judgeコマンドだけ使うなら: pip install pyyaml
```

## クイックスタート

```bash
python src/cli.py run \
  --sop examples/konro_inspection/sop.yaml \
  --video examples/konro_inspection/data/konro_inspection.mp4 \
  --model 4b \
  --out-dir out/
```

フレーム抽出 → VLM観察 → 判定を1コマンドで実行する。同梱の実データだけで動く。

VLMを動かさず、観察済みログだけで判定することもできる:

```bash
python src/cli.py judge \
  --sop examples/konro_inspection/sop.yaml \
  --answer-log examples/konro_inspection/sample_output/answer_log.json
```

## CLI

| コマンド | 内容 |
|---|---|
| `python src/cli.py run --sop --video --model --out-dir` | 抽出→観察→判定を一気通貫で実行 |
| `python src/cli.py observe --sop --frames-dir --out` | Phase 1のみ |
| `python src/cli.py judge --sop --answer-log` | Phase 2のみ |
| `python src/cli.py models` | `--model` に使える動作確認済みエイリアス一覧 |

## モデルを切り替える

`--model` にはエイリアス（`qwen3-4b`・`internvl3-2b`・`minicpm-4.6` など）かHF/mlx-communityのフルIDを渡せる。使えるエイリアスは `python src/cli.py models` で一覧できる。既定は基準の `qwen3-4b`（同梱動画でPASSする）。

同じ動画・同じSOPでも、観察するVLMを変えると結果は割れる。小型モデルほど「yes」を出しすぎたり短いノイズを拾ったりして、決定論的なjudgeがそれを（入力に忠実に）誤判定に変える。観察品質（Phase 1）がそのまま最終判定を左右する、という観察と判定の分離ならではの挙動が見える。

観察の生成まわりは3つのオプションで調整する：

- `--prefill STR`（既定 `{`）— アシスタント応答の先頭に差し込む文字列。JSONの開き括弧を先に置くことで、**Molmoのように最初のトークンでEOSを出して空応答になるモデルや、MiniCPM-V/Cosmosのように思考（`<think>`）でトークンを使い切ってJSONに届かないモデルでも、既定のまま全フレーム回答させられる**。思考の連鎖をあえて使いたい場合は `--prefill ''` で無効化する。
- `--max-tokens N`（既定200）— 1フレームあたりの最大生成トークン。`--prefill ''` で思考モデルを回す場合は1024程度に上げる。
- `--thinking {auto,on,off}`（既定auto）— 思考モードの明示指定。チャットテンプレートが対応する場合のみ有効。

### 試せるモデル

実際に動くことを確認済みのモデル（`--model` にエイリアス or フルIDを渡す）：

| エイリアス / ID | モデル |
|---|---|
| `qwen3-2b` / `qwen3-4b` | Qwen3-VL 2B / 4B（`qwen3-4b` が基準） |
| `qwen2.5-3b` | Qwen2.5-VL-3B |
| `internvl3-2b` | InternVL3-2B |
| `gemma4-e2b` | Gemma4-E2B |
| `minicpm-4.6` | MiniCPM-V 4.6（思考モデル・1.3B） |
| `molmo-7b` | Molmo-7B |
| `cosmos-7b` | Cosmos-Reason1-7B（NVIDIA物理推論・思考モデル） |

思考モデル（`minicpm-4.6` / `cosmos-7b`）も既定の `--prefill '{'` のおかげで追加設定なしに全フレーム回答する。SmolVLM・LFM2-VL は torch 必須で動かない。同梱動画で総合PASSするのは基準の `qwen3-4b` のみ。

## ベンチマーク

同梱の `konro_inspection`（16フレーム / 1fps）で、各ローカルVLMの観察精度を測った。指標は **基準モデル Qwen3-VL-4B の観察との一致率**（6質問 × 16フレーム = 96セルの yes/no/unclear 一致率）。Qwen3-VL-4B は同梱動画で唯一 SOP 判定が PASS になるため、事実上の正解として扱う。「判定」は同じ SOP・同じルールエンジンが出す総合結果。全モデル既定の `--prefill '{'` で96セル全てに回答する。

| モデル | サイズ | 一致率 | 判定 |
|---|---:|---:|:---:|
| **Qwen3-VL-4B**（基準） | 4B | **100%** | ✅ PASS |
| Gemma4-E2B | 2B | 91% | ❌ FAIL |
| Cosmos-Reason1-7B | 7B | 86% | ❌ FAIL |
| Qwen2.5-VL-3B | 3B | 75% | ❌ FAIL |
| MiniCPM-V 4.6 | 1.3B | 70% | ❌ FAIL |
| InternVL3-2B | 2B | 49% | ❌ FAIL |
| Molmo-7B | 7B | 47% | ❌ FAIL |

PASSするのは基準のみ。一致率が高くても、他モデルは「ある質問での過検出（yesを出しすぎ）」でイベントの時系列がずれ、決定論的なjudgeがそれを FAIL に変える。**サイズは効かない**（7BのMolmo/Cosmosより2BのGemma4のほうが一致率が高い）。観察品質（Phase 1）がそのまま判定を決める、という本デモの設計思想を裏づける結果。

<details><summary>再現方法</summary>

```bash
for m in qwen3-4b gemma4-e2b cosmos-7b qwen2.5-3b minicpm-4.6 internvl3-2b molmo-7b; do
  python src/cli.py observe \
    --sop examples/konro_inspection/sop.yaml \
    --frames-dir examples/konro_inspection/sample_output/frames \
    --model "$m" --out "out/al_$m.json"
  python src/cli.py judge \
    --sop examples/konro_inspection/sop.yaml --answer-log "out/al_$m.json"
done
```

一致率は各 `out/al_<model>.json` を基準の `examples/konro_inspection/sample_output/answer_log.json` と突き合わせて算出（argmax の一致セル数 / 96）。
</details>

## 結果の再生ビューア

観察・判定の結果を、フレーム画像と一緒にブラウザで再生できる:

```bash
python tools/replay_viewer/build.py   # tools/replay_viewer/replay.html を生成
```

出力は依存ファイルのない1枚のHTML（フレーム画像も埋め込み済み）で、ダブルクリックで開くだけで動く。「今どのフレームで」「VLMが各質問に何と答え」「どのイベントが検出されて」「最終判定がPASS/FAILか」を1画面で確認できる。同梱データから生成済みの `tools/replay_viewer/replay.html` がそのまま開ける。

`--sop` / `--answer-log` / `--frames-dir` / `--out` で別の実行結果に差し替えられる。例えば `--sop examples/konro_inspection/sop_wrong_order.yaml` を渡すと、同じ動画が順序違反でFAILになる様子を見られる。

## SOPフォーマット

YAML1ファイルに3セクション書く。役割はそれぞれ違う：

1. **questions** — フレームごとにVLMに聞く質問
2. **events** — 質問への回答がNフレーム以上続いたら「起きた」とみなす条件
3. **relations** — eventどうしの前後・同時性・禁止を宣言

`questions`→`events`→`relations`の順に、observeが答えたものをjudgeが検出条件に変換し、その検出結果どうしの関係をチェックする。

```yaml
sop:
  id: konro_inspection
  name: コンロ始業前点検
  domain_hint: "これはガスコンロの点検作業を上から撮った動画の1フレームです"

questions:                           # Phase 1 — VLMへのプロンプトをここから自動生成
  - id: knob
    ask: "手がコンロ手前のつまみを操作しているか"
    values: ["yes", "no"]            # クォート必須。裸のyes/noはYAMLの真偽値になる

events:                              # Phase 2 — 何を検出するか
  ignite:
    evidence: "knob==yes"
    min_frames: 2                    # 持続する動作はここを上げてノイズ耐性を持たせる
  point1:
    evidence: "pointing==yes"
    occurrence: 1                    # 時系列N番目を明示(宣言順に依存しない。後述)

relations:                           # Phase 2 — イベント間の時間的関係
  - ignite before point1
  - point2  overlaps battery         # 同時に起きてよい
  - not gloves_worn                  # 一度も検出されてはいけない
```

上の例を読み下すと：`knob`（つまみを触っているか）を毎フレームVLMに聞く（question）→`knob==yes`が2フレーム以上続いたら`ignite`（点火）が起きたとみなす（event）→`ignite`は`point1`より前に起きなければならない（relation）。

**relationsは3つだけ**
- `before` — Aが先、Bが後
- `overlaps` — AとBは同時に起きてもOK
- `not` — これは一度も起きてはいけない

**occurrence（何回目か）**
同じ質問（例:「指差ししてる？」）を動画中で何度も聞くので、「1回目」「2回目」を区別する番号。指定しないと「YAMLに書いた順番」でなんとなく割り振られ、書く順番を変えると結果が変わってしまう（`tests/test_judge.py::test_occurrence_is_order_independent`で検証）。


## リポジトリ構成

```
small_vlm_video_analysis/
├── src/
│   ├── observe.py   # Phase 1: questionsからプロンプト生成 + VLM呼び出し + 信頼度抽出
│   ├── judge.py     # Phase 2: events/relations ルールエンジン
│   ├── extract.py   # 動画 -> フレーム(cv2)
│   ├── sop.py       # SOP YAMLの読み込み・検証
│   └── cli.py       # `run`/`observe`/`judge` サブコマンド
├── examples/konro_inspection/   # 実動画・フレーム・観察ログ・SOP3種
├── tools/replay_viewer/         # 結果をブラウザで再生する1枚HTMLの生成（生成済みreplay.html同梱）
└── tests/                       # 実データに対する回帰テスト(VLM不要)
```

より野心的なフォーマット（時相match木、非視覚ステップ、judgeモデルへのエスカレーション）の構想もあるが、本リポジトリはエンドツーエンドで検証済みの部分だけを実装している。

## ライセンス

MIT — [LICENSE](LICENSE) を参照。
