# small-vlm-sop-check

[English](README.en.md) | [ドキュメント](docs/README.md)

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)
![Apple Silicon / MLX](https://img.shields.io/badge/Apple%20Silicon-MLX-black)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Status: experimental](https://img.shields.io/badge/status-experimental-orange)

**Small, offline VLMs for industrial egocentric video — detecting SOP step intervals, frame by frame, toward real-time streaming.**

工場・製造現場の一人称視点（egocentric）作業動画から、手順書（SOP）の各ステップがいつ実行されたか（イベント区間）を、ローカルの小型VLMだけで検出・評価する実験フレームワークです。

<p align="center">
  <img src="docs/assets/replay_demo.gif" alt="VLMの回答、検出イベント、正解区間を表示するreplay viewer" width="640"><br>
  <sub>Replay viewer：フレームごとの回答、検出イベント、人手の正解区間を1画面で確認できます。</sub>
</p>

## なぜ作っているか

- 🏭 **産業特化** — 映画・実況・キッチン系egocentric（Ego4Dなど）の一般的な動画理解ではなく、工場・製造の現場作業に特化した動画解析を目指します
- ✅ **手順が主役** — 動画を要約・キャプショニングするのではなく、「SOPのこのステップを実行したか」に**yes / noの粒度**で答え、各ステップが起きた区間を検出します。ローカルVLMは各フレームへの質問に答えるだけで、区間の導出は決定論的なルールが行います
- 🔒 **オフラインで完結** — 現場の映像を外部へ送らず、Apple Silicon上のローカル小型VLMだけで動きます
- ⏱️ **ストリーミング志向** — 動画を丸ごと入力するのではなく、フレームを手前から順に処理する因果的な設計を基本とし、リアルタイムのストリーミング処理を見据えます
- 🎓 **学習まで見据える** — 評価で終わらせず、人手の正解データを整備し、産業特化の小型VLMの学習・改善へ繋げます

## できること

- 動画とSOPから、各手順ステップのイベント区間を検出
- 人手アノテーションを正解として、イベント区間とフレーム回答の精度を評価
- 注釈、推論、評価、結果の再生までをCLIで再現
- Apple Silicon上のローカルVLM 15モデルで同一条件を比較

## まず試す

Python 3.10以上が必要です。最初はVLMを使わず、同梱の回答ログ（VLMがフレームごとの質問に答えた記録、`answer_log.json`）から決定論的ルールでイベント区間を検出します。

```bash
python3 -m pip install -e .

sop-check detect \
  --sop datasets/konro_inspection/sops/konro_inspection/konro_inspection.yaml \
  --answer-log datasets/konro_inspection/fixtures/reference_outputs/answer_log.json
```

成功すると、各イベントの検出区間が表示されます。

```text
event          status          t(s)  span(idx)
knob           detected         3.0  1-5
flame          detected         3.0  3-3
pointing[1]    detected         4.5  4-5
pointing[2]    detected        12.2  10-14
...
gloves         NOT_DETECTED       -

検出: 5/6 イベント
```

同じ動作が複数回起きた場合（この例では指差し `pointing`）は、同じイベントの区間として複数検出されます。

結果をブラウザで確認するには、replay viewerを生成します。

```bash
sop-replay
```

## なぜVLMに区間の導出までさせないのか

VLMには、フレームごとの視覚的な質問だけを任せます。回答列から区間を導く処理（持続時間の要求、短いノイズの橋渡し、複数回の出現の切り出し）はルールエンジンが行います。VLMの自然文推論に時刻や順序の比較を委ねると、単純な比較すら間違えることを実験で確認しています。

```mermaid
flowchart LR
    A[動画] --> B[フレーム抽出]
    B --> C[VLMが質問に回答<br/>フレームごとの yes / no]
    C --> D[決定論的ルールで<br/>イベント区間を検出]
    D --> E[人手の正解区間と評価<br/>tIoU / フレーム一致]
```

この分離には、次の利点があります。

- モデルが何を見誤ったかと、ルールが何を区間としたかを切り分けられる
- 同じ回答ログに対してイベント定義だけを変え、検出をすぐに再実行できる
- 人手の事実、モデルの予測、評価結果を混ぜずに管理できる

設計の詳細は[事実・予測・評価を分離するADR](docs/decisions/0001-separate-facts-predictions-evaluations.md)を参照してください。

## 自分の動画で実行する

### 1. SOPを定義する

SOPには、検出したいイベントを記述します。**イベント = 「〜している」という記述文**で、VLMが各フレームでその成否をyes/noで判定し、"yes" の連続区間が出現になります。

```yaml
sop:
  id: my_inspection
  name: 点検作業
  domain_hint: "作業台を上から撮影した点検動画"

events:
  - id: knob
    ask: "手がつまみを操作している"
    values: ["yes", "no"]
    min_frames: 2
  - id: pointing
    ask: "人が対象を指差している"
    values: ["yes", "no"]
```

全フィールドは[SOPフォーマット](docs/reference/sop-format.md)にあります。

### 2. 人手で正解区間を付ける

注釈ツールで、動画中に各イベントが実際に起きた区間を記録します。ここで作る `ground_truth.json` は、人手で確認した事実だけを含みます。

```bash
sop-annotate            # datasets/ 配下のunitを台帳化し、ヘッダのプルダウンで選ぶ
```

ブラウザ上で「データセット → unit」を切り替えながら、複数unitを1プロセスで注釈できます。区間はタイムラインを**ドラッグ**して直接引き、本体を掴んで移動・両端のつまみで伸縮できます（キーボード <kbd>i</kbd>/<kbd>o</kbd> でも指定可）。イベントの追加・削除、イベントの記述文、撮影状況のヒント（`domain_hint`）もカード上で直接編集でき、SOP YAML へ検証付きで即書き戻されます。すべて自動保存です。詳細は [注釈ツールの仕様](docs/reference/annotator.md) を参照。

台帳を使わず単一unitだけ開くこともできます。

```bash
sop-annotate --sop path/to/sop.yaml --frames-dir path/to/frames --fps 2.0
```

### 3. VLMに回答させ、区間を検出する

MLXバックエンドを使う場合は、macOS / Apple Silicon環境でVLM依存を追加します。

```bash
python3 -m pip install -e ".[vlm]"

sop-check run \
  --sop path/to/sop.yaml \
  --video path/to/video.mp4 \
  --model qwen3-4b \
  --out-dir out/my-run
```

初回実行時はモデルのダウンロードが発生します。回答の収集と区間検出は別々にも実行できます。

```bash
sop-check observe \
  --sop path/to/sop.yaml \
  --frames-dir path/to/frames \
  --model qwen3-4b \
  --out out/answer_log.json

sop-check detect \
  --sop path/to/sop.yaml \
  --answer-log out/answer_log.json
```

モデル一覧と生成設定は[モデルと生成オプション](docs/reference/models.md)を参照してください。

### 4. 人手の正解と突き合わせて評価する

人手の `ground_truth.json` を正解として、イベントごとの検出状態（検出・見逃し・誤検出・正しく未検出）を突き合わせます。

```bash
sop-check eval \
  --sop path/to/sop.yaml \
  --ground-truth path/to/ground_truth.json \
  --answer-log out/answer_log.json
```

評価では、区間の重なり（mean tIoU）と質問ごとのフレーム一致率を出力します。指標の扱いは[評価ポリシー](docs/benchmark/evaluation.md)にあります。

## ベンチマーク

### Konro Inspection

同一の16フレームに対し、15モデルの回答を人手アノテーションの正解区間で評価した完結デモです。上位のみ抜粋:

| モデル | mean tIoU | フレーム一致率 |
|---|---:|---:|
| Qwen3-VL-4B | 0.80 | 96% |
| Qwen2.5-VL-3B | 0.62 | 83% |
| SmolVLM2-2.2B† | 0.60 | 69% |
| Cosmos-Reason1-7B | 0.59 | 81% |
| Gemma4-E2B | 0.56 | 82% |

人手の正解区間をほぼ再現できたのはQwen3-VL-4Bだけでした。ただし、これは単一の短い動画に対する結果であり、一般的な現場性能を示すものではありません。15モデルの全結果と再現コマンドは[Konroベンチマーク結果](docs/benchmark/konro-results.md)にあります（†はtransformersバックエンド計測）。

### Factory Ego

Egocentric-10Kの6工場から作業種類が満遍なく入るよう層化抽出した20 unit（各20秒・2fps・40フレーム）で、手順判定に向けたモデル間の精度比較を準備している開発用データセットです。キッチンや日常系のegocentricデータセット（Ego4Dなど）ではなく工場の一人称視点データを使うのは、産業・製造の現場作業に特化するという本リポジトリの方針のためです。

- クリップ選定と暫定SOP設計には[annotated-egocentric-10k-dataset](https://github.com/fit-alessandro-berti/annotated-egocentric-10k-dataset)（LLM生成・人手検証なし）のtranscriptionを使い、GTとしては扱いません
- 各unitのSOPは手順ステップ粒度の3〜4イベントを持ち、記述は日本語の単文（「作業者が〜している」）です。イベントは抽出フレームの目視で定義します（[イベント定義ガイド](docs/benchmark/events.md)）
- 現行unitは選定・アノテーション過程で閲覧されるため、すべて `dev_seen`
- 人手ground truthは未作成のため、正式なprecision、recall、F1、tIoUは未計測
- upstreamがgated datasetのため、抽出フレームは公開リポジトリに含めず、SHA manifestだけを追跡

SOPイベント定義は実フレーム目視で日本語に再定義済みです（`provisional`・人手レビュー前）。prediction runはClaude Opus 4.8のオンライン推論（直近5フレームの因果窓、20/20 unit）1本と、ローカル小型VLM 12モデルのpilot6サブセットrunがあり、人手GTがないため正式な精度は未評価です。詳細は[Factory Ego README](datasets/factory_ego/README.md)と[モデル比較レポート](reports/model_comparison.md)を参照してください。

## データセットと実験結果の置き場

| パス | 役割 | 公開上の扱い |
|---|---|---|
| `datasets/` | 入力媒体、SOP、人手アノテーション | 事実・仕様 |
| `runs/` | モデル、プロンプト、unitごとの予測 | 実験結果 |
| `evaluations/` | predictionと人手GTの評価 | 派生結果 |
| `reports/` | 複数runの比較と解釈 | レポート |

イベントは各動画unitの `sop_path` が指すSOPで定義します。詳しくは[動画ごとのイベント定義](docs/benchmark/events.md)を参照してください。

## CLI

| コマンド | 内容 |
|---|---|
| `sop-annotate` | 人手の正解区間をブラウザで注釈（unit切替・イベント/SOP編集・タイムライン操作） |
| `sop-check run` | フレーム抽出、VLMの回答収集、区間検出を一括実行 |
| `sop-check observe` | VLMの回答収集だけを実行 |
| `sop-check detect` | 保存済み回答ログからイベント区間を検出 |
| `sop-check eval` | 人手アノテーションに対して回答を評価 |
| `sop-check models` | 登録済みモデルを表示 |
| `sop-replay` | 自己完結型の結果再生HTMLを生成 |

## リポジトリ構成

```text
.
├── src/small_vlm_sop_check/  # CLI、区間検出、評価、VLM推論、Web UI
├── datasets/                 # 動画unit、SOP、人手アノテーション
├── runs/                     # モデル予測
├── evaluations/              # 評価結果
├── reports/                  # 比較レポート
├── schemas/benchmark/v1/     # JSON Schema
├── tools/                    # 移行、検証、公開品質チェック
├── tests/                    # unit / integration tests
└── docs/                     # 設計、運用、意思決定
```

各フォルダの責務は[リポジトリ構造](docs/development/repository-layout.md)、データ契約は[ベンチマーク全体像](docs/benchmark/README.md)にまとめています。

## 開発時の確認

```bash
python3 -m pip install -e ".[test]"
pytest
python3 tools/benchmark/validate.py
python3 tools/quality/check_docs.py
python3 tools/quality/check_public.py
```

Factory Egoのgated mediaをローカルに配置済みの場合は、`python3 tools/benchmark/validate.py --require-media` でハッシュまで検証できます。公開前の確認項目は[公開前チェックリスト](docs/development/public-release.md)を参照してください。

## 現在の制約

- 参照VLMバックエンドはApple Silicon向けのMLXです
- 既定の動画サンプリングは1 fpsです
- Factory Egoは人手GT作成前であり、正式な精度比較には使えません
- ベンチマーク結果は小規模なデモであり、導入判断には対象現場の動画で再評価が必要です

## ライセンス

コードはMIT Licenseです。詳細は[LICENSE](LICENSE)を参照してください。外部データセットとモデルには、それぞれの提供元のライセンスと利用条件が適用されます。
