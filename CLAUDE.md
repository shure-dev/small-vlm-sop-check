# CLAUDE.md

作業動画がSOP（手順書）通りかを判定するCLI package兼実験リポジトリ。ローカルの小型VLM（Qwen3-VL / Apple Silicon / mlx-vlm）がフレームごとの質問に回答し、決定論的ルールが判定する。

## 構成

- `src/small_vlm_sop_check/` — src-layoutのinstallable package
  - `core/`（SOP読込・判定・評価） / `inference/`（frame抽出・VLMの回答収集） / `apps/`（annotator・replay・HTML template） / `cli.py`
- `datasets/konro_inspection/` — モザイク済み実動画、抽出フレーム、SOP 3条件、人手GTを持つ完結したデモデータセット。固定モデルログは`fixtures/reference_outputs/`
- `tests/unit/` — coreロジック、`tests/integration/` — dataset/run/docs契約。VLM不要
- `datasets/factory_ego/` — Egocentric-10K由来の精度比較データ。unit/meta/frame、暫定SOP、splitを保持し、モデル予測は置かない
- `runs/` — Fable・Opus・Qwenを対等に扱う不変のprediction run。raw、正規化予測、入力lockを保持
- `evaluations/` — 人手GT revisionとprediction runを入力にする評価run（Factory Egoは人手GT未作成のため現在は空）
- `reports/` — 比較結果。人手GTがない間は一致率等を「精度」と表記しない
- `tools/benchmark/` — unitの層化サンプリング（`sample_units.py`・決定論的・追記型）、gated媒体の再構成（`fetch_factory_ego.py`・既定dry-run）、整合性検証（`validate.py`）、ローカルモデルのprediction run作成（`run_local_prediction.py`）、referenceとの区間tIoU予備比較（`reference_tiou.py`）
- `tools/quality/` — Markdown link等のrepository品質検査。ユーザー向け機能は置かない
- `schemas/benchmark/v1/` — unit・run・prediction・splitのversioned JSON Schema
- `docs/` — 設計・評価ポリシー・運用・ADRの正本。READMEには概要とクイックスタートだけを置く

## コマンド

```bash
python3 -m pip install -e .              # core CLI・annotator・replay
python3 -m pip install -e ".[vlm,test]"  # VLM推論・testも含む
pytest                            # 17件。VLM・GPUなしで動く
python3 tools/benchmark/validate.py  # Factory Egoのhash・split・run不変条件を検証
python3 tools/benchmark/fetch_factory_ego.py  # gated上流から媒体を再構成(要HF同意。既定dry-run)
python3 tools/quality/check_docs.py   # Markdownのローカルリンクを検証
python3 tools/quality/check_public.py # 公開候補の秘密情報・絶対パス・gated媒体を検査

# VLMなしで動く判定のみの実行（動作確認はまずこれ）
sop-check judge \
  --sop datasets/konro_inspection/sops/konro_inspection/correct.yaml \
  --answer-log datasets/konro_inspection/fixtures/reference_outputs/answer_log.json

# 正解アノテーション（ブラウザ・自動保存）と、それとの突き合わせ評価（どちらもVLM不要）
sop-annotate
sop-check eval \
  --sop datasets/konro_inspection/sops/konro_inspection/correct.yaml \
  --ground-truth datasets/konro_inspection/annotations/human-v001/konro_inspection.json \
  --answer-log datasets/konro_inspection/fixtures/reference_outputs/answer_log.json

# フル実行（mlx-vlm必要・Apple Silicon限定・モデルDLが走る）
sop-check run --sop datasets/konro_inspection/sops/konro_inspection/correct.yaml \
  --video datasets/konro_inspection/units/konro_inspection/media/konro_inspection.mp4 \
  --model 4b --out-dir out/
```

## 設計原則（変更しないこと）

- **回答と判定の分離**: VLMは質問（questions）にフレーム単位で答えるだけ（Phase 1）。順序や遵守の判定は決定論的なルールエンジンが行う（Phase 2）。判定をVLMの自然文推論に委ねない——検証で単純な時刻比較すら間違えることを確認済み。日本語ドキュメントでこの工程を「観察」と呼ばない（わかりにくいため「（フレームごとの質問への）回答」「回答収集」と書く。回答ログ＝`answer_log.json`）。
- **用語**: `questions`（VLMへの質問）/ `answers`・`answer_log.json`（回答）/ `events` / `relations` / `ground_truth.json`（人手の正解区間）。旧称「cue」は廃止済みなので復活させない。
- relationsは `before` / `overlaps` / `not` の3種類のみ。安易に増やさない（Allenの13関係を境界ノイズで壊れない同値類まで潰したのがこの3つ、という整理。READMEのSOPフォーマット節に対応表あり）。
- **アノテーションは事実（いつ何が起きたか＝区間）だけを記録する**。関係や遵守の「べき」を注釈に持ち込まない。回答精度の成功条件は一次が expect（verdict＋理由）の一致で、tIoU・relations正答・フレーム一致は診断用（`src/small_vlm_sop_check/core/evaluate.py` 冒頭のdocstring参照）。境界±数フレームのズレは注釈側でなく tIoU しきい値側で吸収する。
- **Factory Egoのモデル出力をground truthへ昇格しない**。Fable・Opus・Qwenは全て`runs/`のprediction。人手GTができるまではformal accuracyをnullのままにし、評価値はprediction runへ追記せず別evaluation runを作る。
- **splitはfactory/worker単位**。現行unitは選定・アノテーション過程で閲覧されるため`dev_seen`固定でtestへ昇格させない。真のtestは未閲覧クリップ＋人手GTで作る。

## ハマりどころ

- SOP YAMLの `values: ["yes", "no"]` はクォート必須。裸の yes/no はYAML 1.1でブール値になる。
- `occurrence` 未指定のeventはYAML宣言順に早い者勝ちで区間を取るため、宣言順を変えると結果が変わる。時系列N番目に固定したければ `occurrence: N`。
- mlx-vlm実行中に稀にMetal GPU Hangが起きる。回答ログは1フレームごとに逐次保存しているので、再実行すれば途中から再開できる。
- fpsを上げると精度が上がるとは限らない（短いノイズが単独検出として顕在化し、判定が反転した実測あり）。既定の1fpsを基準にする。

## 試せるVLM（実測）

`--model` にエイリアス（`sop-check models` で一覧）かHF/mlx-communityのフルIDを渡す。mlx-vlm がロードでき単一画像で厳密なJSONを返せるモデルが対象。動作確認済み: Qwen3-VL 2B/4B（既定は `qwen3-4b`）・Qwen3.5 0.8B/2B/4B・LFM2.5-VL-1.6B（要mlx-vlm>=0.6.4）・Qwen2.5-VL-3B・InternVL3-2B・Gemma4-E2B・MiniCPM-V 4.6・Molmo-7B・Cosmos-Reason1-7B。

- **torch必須で不可**: SmolVLM・LFM2-VL・FastVLM（mlx-communityのbf16版）（`.venv-vlm` は torch なしで画像プロセッサ生成に失敗）。
- **SmolVLM2（mlx-community変換 256M/500M/2.2B）は torch を足しても実質不可**（2026-07実測）: transformers 5.12.1 は smolvlm系画像プロセッサが PIL 版まで torch+torchvision 必須で、torch なしでは3モデルともロード不可。torch を足すとロード・実行は通るが、mlx-vlm 0.6.3/0.6.4 の経路で視覚入力が潰れ（点火フレームを「白い壁」、青地の大きな赤丸を「Blue background」としか説明できない）、全フレームが同一回答に退化する（256M=全yes、500M/2.2B=全no）。同じ256M重みを公式 transformers(torch/CPU) で動かすと点火フレームを正しく説明したため、モデルではなく mlx 側（変換または mlx-vlm 実装）の問題と切り分け済み。このため SmolVLM2 のベンチ値は `--backend transformers`（公式実装。observe.py の TransformersObserver）で計測した（READMEの†印）。2.2B は relations 4/6・tIoU 0.55 の中堅、256M/500M は視覚が正常でも yes/no 識別に追従できない（256M=ほぼ全yes、500M=空スキーマをエコーしロジット計測では全no）。**新モデル追加時はベンチ前に1フレームを自由記述で説明させ、視覚が生きているか確認する**。
- **JSON形式に追従できず不可**: Qwen2-VL-2B・Gemma-3n-E2B。`mlx-community/Perception-LM-*` は config.json 欠落でロード不可。
- **重み名不一致でロード不可**: `InsightKeeper/FastVLM-*-MLX-4bit`（mlx-vlmのfastvlm実装は `mm_projector.*`、チェックポイントは `multi_modal_projector.linear_*`）。
- **LFM2.5-VL-1.6B は mlx-vlm 0.6.3 でロード不可**（lfm2_vlが `layer_norm` を無条件生成する実装バグ。0.6.4で修正済みだが上記条件付き）。
- **Qwen3-VL-2B は mlx-vlm 0.6.3 の再実測で半数のフレームのJSONが崩壊**（クォート欠落・同一キー繰り返し。一致率18%）。以前の「動作確認済み」から劣化しており要注意。
- **InternVL3.5-30B-A3B は RAM 24GB では非現実的**（4bitでも重み約17GB）。8B級（Qwen3-VL-8B・Qwen3.5-9B・InternVL3-8B）は方針により未計測。

**プロンプトは英語指示＋質問文をlegendに分離**（`small_vlm_sop_check.inference.observe.build_prompt`）。値スロットに質問文を入れると MiniCPM-V 等が値に質問文をエコーして yes/no が出ないため。`--prefill`（既定 `{"`）でアシスタント応答をJSONの最初のキーの途中まで固定する。これで (1) Molmoのように最初のトークンでEOSを出す空応答、(2) MiniCPM-V/Cosmosのように`<think>`でトークンを使い切りJSONに届かない、の両方を既定のまま回避でき、Qwen3-VL-2Bを除く全モデルでクリーンな yes/no JSON が出る（実測）。思考の連鎖を使いたい時だけ `--prefill '' --max-tokens 1024`。

## 検証のしかた

変更したら必ず `pytest` と上記の `judge` コマンドを実行し、総合判定が PASS のままであることを確認する。
