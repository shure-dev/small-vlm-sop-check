# CLAUDE.md

作業動画がSOP（手順書）通りかを、ローカルの小型VLM（Qwen3-VL / Apple Silicon / mlx-vlm）だけで判定するデモ。ライブラリではなく実験コードの公開リポジトリ。

## 構成

- `src/` — モジュール直置き（パッケージ化していない。pip install不可・する予定もない）
  - `cli.py`（`run`/`observe`/`judge`/`eval`） / `observe.py`（Phase 1: VLM観察） / `judge.py`（Phase 2: ルールエンジン） / `evaluate.py`（正解アノテーションとの突き合わせ評価） / `extract.py`（動画→フレーム） / `sop.py`（SOP YAML読み込み）
- `examples/konro_inspection/` — モザイク済み実動画・抽出フレーム・回答ログ・SOP YAML 3種（正解 / 順序違反 / ステップ欠落）。人手注釈すると `ground_truth.json` もここに入る（コミット対象のデータ）
- `tools/replay_viewer/` — 結果をフレーム画像ごと1枚のHTMLにして再生するビューア。`python tools/replay_viewer/build.py` で生成（`replay.html` はbase64画像を埋め込む生成物のためgit管理外。`frames/` は同梱済み）。SOPと同じディレクトリに `ground_truth.json` があれば正解区間を自動で重ねる
- `tools/annotator/` — 正解区間をブラウザで注釈するツール（標準ライブラリのみ）。`python tools/annotator/serve.py` で起動。操作のたびに `ground_truth.json` へ自動保存・途中再開可
- `tests/` — 実データに対する回帰テスト。VLM不要

## コマンド

```bash
pip install -r requirements.txt   # judgeだけなら pyyaml のみでよい
pytest                            # 12件。VLM・GPUなしで動く（src/へのパスはテスト内で追加済み）

# VLMなしで動く判定のみの実行（動作確認はまずこれ）
python src/cli.py judge \
  --sop examples/konro_inspection/sop.yaml \
  --answer-log examples/konro_inspection/sample_output/answer_log.json

# 正解アノテーション（ブラウザ・自動保存）と、それとの突き合わせ評価（どちらもVLM不要）
python tools/annotator/serve.py
python src/cli.py eval --sop examples/konro_inspection/sop.yaml \
  --answer-log examples/konro_inspection/sample_output/answer_log.json

# フル実行（mlx-vlm必要・Apple Silicon限定・モデルDLが走る）
python src/cli.py run --sop examples/konro_inspection/sop.yaml \
  --video examples/konro_inspection/data/konro_inspection.mp4 --model 4b --out-dir out/
```

## 設計原則（変更しないこと）

- **観察と判定の分離**: VLMは質問（questions）にフレーム単位で答えるだけ（Phase 1）。順序や遵守の判定は決定論的なルールエンジンが行う（Phase 2）。判定をVLMの自然文推論に委ねない——検証で単純な時刻比較すら間違えることを確認済み。
- **用語**: `questions`（VLMへの質問）/ `answers`・`answer_log.json`（回答）/ `events` / `relations` / `ground_truth.json`（人手の正解区間）。旧称「cue」は廃止済みなので復活させない。
- relationsは `before` / `overlaps` / `not` の3種類のみ。安易に増やさない（Allenの13関係を境界ノイズで壊れない同値類まで潰したのがこの3つ、という整理。READMEのSOPフォーマット節に対応表あり）。
- **アノテーションは事実（いつ何が起きたか＝区間）だけを記録する**。関係や遵守の「べき」を注釈に持ち込まない。観察精度の成功条件は一次が expect（verdict＋理由）の一致で、tIoU・relations正答・フレーム一致は診断用（`src/evaluate.py` 冒頭のdocstring参照）。境界±数フレームのズレは注釈側でなく tIoU しきい値側で吸収する。

## ハマりどころ

- SOP YAMLの `values: ["yes", "no"]` はクォート必須。裸の yes/no はYAML 1.1でブール値になる。
- `occurrence` 未指定のeventはYAML宣言順に早い者勝ちで区間を取るため、宣言順を変えると結果が変わる。時系列N番目に固定したければ `occurrence: N`。
- mlx-vlm実行中に稀にMetal GPU Hangが起きる。回答ログは1フレームごとに逐次保存しているので、再実行すれば途中から再開できる。
- fpsを上げると精度が上がるとは限らない（短いノイズが単独検出として顕在化し、判定が反転した実測あり）。既定の1fpsを基準にする。

## 試せるVLM（実測）

`--model` にエイリアス（`python src/cli.py models` で一覧）かHF/mlx-communityのフルIDを渡す。mlx-vlm がロードでき単一画像で厳密なJSONを返せるモデルが対象。動作確認済み: Qwen3-VL 2B/4B（既定は `qwen3-4b`）・Qwen3.5 0.8B/2B/4B・LFM2.5-VL-1.6B（要mlx-vlm>=0.6.4）・Qwen2.5-VL-3B・InternVL3-2B・Gemma4-E2B・MiniCPM-V 4.6・Molmo-7B・Cosmos-Reason1-7B。

- **torch必須で不可**: SmolVLM・LFM2-VL・FastVLM（mlx-communityのbf16版）（`.venv-vlm` は torch なしで画像プロセッサ生成に失敗）。
- **JSON形式に追従できず不可**: Qwen2-VL-2B・Gemma-3n-E2B。`mlx-community/Perception-LM-*` は config.json 欠落でロード不可。
- **重み名不一致でロード不可**: `InsightKeeper/FastVLM-*-MLX-4bit`（mlx-vlmのfastvlm実装は `mm_projector.*`、チェックポイントは `multi_modal_projector.linear_*`）。
- **LFM2.5-VL-1.6B は mlx-vlm 0.6.3 でロード不可**（lfm2_vlが `layer_norm` を無条件生成する実装バグ。0.6.4で修正済みだが上記条件付き）。
- **Qwen3-VL-2B は mlx-vlm 0.6.3 の再実測で半数のフレームのJSONが崩壊**（クォート欠落・同一キー繰り返し。一致率18%）。以前の「動作確認済み」から劣化しており要注意。
- **InternVL3.5-30B-A3B は RAM 24GB では非現実的**（4bitでも重み約17GB）。8B級（Qwen3-VL-8B・Qwen3.5-9B・InternVL3-8B）は方針により未計測。

**プロンプトは英語指示＋質問文をlegendに分離**（`observe.py::build_prompt`）。値スロットに質問文を入れると MiniCPM-V 等が値に質問文をエコーして yes/no が出ないため。`--prefill`（既定 `{"`）でアシスタント応答をJSONの最初のキーの途中まで固定する。これで (1) Molmoのように最初のトークンでEOSを出す空応答、(2) MiniCPM-V/Cosmosのように`<think>`でトークンを使い切りJSONに届かない、の両方を既定のまま回避でき、Qwen3-VL-2Bを除く全モデルでクリーンな yes/no JSON が出る（実測）。思考の連鎖を使いたい時だけ `--prefill '' --max-tokens 1024`。

## 検証のしかた

変更したら必ず `pytest` と上記の `judge` コマンドを実行し、総合判定が PASS のままであることを確認する。
