# CLAUDE.md

作業動画がSOP（手順書）通りかを、ローカルの小型VLM（Qwen3-VL / Apple Silicon / mlx-vlm）だけで判定するデモ。ライブラリではなく実験コードの公開リポジトリ。

## 構成

- `src/` — モジュール直置き（パッケージ化していない。pip install不可・する予定もない）
  - `cli.py`（`run`/`observe`/`judge`） / `observe.py`（Phase 1: VLM観察） / `judge.py`（Phase 2: ルールエンジン） / `extract.py`（動画→フレーム） / `sop.py`（SOP YAML読み込み）
- `examples/konro_inspection/` — モザイク済み実動画・抽出フレーム・回答ログ・SOP YAML 3種（正解 / 順序違反 / ステップ欠落）
- `tools/replay_viewer/` — 結果をフレーム画像ごと1枚のHTMLにして再生するビューア。`python tools/replay_viewer/build.py` で再生成（同梱の `replay.html` は生成物）
- `tests/` — 実データに対する回帰テスト。VLM不要

## コマンド

```bash
pip install -r requirements.txt   # judgeだけなら pyyaml のみでよい
pytest                            # 4件。VLM・GPUなしで動く（src/へのパスはテスト内で追加済み）

# VLMなしで動く判定のみの実行（動作確認はまずこれ）
python src/cli.py judge \
  --sop examples/konro_inspection/sop.yaml \
  --answer-log examples/konro_inspection/sample_output/answer_log.json

# フル実行（mlx-vlm必要・Apple Silicon限定・モデルDLが走る）
python src/cli.py run --sop examples/konro_inspection/sop.yaml \
  --video examples/konro_inspection/data/konro_inspection.mp4 --model 4b --out-dir out/
```

## 設計原則（変更しないこと）

- **観察と判定の分離**: VLMは質問（questions）にフレーム単位で答えるだけ（Phase 1）。順序や遵守の判定は決定論的なルールエンジンが行う（Phase 2）。判定をVLMの自然文推論に委ねない——検証で単純な時刻比較すら間違えることを確認済み。
- **用語**: `questions`（VLMへの質問）/ `answers`・`answer_log.json`（回答）/ `events` / `relations`。旧称「cue」は廃止済みなので復活させない。
- relationsは `before` / `overlaps` / `not` の3種類のみ。安易に増やさない。

## ハマりどころ

- SOP YAMLの `values: ["yes", "no"]` はクォート必須。裸の yes/no はYAML 1.1でブール値になる。
- `occurrence` 未指定のeventはYAML宣言順に早い者勝ちで区間を取るため、宣言順を変えると結果が変わる。時系列N番目に固定したければ `occurrence: N`。
- mlx-vlm実行中に稀にMetal GPU Hangが起きる。回答ログは1フレームごとに逐次保存しているので、再実行すれば途中から再開できる。
- fpsを上げると精度が上がるとは限らない（短いノイズが単独検出として顕在化し、判定が反転した実測あり）。既定の1fpsを基準にする。

## 試せるVLM（実測）

`--model` にエイリアス（`python src/cli.py models` で一覧）かHF/mlx-communityのフルIDを渡す。mlx-vlm がロードでき単一画像で厳密なJSONを返せるモデルが対象。動作確認済み: Qwen3-VL 2B/4B（基準は `qwen3-4b`）・Qwen2.5-VL-3B・InternVL3-2B・Gemma4-E2B・MiniCPM-V 4.6・Molmo-7B・Cosmos-Reason1-7B。

- **torch必須で不可**: SmolVLM・LFM2-VL（`.venv-vlm` は torch なしで画像プロセッサ生成に失敗）。
- **JSON形式に追従できず不可**: Qwen2-VL-2B・Gemma-3n-E2B。`mlx-community/Perception-LM-*` は config.json 欠落でロード不可。

`--prefill`（既定 `{`）でアシスタント応答をJSONの開き括弧から始めさせる。これで (1) Molmoのように末尾が`}`のプロンプトを見て最初のトークンでEOSを出す空応答、(2) MiniCPM-V/Cosmosのように`<think>`でトークンを使い切りJSONに届かない、の両方を既定のまま回避できる（思考の連鎖を使いたい時だけ `--prefill '' --max-tokens 1024`）。

## 検証のしかた

変更したら必ず `pytest` と上記の `judge` コマンドを実行し、総合判定が PASS のままであることを確認する。
