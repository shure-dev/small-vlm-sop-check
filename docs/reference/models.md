# モデルと生成オプション

`sop-check observe` と `sop-check run` の `--model` には、エイリアスまたはHugging Face / mlx-communityの完全なモデルIDを渡せます。既定値は `qwen3-4b` です。

## 登録済みエイリアス

| エイリアス | モデル | 注意点 |
|---|---|---|
| `qwen3-2b` / `qwen3-4b` | Qwen3-VL 2B / 4B | 2BはJSON出力が崩れやすい |
| `qwen3.5-0.8b` / `qwen3.5-2b` / `qwen3.5-4b` | Qwen3.5 0.8B / 2B / 4B | 早期fusionのネイティブVLM |
| `lfm2.5-1.6b` | LFM2.5-VL 1.6B | mlx-vlm 0.6.3では変換重みとprojector実装が不一致 |
| `qwen2.5-3b` | Qwen2.5-VL-3B | — |
| `internvl3-2b` | InternVL3-2B | — |
| `gemma4-e2b` | Gemma4-E2B | — |
| `minicpm-4.6` | MiniCPM-V 4.6 | 思考モデル |

利用できるエイリアスは、インストール後に次のコマンドでも確認できます。

```bash
sop-check models
```

## 生成オプション

| オプション | 既定値 | 用途 |
|---|---:|---|
| `--prefill STR` | `{"` | JSONの先頭を固定し、空応答や長い思考を抑える |
| `--max-tokens N` | `200` | 最大生成トークン数を指定する |
| `--thinking {auto,on,off}` | `auto` | 対応モデルの思考モードを指定する |

思考過程を使う場合は `--prefill '' --max-tokens 1024` などに変更します。ただし、このプロジェクトのフレームごとの質問回答では、短いyes/no JSONを安定して返す設定を優先しています。

新規の標準実験は4B以下に限定します。過去に取得済みの7Bモデル結果は比較履歴として残しますが、CLIの推奨aliasからは外しています。必要な任意モデルはフルmodel IDで明示指定できます。

全動画Temporal Groundingの動作可否はフレーム分類と異なります。2026-07-18のmodel matrixでは、MiniCPM-V-4.6は20秒外の時刻を返し、InternVL3-2BとLFM2.5-VL-1.6Bは現行MLX動画経路のruntime互換性ゲートを通過しませんでした。失敗をモデル精度と混同せず、条件と理由は[集約artifact](../../evaluations/factory_ego_model_matrix_v1.json)へ残します。

## SmolVLM2について

Konroベンチマークでは、SmolVLM2の3モデルだけ `--backend transformers` で計測しました。mlx-community変換とmlx-vlmの組み合わせでは視覚入力の問題があり、同じ条件で比較できなかったためです。再現条件は[Konroベンチマーク結果](../benchmark/konro-results.md)に記載しています。
