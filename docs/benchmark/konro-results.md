# Konroベンチマーク結果

ガスコンロの始業前点検を撮影した同一の16フレームに対し、4B以下のローカルVLM 13モデルの回答を人手アノテーションで評価しました。

VLMの回答は、人手アノテーションに対するイベント区間のmean tIoUと、質問ごとのフレーム一致率で評価します。指標の扱いは[評価ポリシー](evaluation.md)を参照してください。

> この結果は、同一の短いデモ動画に対する比較です。一般的な製造現場での性能を示すものではありません。

## 回答の評価

| モデル | mean tIoU | 総合 | 点火<br>`knob` | 炎<br>`flame` | 指差し<br>`pointing` | グリル<br>`grill` | 電池<br>`battery` | 手袋<br>`gloves` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-4B | 0.80 | 96% | 94% | 100% | 81% | 100% | 100% | 100% |
| Qwen2.5-VL-3B | 0.62 | 83% | 50% | 100% | 75% | 94% | 81% | 100% |
| SmolVLM2-2.2B† | 0.60 | 69% | 62% | 100% | 88% | 75% | 88% | 0% |
| Gemma4-E2B | 0.56 | 82% | 88% | 100% | 31% | 88% | 88% | 100% |
| MiniCPM-V 4.6 | 0.55 | 83% | 44% | 100% | 69% | 88% | 100% | 100% |
| Qwen3.5-4B | 0.53 | 77% | 38% | 100% | 75% | 50% | 100% | 100% |
| LFM2.5-VL-1.6B | 0.32 | 50% | 31% | 100% | 44% | 12% | 12% | 100% |
| InternVL3-2B | 0.32 | 50% | 31% | 100% | 25% | 12% | 31% | 100% |
| Qwen3.5-2B | 0.31 | 47% | 25% | 100% | 31% | 12% | 12% | 100% |
| Qwen3.5-0.8B | 0.17 | 67% | 62% | 94% | 31% | 38% | 75% | 100% |
| SmolVLM2-256M† | 0.14 | 17% | 44% | 6% | 19% | 12% | 19% | 0% |
| Qwen3-VL-2B | 0.05 | 77% | 75% | 94% | 88% | 44% | 62% | 100% |
| SmolVLM2-500M† | — | 71% | 75% | 94% | 81% | 88% | 88% | 0% |

mean tIoUが最も高いのはQwen3-VL-4B（0.80）でした。指差しには3フレームの過検出がありますが、各イベント区間の重なりは保たれています。一方、フレーム一致率が77%でもmean tIoUが0.05のモデル（Qwen3-VL-2B）があり、フレーム一致率だけでは区間検出の性能を選べません。起きていない動作（手袋）に yes と答える過検出も誤検出として表面化します。

## 再現方法

```bash
for m in qwen3-4b qwen3-2b qwen3.5-4b qwen3.5-2b qwen3.5-0.8b lfm2.5-1.6b \
         gemma4-e2b qwen2.5-3b minicpm-4.6 internvl3-2b; do
  sop-check observe \
    --sop datasets/konro_inspection/sops/konro_inspection/konro_inspection.yaml \
    --frames-dir datasets/konro_inspection/units/konro_inspection/frames \
    --model "$m" --out "out/al_$m.json"

  sop-check eval \
    --sop datasets/konro_inspection/sops/konro_inspection/konro_inspection.yaml \
    --ground-truth datasets/konro_inspection/annotations/human/konro_inspection.json \
    --answer-log "out/al_$m.json"
done
```

SmolVLM2は公式transformers実装で回答を収集します。`torch`、`torchvision`、`num2words` が別途必要です。

```bash
for m in HuggingFaceTB/SmolVLM2-256M-Video-Instruct \
         HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
         HuggingFaceTB/SmolVLM2-2.2B-Instruct; do
  sop-check observe --backend transformers \
    --sop datasets/konro_inspection/sops/konro_inspection/konro_inspection.yaml \
    --frames-dir datasets/konro_inspection/units/konro_inspection/frames \
    --model "$m" --out "out/al_$(basename "$m").json"
done
```

人手アノテーションは [`datasets/konro_inspection/annotations/human/konro_inspection.json`](../../datasets/konro_inspection/annotations/human/konro_inspection.json) です。

† SmolVLM2の3モデルは `--backend transformers` で計測しています。`gloves` の値が空のJSONになったため、その列は無回答として0%になっています。SmolVLM2-500Mはイベント区間を1つも検出せず、mean tIoUの母数（GTと検出の両方があるイベント）が無いため「—」です。
