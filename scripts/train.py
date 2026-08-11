"""QLoRA 学習: Vision Encoder 凍結 / Projector(merger) フル学習 / LLM LoRA。

対象タスク: 窓ベースのイベント区間リスト出力 (data/goalstep/sft_window)。
mlx-vlm 0.6.3 / transformers の既知バグ回避を5件同梱(各関数の docstring と
rtva/imaging.py を参照)。

検証のみ(学習しない): python scripts/train.py --dry-run
メモリプローブ:       python scripts/train.py --max-iters 8 --output-path <scratch>
本実行:               caffeinate -is .venv-vlm/bin/python -u scripts/train.py \\
                          2>&1 | tee training_runs/<run>/train.log
"""

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from datasets import load_dataset
from mlx_vlm.lora import transform_dataset_to_messages
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.sft_trainer import TrainingArgs, train, vision_language_loss_fn
from mlx_vlm.trainer.utils import (
    find_all_linear_names,
    get_peft_model,
    print_trainable_parameters,
    unfreeze_modules,
)
from mlx_vlm.utils import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtva.imaging import set_max_image_pixels


def force_differentiable_gated_delta() -> None:
    """回避1: Gated DeltaNet の Metal カーネルは VJP 未実装。

    メイン経路は use_kernel=not self.training で切り替わるが、状態更新経路
    (language.py の gated_delta_state_update 呼び出し)は use_kernel=True 固定で
    backward が落ちる。全 gated_delta 関数を微分可能な純MLX実装に固定する。
    """
    import mlx_vlm.models.qwen3_5.gated_delta as gd
    import mlx_vlm.models.qwen3_5.language as lang

    for name in ("gated_delta_update", "gated_delta_update_with_states",
                 "gated_delta_state_update", "gated_delta_accept_states"):
        orig = getattr(gd, name)

        def patched(*a, __orig=orig, **k):
            k["use_kernel"] = False
            return __orig(*a, **k)

        setattr(gd, name, patched)
        if hasattr(lang, name):  # language.py は from-import で名前を束縛している
            setattr(lang, name, patched)


def disable_fused_rope(model) -> None:
    """回避2: fused M-RoPE カーネル(mrope_apply_*)も VJP 未実装。非fused実装へ。"""
    n = 0
    for _, m in model.named_modules():
        if hasattr(m, "fused_apply"):
            m.fused_apply = False
            n += 1
    print(f"fused rope 無効化: {n}モジュール")


_loss_calls = 0


def loss_fn_multiimage_fix(model, batch, **kw):
    """回避3: トレーナがマルチ画像の image_grid_thw を (batch, n_img, 3) に stack するが、
    vision 側は (n_img, 3) を期待して unpack に失敗する。(-1, 3) に戻す。

    あわせて回避4: 長時間走行でMLXバッファキャッシュが蓄積しMetal OOMになるため
    定期的に clear_cache する(実測: cache制御なしでは iter 20台で Insufficient Memory)。
    """
    global _loss_calls
    _loss_calls += 1
    if _loss_calls % 100 == 0:
        mx.clear_cache()
    g = batch.get("image_grid_thw")
    if g is not None and g.ndim == 3:
        batch["image_grid_thw"] = g.reshape(-1, 3)
    pv = batch.get("pixel_values")
    if pv is not None and pv.ndim == 3:
        batch["pixel_values"] = pv.reshape(-1, pv.shape[-1])
    return vision_language_loss_fn(model, batch, **kw)


def apply_vision_lora(model, rank: float, alpha: float, dropout: float) -> list[str]:
    """増強ラダー②: Vision Encoder の全線形層(qkv/proj/fc1/fc2×12ブロック)に LoRA。

    LLM側と同一の rank/α を使い、config.lora の keys にフルパスで追記する
    (評価時の load(adapter_path) が同じ構成を自動再現できるようにするため)。
    merger は対象外(フル学習側)。
    """
    import mlx.nn as nn
    from mlx_vlm.trainer.utils import _to_lora, set_module_by_name

    params = {"rank": rank, "dropout": dropout, "scale": alpha / rank}
    keys = []
    for name, module in model.vision_tower.named_modules():
        if "merger" in name or not isinstance(module, (nn.Linear, nn.QuantizedLinear)):
            continue
        if name.split(".")[-1] in ("qkv", "proj", "linear_fc1", "linear_fc2"):
            set_module_by_name(model.vision_tower, name, _to_lora(module, params))
            keys.append(f"vision_tower.{name}")
    model.config.lora["lora_parameters"]["keys"].extend(keys)
    return keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="mlx-community/Qwen3.5-0.8B-MLX-4bit")
    ap.add_argument("--dataset", default="data/goalstep/sft_window")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-seq-length", type=int, default=3072)
    ap.add_argument("--max-image-pixels", type=int, default=448 * 448,
                    help="フレームあたり総ピクセル上限(=画像トークン数の制御)。"
                         "評価側の --max-pixels と必ず一致させる")
    ap.add_argument("--assistant-id", type=int, default=74455,
                    help="Qwen3.5 の 'assistant' トークンID(実測)。損失マスクに使用")
    ap.add_argument("--steps-per-report", type=int, default=10)
    ap.add_argument("--steps-per-save", type=int, default=100)
    ap.add_argument("--output-path", default="training_runs/window-v1/adapters.safetensors")
    ap.add_argument("--cache-limit-gb", type=float, default=1.0,
                    help="MLXバッファキャッシュ上限。無制限だと蓄積でMetal OOM")
    ap.add_argument("--vision-lora", action="store_true",
                    help="増強ラダー②: Vision Encoder にも LoRA (rank/αはLLM側と共通)")
    ap.add_argument("--max-iters", type=int, default=None,
                    help="イテレーション数の上書き。少数指定でメモリプローブに使う")
    ap.add_argument("--dry-run", action="store_true", help="設定検証のみ。学習しない")
    args = ap.parse_args()

    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))
    force_differentiable_gated_delta()
    model, processor = load(args.model_path, processor_config={"trust_remote_code": True})
    disable_fused_rope(model)
    if hasattr(processor, "image_processor") and args.max_image_pixels:
        set_max_image_pixels(processor, args.max_image_pixels)  # 回避5(rtva/imaging.py)
    config = model.config.__dict__

    dataset = load_dataset(args.dataset, split="train")
    dataset = transform_dataset_to_messages(dataset, config.get("model_type"))
    train_dataset = VisionDataset(dataset, config, processor)

    # LLM に LoRA (vision/projector は同時に全凍結される)
    modules = find_all_linear_names(model.language_model)
    model = get_peft_model(
        model, modules,
        rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout,
        verbose=False,
    )
    # Projector (vision_tower.merger) だけフル学習へ解凍。非量子化fp16を確認済み
    unfreeze_modules(model, ["merger"])
    if args.vision_lora:
        vkeys = apply_vision_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
        print(f"vision LoRA: {len(vkeys)}層 (qkv/proj/fc1/fc2 × 12ブロック)")
    print_trainable_parameters(model)

    # 凍結/解凍状態の検証
    from mlx.utils import tree_flatten
    trainable = {k for k, _ in tree_flatten(model.trainable_parameters())}
    n_merger = sum(1 for k in trainable if "merger" in k)
    n_lora = sum(1 for k in trainable if "lora" in k.lower())
    n_enc_lora = sum(1 for k in trainable if "vision_tower" in k and "merger" not in k and "lora" in k.lower())
    n_enc_full = sum(1 for k in trainable if "vision_tower" in k and "merger" not in k and "lora" not in k.lower())
    print(f"trainable keys: merger={n_merger} lora={n_lora} "
          f"vision_lora={n_enc_lora} vision_full={n_enc_full}(0であるべき)")
    assert n_merger > 0 and n_lora > 0 and n_enc_full == 0, "学習対象の凍結状態が不正"
    if args.vision_lora:
        assert n_enc_lora > 0, "--vision-lora 指定なのに vision側LoRAが0"

    iters = args.max_iters or (len(dataset) // args.batch_size) * args.epochs
    print(f"examples={len(dataset)} epochs={args.epochs} -> iters={iters}")
    if args.dry_run:
        print("dry-run: 検証OK。学習は実行しない")
        return

    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=iters,
        steps_per_report=args.steps_per_report,
        steps_per_save=args.steps_per_save,
        max_seq_length=args.max_seq_length,
        adapter_file=args.output_path,
        grad_checkpoint=True,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    train(
        model=model,
        optimizer=optim.Adam(learning_rate=args.learning_rate),
        train_dataset=train_dataset,
        val_dataset=None,
        args=training_args,
        loss_fn=loss_fn_multiimage_fix,
        train_on_completions=True,
        assistant_id=args.assistant_id,
    )
    print(f"peak memory: {mx.get_peak_memory() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
