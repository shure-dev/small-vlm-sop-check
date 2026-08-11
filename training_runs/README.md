# training_runs/ — Git管理外

`scripts/train.py` の成果物（LoRAアダプタ `adapters.safetensors`・途中チェックポイント・
`train.log`）を run ごとのサブディレクトリに置く。`README.md` 以外はGitに入れない。

```bash
caffeinate -is .venv-vlm/bin/python -u scripts/train.py ... \
    --output-path training_runs/<run名>/adapters.safetensors \
    2>&1 | tee training_runs/<run名>/train.log
```
