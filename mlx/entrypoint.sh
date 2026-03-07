#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MLX_MODEL_PATH:-mlx-community/Qwen2.5-Coder-7B-Instruct-4bit}"
PORT="${MLX_PORT:-8081}"

python -m mlx_lm.server   --model "$MODEL_PATH"   --host 0.0.0.0   --port "$PORT"
