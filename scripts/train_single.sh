#!/usr/bin/env bash
set -euo pipefail
cd /home/litto/work/disagg_quant
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR=/home/litto/work/disagg_quant/cache/inductor
quantizer=${1:-nvfp4pdshared}
tokens=${2:-100000000}
exec .venv/bin/torchrun --standalone --nproc-per-node=1 scripts/train_single.py \
  --model /home/litto/work/models/Qwen/Qwen3-0.6B \
  --quantizer "$quantizer" --train-tokens "$tokens" --val-tokens 100000 \
  --max-seq-len 2048 --micro-batch-size 1 --global-batch-size 64 \
  --lr 3e-6 --warmup-steps 100 --weight-decay 0.1 --grad-clip 1.0 \
  --run-name spark-qwen06 --ckpt-dir checkpoints --chunk-cache-dir cache/chunks \
  --save-every 50 --val-every 250 --export-every 250 --resume auto
