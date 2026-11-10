#!/usr/bin/env bash
set -euo pipefail
cd /home/litto/work/disagg_quant
export CPATH=/home/litto/work/disagg_quant/deps/python-dev/usr/include/python3.12:/home/litto/work/disagg_quant/deps/python-dev/usr/include
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR=/home/litto/work/disagg_quant/cache/inductor
python3 scripts/make_memory_train.py
python3 -m py_compile scripts/train_memory.py scripts/qad_memory_generated.py scripts/cache_weight_views.py
for format in nvfp4 nvfp4pdshared nvfp4a16; do
  echo "Starting $format $(date -Is)"
  .venv/bin/torchrun --standalone --nproc-per-node=1 scripts/train_memory.py \
   --model /home/litto/work/models/Qwen/Qwen3-0.6B --quantizer "$format" \
   --train-tokens 100000000 --val-tokens 100000 --max-seq-len 2048 \
   --micro-batch-size 1 --global-batch-size 64 --lr 3e-6 --warmup-steps 100 \
   --weight-decay 0.1 --grad-clip 1.0 --run-name spark-qwen06-memory \
   --ckpt-dir checkpoints-memory --chunk-cache-dir cache/chunks \
   --save-every 50 --val-every 250 --export-every 250 --resume auto > "logs/train-memory-$format.log" 2>&1
  echo "Finished $format $(date -Is)"
done
