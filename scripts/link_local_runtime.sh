#!/usr/bin/env bash
# Attach the existing Spark experiment artifacts without copying or modifying them.
set -euo pipefail
repo=/home/litto/work/disagg_quant
legacy=${DISAGG_LEGACY_ROOT:-/home/litto/work/disagg_quan}
cd "$repo"
for name in .venv .venv-eval data deps cache checkpoints checkpoints-memory checkpoints-fast disaggregated-quantization-main; do
  if [[ -e "$name" || -L "$name" ]]; then
    printf 'Keeping existing %s\n' "$name"
  elif [[ -e "$legacy/$name" ]]; then
    ln -s "$legacy/$name" "$name"
    printf 'Linked %s\n' "$name"
  else
    printf 'Missing optional runtime path: %s\n' "$legacy/$name" >&2
  fi
done
mkdir -p logs results
