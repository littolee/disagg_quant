#!/bin/bash
set -euo pipefail
cd /home/litto/work/disagg_quant
export PYTHONPATH="/home/litto/work/disagg_quant/scripts" MAX_JOBS=2 PATH="/home/litto/work/disagg_quant/.venv-eval/bin:$PATH" TOKENIZERS_PARALLELISM=false VLLM_HOST_IP=127.0.0.1 UCX_NET_DEVICES=all
base=checkpoints-memory/spark-qwen06-memory-nvfp4pdshared-2d557910/weights/step_0002484
export VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1
VLLM_NIXL_SIDE_CHANNEL_PORT=5640 nohup .venv-eval/bin/vllm serve "$base/prefill" --tokenizer /home/litto/work/models/Qwen/Qwen3-0.6B --served-model-name pdshared --host 127.0.0.1 --port 18140 --gpu-memory-utilization 0.12 --kv-cache-memory-bytes 2G --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 8192 --kv-transfer-config '{"kv_connector":"InspectNixlConnector","kv_connector_module_path":"ipc_inspect_connector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}' > logs/ipc-live-prefill.log 2>&1 < /dev/null &
for i in $(seq 1 360); do curl -fsS --max-time 2 http://127.0.0.1:18140/health >/dev/null 2>&1 && break; sleep 5; done
curl -fsS --max-time 2 http://127.0.0.1:18140/health >/dev/null || exit 1
VLLM_NIXL_SIDE_CHANNEL_PORT=5661 nohup .venv-eval/bin/vllm serve "$base/decode" --tokenizer /home/litto/work/models/Qwen/Qwen3-0.6B --served-model-name pdshared --host 127.0.0.1 --port 18260 --gpu-memory-utilization 0.12 --kv-cache-memory-bytes 2G --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 8192 --kv-transfer-config '{"kv_connector":"IpcNixlConnector","kv_connector_module_path":"ipc_profile_connector","kv_role":"kv_consumer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"enforce_handshake_compat":false}}' > logs/ipc-profile-decode.log 2>&1 < /dev/null &
for i in $(seq 1 360); do curl -fsS --max-time 2 http://127.0.0.1:18240/health >/dev/null 2>&1 && break; sleep 5; done
curl -fsS --max-time 2 http://127.0.0.1:18240/health >/dev/null || exit 1
echo "BLHNC servers ready"
