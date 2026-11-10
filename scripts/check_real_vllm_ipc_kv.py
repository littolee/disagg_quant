"""Validate CUDA IPC mapping and copying of a real vLLM P cache."""
import gc
import pickle
import requests
import torch
import zmq

payload = {
    "model": "pdshared",
    "prompt": "Think carefully about this short question. The capital city of France has a famous tower and a river running through the city. What is the capital of France?",
    "max_tokens": 1,
    "temperature": 0,
    "kv_transfer_params": {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    },
}
response = requests.post("http://127.0.0.1:18140/v1/completions", json=payload, timeout=120)
response.raise_for_status()
result = response.json()
kv = result["kv_transfer_params"]
blocks = kv["remote_block_ids"][0]
assert blocks, kv

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, 30000)
socket.setsockopt(zmq.LINGER, 0)
socket.connect("tcp://127.0.0.1:19041")
socket.send(b"GET")
handles = pickle.loads(socket.recv())
layers = {name: rebuild(*args) for name, (rebuild, args) in handles.items()}
assert len(layers) == 28

index = torch.tensor(blocks, device="cuda", dtype=torch.long)
total_nonzero = 0
for name, source in layers.items():
    selected = source.index_select(0, index)
    target = torch.empty_like(selected)
    target.copy_(selected)
    torch.cuda.synchronize()
    if not torch.equal(target, selected):
        raise AssertionError(name)
    total_nonzero += torch.count_nonzero(target).item()
assert total_nonzero > 0, "KV was unexpectedly empty"
print("PASS real vLLM IPC mapped layers", len(layers), "blocks", len(blocks),
      "KV nonzero", total_nonzero, flush=True)
socket.close()
context.term()
del source, target, selected, index, layers, handles
gc.collect()
torch.cuda.ipc_collect()