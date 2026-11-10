"""Compare NIXL and CUDA IPC on noncontiguous paged KV blocks."""
import gc
import json
import multiprocessing as mp
import statistics
import time
import traceback
from pathlib import Path

import torch

LAYERS, POOL_BLOCKS, BLOCK_BYTES = 28, 64, 65536
BLOCKS_PER_REQUEST = 4
WARM, REPEATS = 3, 12


def producer(queue, control):
    from nixl._api import nixl_agent, nixl_agent_config
    try:
        source = torch.randint(0, 256, (LAYERS, POOL_BLOCKS, BLOCK_BYTES),
                               dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        queue.put(source)
        if control.recv() != "mapped":
            raise RuntimeError("IPC mapping failed")
        agent = nixl_agent("paged_source", nixl_agent_config(backends=["UCX"]))
        registration = agent.register_memory(source)
        if not registration:
            raise RuntimeError("source registration failed")
        descriptors = agent.get_xfer_descs(
            [source[layer, block] for layer in range(LAYERS)
             for block in range(POOL_BLOCKS)]
        )
        control.send((agent.get_agent_metadata(), agent.get_serialized_descs(descriptors)))
        agent.add_remote_agent(control.recv())
        if control.recv() != "released":
            raise RuntimeError("consumer did not release shared memory")
    except Exception:
        control.send(("error", traceback.format_exc()))
        raise


def main():
    from nixl._api import nixl_agent, nixl_agent_config
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    control, child_control = ctx.Pipe()
    child = ctx.Process(target=producer, args=(queue, child_control))
    child.start()
    source = queue.get(timeout=90)
    control.send("mapped")
    metadata, remote_serialized = control.recv()
    if metadata == "error":
        raise RuntimeError(remote_serialized)
    target = torch.empty_like(source)
    agent = nixl_agent("paged_target", nixl_agent_config(backends=["UCX"]))
    registration = agent.register_memory(target)
    if not registration:
        raise RuntimeError("destination registration failed")
    peer = agent.add_remote_agent(metadata)
    control.send(agent.get_agent_metadata())
    descriptors = agent.get_xfer_descs(
        [target[layer, block] for layer in range(LAYERS)
         for block in range(POOL_BLOCKS)]
    )
    local = agent.prep_xfer_dlist("", descriptors)
    remote = agent.prep_xfer_dlist(peer, agent.deserialize_descs(remote_serialized))
    results = []
    for concurrency in (1, 8):
        count = concurrency * BLOCKS_PER_REQUEST
        source_ids = list(range(0, count * 2, 2))
        target_ids = list(reversed(range(count)))
        handles = []
        for request in range(concurrency):
            start = request * BLOCKS_PER_REQUEST
            stop = start + BLOCKS_PER_REQUEST
            src_indices = [layer * POOL_BLOCKS + source_ids[i]
                           for layer in range(LAYERS) for i in range(start, stop)]
            dst_indices = [layer * POOL_BLOCKS + target_ids[i]
                           for layer in range(LAYERS) for i in range(start, stop)]
            handles.append(agent.make_prepped_xfer("READ", local, dst_indices,
                                                  remote, src_indices, b""))
        for method in ("nixl", "ipc"):
            durations = []
            for rep in range(WARM + REPEATS):
                begin = time.perf_counter()
                if method == "nixl":
                    for handle in handles:
                        if agent.transfer(handle) == "ERR":
                            raise RuntimeError("NIXL transfer failed")
                    pending = list(handles)
                    while pending:
                        remaining = []
                        for handle in pending:
                            status = agent.check_xfer_state(handle)
                            if status == "ERR":
                                raise RuntimeError("NIXL transfer failed")
                            if status != "DONE":
                                remaining.append(handle)
                        pending = remaining
                        if time.perf_counter() - begin > 60:
                            raise TimeoutError("NIXL transfer timeout")
                else:
                    src_index = torch.tensor(source_ids, device="cuda")
                    dst_index = torch.tensor(target_ids, device="cuda")
                    for layer in range(LAYERS):
                        target[layer].index_copy_(0, dst_index,
                                                  source[layer].index_select(0, src_index))
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - begin) * 1000
                if rep >= WARM:
                    durations.append(elapsed)
            torch.cuda.synchronize()
            src_index = torch.tensor(source_ids, device="cuda")
            dst_index = torch.tensor(target_ids, device="cuda")
            valid = all(torch.equal(source[layer].index_select(0, src_index),
                                    target[layer].index_select(0, dst_index))
                        for layer in range(LAYERS))
            result = {"method":method,"concurrency":concurrency,"layers":LAYERS,
                      "blocks_per_request":BLOCKS_PER_REQUEST,"bytes_per_block":BLOCK_BYTES,
                      "descriptors_per_request":LAYERS*BLOCKS_PER_REQUEST,
                      "p50_ms":statistics.median(durations),
                      "p95_ms":sorted(durations)[int(REPEATS*.95)-1],"valid":valid}
            print(json.dumps(result),flush=True)
            if not valid:
                raise AssertionError("KV data mismatch")
            results.append(result)
        for handle in handles:
            agent.release_xfer_handle(handle)
    agent.release_dlist_handle(local)
    agent.release_dlist_handle(remote)
    del source
    gc.collect()
    torch.cuda.ipc_collect()
    control.send("released")
    child.join(timeout=30)
    if child.exitcode != 0:
        raise RuntimeError(f"producer exit {child.exitcode}")
    Path("/home/litto/work/disagg_quant/results/ipc-nixl-paged-blocks.json").write_text(
        json.dumps(results, indent=2))


if __name__ == "__main__":
    main()