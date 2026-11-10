"""Exercise noncontiguous cross-process CUDA IPC copies and error paths."""
import gc
import multiprocessing as mp
import torch
from ipc_kv_copy import copy_kv_blocks

def producer(queue, control):
    torch.manual_seed(101)
    layers = [torch.randn((64, 2, 8, 16, 128), device="cuda",
                          dtype=torch.bfloat16) for _ in range(28)]
    torch.cuda.synchronize()
    queue.put(layers)
    assert control.recv() == "released"

def main():
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    control, peer = ctx.Pipe()
    child = ctx.Process(target=producer, args=(queue, peer))
    child.start()
    source = queue.get(timeout=120)
    target = [torch.zeros_like(x) for x in source]
    stream = torch.cuda.Stream()
    for count in (4, 32):
        src_ids = list(range(0, count * 2, 2))
        dst_ids = list(reversed(range(count)))
        ticket = copy_kv_blocks(source, target, src_ids, dst_ids, stream)
        ticket.wait()
        assert ticket.done()
        for src, dst in zip(source, target):
            a = src.index_select(0, torch.tensor(src_ids, device="cuda"))
            b = dst.index_select(0, torch.tensor(dst_ids, device="cuda"))
            assert torch.equal(a, b)
    for src_ids, dst_ids, expected in [
        ([], [], ValueError),
        ([0, 1], [1, 1], ValueError),
        ([64], [1], IndexError),
        ([1], [64], IndexError),
    ]:
        try:
            copy_kv_blocks(source, target, src_ids, dst_ids, stream)
        except expected:
            pass
        else:
            raise AssertionError(f"missing {expected.__name__}")
    torch.cuda.synchronize()
    del ticket, source, target, src, dst, a, b
    gc.collect()
    torch.cuda.ipc_collect()
    control.send("released")
    child.join(timeout=30)
    assert child.exitcode == 0, child.exitcode
    print("PASS 28 layers, noncontiguous blocks, 1/8 requests, errors, release")

if __name__ == "__main__":
    main()