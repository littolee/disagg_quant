import multiprocessing as mp
import statistics,time,json
from pathlib import Path
import torch

SIZES=(1835008,7340032,14680064,83886080)
WARM=10
REPEATS=100

def worker(q,conn):
    try:
        source=q.get(timeout=60)
        results=[]
        for size in SIZES:
            src=source[:size]
            dst=torch.empty_like(src)
            for _ in range(WARM):dst.copy_(src)
            torch.cuda.synchronize()
            times=[]
            for _ in range(REPEATS):
                start=torch.cuda.Event(enable_timing=True)
                end=torch.cuda.Event(enable_timing=True)
                start.record()
                dst.copy_(src)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            valid=bool(torch.equal(src,dst))
            results.append({"bytes":size,"copy_gpu_ms_p50":statistics.median(times),"copy_gpu_ms_p95":sorted(times)[94],"copy_valid":valid})
        conn.send(("ready",results))
        for _ in range(30):
            cmd=conn.recv()
            if cmd!="copy":raise ValueError(cmd)
            dst.copy_(src)
            torch.cuda.synchronize()
            conn.send("done")
        conn.send(("final",bool(torch.equal(src,dst))))
    except Exception as e:
        conn.send(("error",repr(e)))

def main():
    ctx=mp.get_context("spawn")
    q=ctx.Queue()
    parent,child_conn=ctx.Pipe()
    source=torch.arange(SIZES[-1],device="cuda",dtype=torch.uint8)
    torch.cuda.synchronize()
    child=ctx.Process(target=worker,args=(q,child_conn))
    start=time.perf_counter()
    child.start()
    q.put(source)
    if not parent.poll(180):raise TimeoutError("IPC setup timeout")
    kind,info=parent.recv()
    if kind!="ready":raise RuntimeError(info)
    setup=time.perf_counter()-start
    rtt=[]
    for _ in range(30):
        start=time.perf_counter()
        parent.send("copy")
        if not parent.poll(30):raise TimeoutError("copy timeout")
        if parent.recv()!="done":raise RuntimeError("copy failed")
        rtt.append(1000*(time.perf_counter()-start))
    finish=parent.recv()
    child.join(timeout=30)
    result={"setup_including_process_spawn_s":setup,"samples":info,"copy_size_for_rtt_bytes":SIZES[-1],"rtt_ms_p50":statistics.median(rtt),"rtt_ms_p95":sorted(rtt)[28],"last_copy_valid":finish,"exit_code":child.exitcode}
    path=Path("/home/litto/work/disagg_quant/results/cuda-ipc-copy-microbench.json")
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if any(not x["copy_valid"] for x in info) or finish!=("final",True) or child.exitcode!=0:raise RuntimeError("validation failed")
if __name__=="__main__":main()