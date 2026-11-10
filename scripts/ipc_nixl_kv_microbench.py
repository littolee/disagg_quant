import multiprocessing as mp
import gc,json,os,statistics,time,traceback
from pathlib import Path
import torch

BLOCKS=28
BLOCK_BYTES=256*1024
MAX_CONCURRENCY=8
WARM=5
ITERATIONS=40

def source_process(queue,connection):
    try:
        from nixl._api import nixl_agent,nixl_agent_config
        source=torch.randint(0,256,(MAX_CONCURRENCY,BLOCKS,BLOCK_BYTES),device="cuda",dtype=torch.uint8)
        torch.cuda.synchronize()
        queue.put(source)
        if connection.recv()!="ipc_opened":raise RuntimeError("IPC sharing failed")
        agent=nixl_agent("kv_source",nixl_agent_config(backends=["UCX"]))
        registration=agent.register_memory(source)
        if not registration:raise RuntimeError("source registration failed")
        descs=agent.get_xfer_descs([source[i,j] for i in range(MAX_CONCURRENCY) for j in range(BLOCKS)])
        connection.send((agent.get_agent_metadata(),agent.get_serialized_descs(descs)))
        agent.add_remote_agent(connection.recv())
        if connection.recv()!="done":raise RuntimeError("bad shutdown")
    except Exception:
        connection.send(("error",traceback.format_exc()))
        raise

def main():
    from nixl._api import nixl_agent,nixl_agent_config
    ctx=mp.get_context("spawn")
    queue=ctx.Queue()
    parent,child_pipe=ctx.Pipe()
    child=ctx.Process(target=source_process,args=(queue,child_pipe))
    child.start()
    src=queue.get(timeout=90)
    parent.send("ipc_opened")
    data=parent.recv()
    if data[0]=="error":raise RuntimeError(data[1])
    remote_metadata,remote_serialized=data
    dst=torch.empty_like(src)
    torch.cuda.synchronize()
    agent=nixl_agent("kv_destination",nixl_agent_config(backends=["UCX"]))
    registration=agent.register_memory(dst)
    if not registration:raise RuntimeError("destination registration failed")
    remote_name=agent.add_remote_agent(remote_metadata)
    parent.send(agent.get_agent_metadata())
    local_descs=agent.get_xfer_descs([dst[i,j] for i in range(MAX_CONCURRENCY) for j in range(BLOCKS)])
    remote_descs=agent.deserialize_descs(remote_serialized)
    local_handle=agent.prep_xfer_dlist("",local_descs)
    remote_handle=agent.prep_xfer_dlist(remote_name,remote_descs)
    results=[]
    for concurrency in (1,8):
        handles=[]
        for i in range(concurrency):
            idx=list(range(i*BLOCKS,(i+1)*BLOCKS))
            handles.append(agent.make_prepped_xfer("READ",local_handle,idx,remote_handle,idx,b""))
        for mode in ("nixl","ipc"):
            durations=[]
            for rep in range(WARM+ITERATIONS):
                start=time.perf_counter()
                if mode=="nixl":
                    pending=list(handles)
                    for handle in pending:
                        status=agent.transfer(handle)
                        if status=="ERR":raise RuntimeError("NIXL transfer failure")
                    while pending:
                        pending=[h for h in pending if agent.check_xfer_state(h)!="DONE"]
                        if time.perf_counter()-start>60:raise TimeoutError("NIXL transfer timeout")
                else:
                    for i in range(concurrency):
                        for j in range(BLOCKS):
                            dst[i,j].copy_(src[i,j],non_blocking=True)
                    torch.cuda.synchronize()
                duration=(time.perf_counter()-start)*1000
                if rep>=WARM:durations.append(duration)
            torch.cuda.synchronize()
            valid=bool(torch.equal(dst[:concurrency],src[:concurrency]))
            item={"mode":mode,"concurrency":concurrency,"blocks_per_request":BLOCKS,"bytes_per_block":BLOCK_BYTES,"iterations":ITERATIONS,"batch_ms_p50":statistics.median(durations),"batch_ms_p95":sorted(durations)[int(.95*len(durations))-1],"valid":valid}
            print(json.dumps(item),flush=True)
            if not valid:raise RuntimeError("data mismatch")
            results.append(item)
        for h in handles:agent.release_xfer_handle(h)
    agent.release_dlist_handle(local_handle)
    agent.release_dlist_handle(remote_handle)
    del src
    gc.collect()
    torch.cuda.ipc_collect()
    parent.send("done")
    child.join(timeout=30)
    if child.exitcode!=0:raise RuntimeError(f"source exit {child.exitcode}")
    Path("/home/litto/work/disagg_quant/results/ipc-nixl-kv-microbench.json").write_text(json.dumps(results,indent=2))
if __name__=="__main__":main()