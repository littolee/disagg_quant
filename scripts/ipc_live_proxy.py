import aiohttp
from fastapi import FastAPI,HTTPException,Request
app=FastAPI()
@app.post("/v1/completions")
async def completions(request:Request):
    data=await request.json()
    data["model"]="pdshared"
    data["stream"]=False
    if isinstance(data.get("prompt"),list) and data["prompt"] and isinstance(data["prompt"][0],list):
        if len(data["prompt"])!=1: raise HTTPException(400,"batch must be 1")
        data["prompt"]=data["prompt"][0]
    p=dict(data)
    p["max_tokens"]=1
    p.pop("max_completion_tokens",None)
    p["kv_transfer_params"]={"do_remote_decode":True,"do_remote_prefill":False,"remote_engine_id":None,"remote_block_ids":None,"remote_host":None,"remote_port":None}
    timeout=aiohttp.ClientTimeout(total=1200)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post("http://127.0.0.1:18140/v1/completions",json=p) as r:
            pre=await r.json()
            if r.status!=200: raise HTTPException(r.status,str(pre))
        kv=pre.get("kv_transfer_params")
        if not kv or not kv.get("do_remote_prefill") or not kv.get("remote_block_ids"): raise HTTPException(502,"P did not return KV transfer parameters")
        data["kv_transfer_params"]=kv
        async with session.post("http://127.0.0.1:18240/v1/completions",json=data) as r:
            out=await r.json()
            if r.status!=200: raise HTTPException(r.status,str(out))
            return out