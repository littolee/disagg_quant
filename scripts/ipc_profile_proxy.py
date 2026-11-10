import aiohttp,json,time
from fastapi import FastAPI,HTTPException,Request
app=FastAPI()
@app.post('/v1/completions')
async def completions(request:Request):
    t0=time.monotonic_ns(); data=await request.json(); t1=time.monotonic_ns()
    data['model']='pdshared'; data['stream']=False
    p=dict(data); p['max_tokens']=1; p['kv_transfer_params']={'do_remote_decode':True,'do_remote_prefill':False,'remote_engine_id':None,'remote_block_ids':None,'remote_host':None,'remote_port':None}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        p0=time.monotonic_ns()
        async with session.post('http://127.0.0.1:18140/v1/completions',json=p) as r:
            pre=await r.json(); p1=time.monotonic_ns()
            if r.status!=200: raise HTTPException(r.status,str(pre))
        kv=pre.get('kv_transfer_params')
        if not kv or not kv.get('remote_block_ids'): raise HTTPException(502,'P KV missing')
        data['kv_transfer_params']=kv
        d0=time.monotonic_ns()
        async with session.post('http://127.0.0.1:18260/v1/completions',json=data) as r:
            out=await r.json(); d1=time.monotonic_ns()
            if r.status!=200: raise HTTPException(r.status,str(out))
    record={'id':out['id'],'proxy_start':t0,'parsed':t1,'p_send':p0,'p_done':p1,'d_send':d0,'d_done':d1,'proxy_end':time.monotonic_ns(),'prompt_tokens':out['usage']['prompt_tokens']}
    print('PROFILE '+json.dumps(record),flush=True)
    return out
