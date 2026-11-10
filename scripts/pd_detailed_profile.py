import asyncio,aiohttp,hashlib,json,statistics,time
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
ROOT=Path("/home/litto/work/disagg_quant")
MODEL="/home/litto/work/models/Qwen/Qwen3-0.6B"
tok=AutoTokenizer.from_pretrained(MODEL)
rows=pq.read_table(ROOT/"data/TIGER-Lab__MMLU-Pro/test-00000-of-00001.parquet").slice(0,64).to_pylist()
def prompt(row,i,run_id):
 nonce=hashlib.sha256(f"stageprof:{run_id}:{i}".encode()).hexdigest()
 options="\n".join(f"{chr(65+j)}. {v}" for j,v in enumerate(row["options"]) if v)
 return tok.apply_chat_template([{"role":"user","content":nonce+"\n"+row["question"]+"\n"+options+"\nAnswer with the correct option."}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
def metric(text,name):
 for line in text.splitlines():
  if line.startswith(name+"{"): return float(line.rsplit(" ",1)[1])
 raise RuntimeError("missing metric "+name)
async def metrics(session):
 async with session.get("http://127.0.0.1:18200/metrics") as resp:
  text=await resp.text()
  names=["vllm:nixl_xfer_time_seconds_count","vllm:nixl_xfer_time_seconds_sum","vllm:nixl_num_failed_transfers_total","vllm:external_prefix_cache_hits_total"]
  return {n:metric(text,n) for n in names}
async def main():
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
  async def post(port,payload):
   async with s.post(f"http://127.0.0.1:{port}/v1/completions",json=payload) as r:
    obj=await r.json()
    if r.status!=200:raise RuntimeError(f"{r.status} {obj}")
    return obj
  async def run(concurrency,nout,run_id,count):
   sem=asyncio.Semaphore(concurrency); records=[None]*count
   before=await metrics(s)
   async def item(i):
    async with sem:
     body={"model":"pdshared","prompt":prompt(rows[i],i,run_id),"temperature":0,"max_tokens":nout,"ignore_eos":True,"stream":False}
     params={"do_remote_decode":True,"do_remote_prefill":False,"remote_engine_id":None,"remote_block_ids":None,"remote_host":None,"remote_port":None}
     t0=time.perf_counter()
     p=await post(18100,dict(body,max_tokens=1,kv_transfer_params=params))
     t1=time.perf_counter()
     kv=p.get("kv_transfer_params")
     if not kv or not kv.get("remote_block_ids"):raise RuntimeError("KV missing")
     d=await post(18200,dict(body,kv_transfer_params=kv))
     t2=time.perf_counter()
     records[i]={"p_s":t1-t0,"d_including_kv_s":t2-t1,"total_s":t2-t0,"output_tokens":d["usage"]["completion_tokens"],"text":d["choices"][0]["text"]}
   t0=time.perf_counter()
   await asyncio.gather(*(item(i) for i in range(count)))
   wall=time.perf_counter()-t0
   await asyncio.sleep(2)
   after=await metrics(s)
   def stat(key):
    seq=sorted(x[key] for x in records)
    return {"p50_ms":statistics.median(seq)*1000,"p95_ms":seq[min(len(seq)-1,int(len(seq)*.95))]*1000}
   delta={k:after[k]-before[k] for k in before}
   summary={"concurrency":concurrency,"max_tokens":nout,"run_id":run_id,"count":count,"wall_s":wall,"throughput_tokens_s":sum(x["output_tokens"] for x in records)/wall,"p":stat("p_s"),"d_including_kv":stat("d_including_kv_s"),"total":stat("total_s"),"metrics_delta":delta}
   print(json.dumps(summary),flush=True)
   return {"summary":summary,"records":records}
  result=[]
  for concurrency in (1,8):
   for nout in (1,128):
    await run(concurrency,nout,f"warm-{concurrency}-{nout}",8)
    for rep in range(3):result.append(await run(concurrency,nout,f"measured-{concurrency}-{nout}-{rep}",64))
  (ROOT/"results/pd-detailed-profile.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
asyncio.run(main())