import asyncio,aiohttp,json,time,statistics,subprocess,os,signal,hashlib,random
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
ROOT=Path("/home/litto/work/disagg_quant")
MODEL="/home/litto/work/models/Qwen/Qwen3-0.6B"
OUT=ROOT/"results/nixl-graph-benchmark"
OUT.mkdir(parents=True,exist_ok=True)
tok=AutoTokenizer.from_pretrained(MODEL)
rows=pq.read_table(ROOT/"data/TIGER-Lab__MMLU-Pro/test-00000-of-00001.parquet").to_pylist()
by={}
for row in rows:
 key=row["category"]
 if len(by.setdefault(key,[]))<20: by[key].append(row)
rows=[row for k in sorted(by) for row in by[k]]
assert len(rows)==280
def prompt(row,i,round_id):
 nonce=hashlib.sha256(f"{round_id}:{i}".encode()).hexdigest()
 options="\n".join(f"{chr(65+j)}. {v}" for j,v in enumerate(row["options"]) if v)
 text=nonce+"\n"+row["question"]+"\n"+options+"\nAnswer with the correct option."
 return tok.apply_chat_template([{"role":"user","content":text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
async def ready(session,port):
 for _ in range(180):
  try:
   async with session.get(f"http://127.0.0.1:{port}/health",timeout=aiohttp.ClientTimeout(total=2)) as resp:
    if resp.status==200:return
  except Exception:pass
  await asyncio.sleep(2)
 raise RuntimeError(f"server {port} failed to start")
async def run(session,mode,port,subset,round_id,max_tokens,concurrency):
 sem=asyncio.Semaphore(concurrency)
 result=[None]*len(subset)
 async def item(j,row):
  async with sem:
   body={"model":mode,"prompt":prompt(row,j,round_id),"max_tokens":max_tokens,"temperature":0,"ignore_eos":True,"stream":False}
   start=time.monotonic()
   async with session.post(f"http://127.0.0.1:{port}/v1/completions",json=body,timeout=aiohttp.ClientTimeout(total=600)) as resp:
    data=await resp.json()
    if resp.status!=200: raise RuntimeError(f"{mode} {resp.status} {data}")
   elapsed=time.monotonic()-start
   result[j]={"seconds":elapsed,"completion_tokens":data["usage"]["completion_tokens"],"prompt_tokens":data["usage"]["prompt_tokens"],"text":data["choices"][0]["text"]}
 start=time.monotonic()
 await asyncio.gather(*(item(j,row) for j,row in enumerate(subset)))
 elapsed=time.monotonic()-start
 lat=sorted(x["seconds"] for x in result)
 total=sum(x["completion_tokens"] for x in result)
 return {"mode":mode,"round":round_id,"count":len(result),"max_tokens":max_tokens,"elapsed_s":elapsed,"items_per_s":len(result)/elapsed,"output_tokens_per_s":total/elapsed,"completion_tokens":total,"p50_s":statistics.median(lat),"p95_s":lat[min(len(lat)-1,int(len(lat)*.95))],"samples":result}
async def main():
 timeout=aiohttp.ClientTimeout(total=600)
 async with aiohttp.ClientSession(timeout=timeout) as session:
  await ready(session,18140);await ready(session,18240)
  configs=[("pdshared",18041,None)]
  for mode,port,model in configs:
   proc=None
   try:
    if model:
     env=os.environ.copy()
     env["PATH"]=str(ROOT/".venv-eval/bin")+":"+env["PATH"]
     env["MAX_JOBS"]="2"
     args=[str(ROOT/".venv-eval/bin/vllm"),"serve",model,"--tokenizer",MODEL,"--served-model-name",mode,"--host","127.0.0.1","--port",str(port),"--enforce-eager","--gpu-memory-utilization","0.12","--kv-cache-memory-bytes","2G","--max-model-len","8192","--max-num-seqs","8","--max-num-batched-tokens","8192","--no-enable-prefix-caching"]
     log=open(ROOT/f"logs/warm-benchmark-{mode}-server.log","w")
     proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
     await ready(session,port)
    print("START",mode,flush=True)
    for w in range(2):
     result=await run(session,mode,port,rows[:16],f"warm{w}",128,8)
     print("WARM",mode,w,result["elapsed_s"],flush=True)
    for rep in range(3):
     result=await run(session,mode,port,rows,f"measured{rep}",128,8)
     (OUT/f"{mode}-round{rep}.json").write_text(json.dumps(result,ensure_ascii=False))
     print("RESULT",json.dumps({k:v for k,v in result.items() if k!="samples"}),flush=True)
    for rep in range(3):
     result=await run(session,mode,port,rows[:32],f"first{rep}",1,1)
     (OUT/f"{mode}-first{rep}.json").write_text(json.dumps(result,ensure_ascii=False))
     print("FIRST",json.dumps({k:v for k,v in result.items() if k!="samples"}),flush=True)
   finally:
    if proc:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=30)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
     log.close()
if __name__=="__main__":asyncio.run(main())