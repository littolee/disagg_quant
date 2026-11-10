import asyncio,aiohttp,json,time,hashlib,statistics,sys
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
root=Path('/home/litto/work/disagg_quant'); model='/home/litto/work/models/Qwen/Qwen3-0.6B'
mode=sys.argv[1]; port=int(sys.argv[2]); out=root/'results/latency-breakdown/gpu-run'; out.mkdir(parents=True,exist_ok=True)
tok=AutoTokenizer.from_pretrained(model)
rows=pq.read_table(root/'data/TIGER-Lab__MMLU-Pro/test-00000-of-00001.parquet').to_pylist(); cats={}
for row in rows:
 if len(cats.setdefault(row['category'],[]))<20: cats[row['category']].append(row)
rows=[r for k in sorted(cats) for r in cats[k]][:32]
def make_prompt(i,run):
 row=rows[i%len(rows)]; nonce=hashlib.sha256(f'latencytrace-gpu:{run}:{i}'.encode()).hexdigest()
 options='\n'.join(f'{chr(65+j)}. {v}' for j,v in enumerate(row['options']) if v)
 return tok.apply_chat_template([{'role':'user','content':nonce+'\n'+row['question']+'\n'+options+'\nAnswer with the correct option.'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
async def main():
 timeout=aiohttp.ClientTimeout(total=120)
 async with aiohttp.ClientSession(timeout=timeout) as session:
  async def sample(i,run):
   body={'model': 'pdshared' if mode=='pd' else 'bf16','prompt':make_prompt(i,run),'max_tokens':1,'temperature':0,'ignore_eos':True,'stream':False}
   start=time.monotonic_ns()
   async with session.post(f'http://127.0.0.1:{port}/v1/completions',json=body) as res:
    v=await res.json(); end=time.monotonic_ns()
    if res.status!=200: raise RuntimeError(str(v))
   return {'id':v['id'],'start':start,'end':end,'prompt_tokens':v['usage']['prompt_tokens'],'text':v['choices'][0]['text']}
  for i in range(16): await sample(i,'warm')
  records=[]
  for i in range(96): records.append(await sample(i,'measured'))
  path=out/f'{mode}-client.json'; path.write_text(json.dumps(records,ensure_ascii=False))
  x=sorted((v['end']-v['start'])/1e6 for v in records)
  print(json.dumps({'mode':mode,'n':len(x),'p50_ms':statistics.median(x),'p95_ms':x[int(len(x)*.95)],'mean_ms':statistics.mean(x),'path':str(path)}),flush=True)
asyncio.run(main())
